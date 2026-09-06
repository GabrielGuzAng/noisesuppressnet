"""
scripts/squim_scale_sweep.py

Paso 1 del mini-sweep de calibración de V4 (docs/PLAN_V4.md Fase 3):
barrido de `alpha` para `mse_plus_squim`, sin warm-start, más una corrida
de control (`mse_magnitude` puro, equivalente a alpha=1.0) al mismo
presupuesto de épocas. Dataset completo (EN, mismo que V1), lr fijo (sin
StepLR) para medir el efecto de cada alpha de forma aislada, sin decay
que lo enmascare.

`squim_scale` queda fijo en None (=1.0 sin escalar) para las 5 corridas:
el smoke test de Fase 2 mostró mse_component y squim_component en el
mismo orden de magnitud (~0.5-1.5 vs ~-1.1), a diferencia de SI-SDR en
V2 que sí necesitó sisdr_scale=0.03 por una diferencia de escala grande.
Si los resultados de este sweep muestran que un término domina al otro
de forma no intencional, se agrega un sub-sweep de squim_scale después
-- no se asume de entrada.

Cada corrida guarda checkpoint en checkpoints/v4_sweep/<nombre>/
Evaluación posterior en scripts/squim_scale_sweep_evaluate.py

Ver docs/PLAN_V4.md Fase 3 para el diseño completo y docs/decisions.md
30/08/2026 (dos entradas: diseño de V4, y por qué se descartó el
esquema alternado de Xu) para el contexto metodológico.

USO:
    python -m scripts.squim_scale_sweep

    # En background (setsid, no nohup suelto -- ver docs/decisions.md
    # 19/08/2026 sobre por qué nohup solo no sobrevive el cierre de
    # terminal). Presupuesto esperado: ~11.1h (1 control ~1.2h + 4
    # candidatos ~9.9h, N_EPOCHS_SWEEP=3 -- ver docs/decisions.md 30/08/2026,
    # "Paso 1 de V4 acotado a 3 épocas").
    setsid nohup python -m scripts.squim_scale_sweep > logs/squim_sweep.log 2>&1 < /dev/null &
    echo $! > logs/squim_sweep.pid
"""
import copy
import json
import logging
import time
from pathlib import Path

from training.config import CONFIG_V1
from training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALPHA_CANDIDATES = [0.95, 0.9, 0.8, 0.7]
N_EPOCHS_SWEEP = 3
SWEEP_DIR = PROJECT_ROOT / "checkpoints" / "v4_sweep"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _base_config():
    """Config base compartida por control y candidatos: dataset EN
    completo (mismo que V1), lr fijo sin scheduler, sweep descartable."""
    return {
        "train_dir": CONFIG_V1["train_dir"],
        "val_dir": CONFIG_V1["val_dir"],
        "batch_size": 4,
        "lr": 2e-4,
        "n_epochs": N_EPOCHS_SWEEP,
        "seed": 42,
        "train_shuffle": True,
        "val_shuffle": False,
        "cudnn_deterministic": False,  # corrida exploratoria descartable
        # sin scheduler_step/scheduler_gamma -> lr fijo (ver trainer.py)
    }


def _run(nombre, extra_config, results_summary):
    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"Corrida: {nombre}")
    logger.info("=" * 70)

    config = {**_base_config(), **extra_config}
    config["checkpoint_dir"] = SWEEP_DIR / nombre

    try:
        trainer = Trainer(config)
        trainer.fit()
        elapsed = time.time() - t0
        logger.info(f"{nombre} completa en {elapsed/60:.1f} min")
        results_summary.append({
            "nombre": nombre,
            "checkpoint_dir": str(config["checkpoint_dir"]),
            "time_min": elapsed / 60,
            "status": "success",
        })
    except Exception as e:
        logger.error(f"{nombre} falló: {e}")
        results_summary.append({
            "nombre": nombre,
            "checkpoint_dir": str(config["checkpoint_dir"]),
            "status": "failed",
            "error": str(e),
        })


def run_sweep():
    t_start = time.time()
    logger.info(f"Iniciando Paso 1 del sweep V4: control + {len(ALPHA_CANDIDATES)} alphas")
    logger.info(f"Alphas: {ALPHA_CANDIDATES}, épocas por corrida: {N_EPOCHS_SWEEP}")

    results_summary = []

    # Control: sin término Squim, mismo presupuesto de épocas que los candidatos
    _run("control_mse", {"loss": "mse_magnitude"}, results_summary)

    for alpha in ALPHA_CANDIDATES:
        alpha_str = f"{alpha:.2f}".replace(".", "")
        _run(
            f"alpha_{alpha_str}",
            {"loss": "mse_plus_squim", "squim_alpha": alpha, "squim_scale": None},
            results_summary,
        )

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"Paso 1 completo en {t_total/3600:.2f} horas")
    logger.info("=" * 70)

    summary_path = PROJECT_ROOT / "results" / "v4_sweep" / "paso1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "alpha_candidates": ALPHA_CANDIDATES,
            "n_epochs": N_EPOCHS_SWEEP,
            "total_time_hours": t_total / 3600,
            "runs": results_summary,
        }, f, indent=2, default=str)
    logger.info(f"Resumen guardado en {summary_path}")
    logger.info("Siguiente paso: python -m scripts.squim_scale_sweep_evaluate")


if __name__ == "__main__":
    run_sweep()
