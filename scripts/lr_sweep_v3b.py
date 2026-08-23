"""
scripts/lr_sweep_v3b.py

Barrido logarítmico de learning rate para V3b (fine-tuning conservador
de V1 sobre Common Voice ES). 5 puntos entre 5e-6 y 2e-4, 5 épocas cada
uno, lr fijo (sin StepLR) para medir el efecto de cada lr de forma aislada.

Cada corrida guarda checkpoint en checkpoints/v3_sweep/lr_<lr>/
Evaluación posterior en scripts/lr_sweep_v3b_evaluate.py

Ver docs/PLAN_V3B.md (en noisesuppressnet-ai) para el diseño completo,
y docs/decisions.md 19-21/08/2026 para el contexto metodológico (por qué
V3 con full fine-tuning agresivo mostró catastrophic forgetting en
inglés, y qué se espera que corrija un lr más conservador).

USO:
    python -m scripts.lr_sweep_v3b

    # En background (usar setsid, no nohup suelto -- ver docs/decisions.md
    # 19/08/2026 sobre por qué nohup solo no sobrevive el cierre de terminal):
    setsid nohup python -m scripts.lr_sweep_v3b > logs/lr_sweep.log 2>&1 < /dev/null &
    echo $! > logs/lr_sweep.pid
"""
import copy
import json
import logging
import time
from pathlib import Path

from training.config import CONFIG_V3_SWEEP_BASE
from training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LR_CANDIDATES = [5e-6, 2e-5, 5e-5, 1e-4, 2e-4]
N_EPOCHS_SWEEP = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _lr_str(lr: float) -> str:
    """Formatea un lr para nombres de directorio (ej: 2e-05 -> '2e-5')."""
    return f"{lr:.0e}".replace("+0", "").replace("-0", "-")


def run_sweep():
    """Ejecuta el sweep completo de learning rates."""
    t_start = time.time()
    logger.info(f"Iniciando sweep con {len(LR_CANDIDATES)} learning rates")
    logger.info(f"LR candidatos: {LR_CANDIDATES}")
    logger.info(f"Épocas por corrida: {N_EPOCHS_SWEEP}")

    init_ckpt = CONFIG_V3_SWEEP_BASE["init_checkpoint"]
    if not Path(init_ckpt).exists():
        logger.error(f"No existe {init_ckpt}, abortando")
        return

    results_summary = []

    for i, lr in enumerate(LR_CANDIDATES, 1):
        t_run_start = time.time()
        lr_str = _lr_str(lr)

        logger.info("=" * 70)
        logger.info(f"Corrida {i}/{len(LR_CANDIDATES)}: lr={lr:.2e}")
        logger.info("=" * 70)

        config = copy.deepcopy(CONFIG_V3_SWEEP_BASE)
        config["variant"] = f"V3_sweep_lr{lr_str}"
        config["lr"] = lr
        config["n_epochs"] = N_EPOCHS_SWEEP
        config["checkpoint_dir"] = PROJECT_ROOT / "checkpoints" / "v3_sweep" / f"lr_{lr_str}"
        # Sin scheduler_step/scheduler_gamma: Trainer usa lr fijo (ver training/trainer.py)

        try:
            trainer = Trainer(config)
            trainer.fit()

            t_run = time.time() - t_run_start
            logger.info(f"Corrida {i} completa en {t_run/60:.1f} min")

            results_summary.append({
                "lr": lr,
                "checkpoint_dir": str(config["checkpoint_dir"]),
                "time_min": t_run / 60,
                "status": "success",
            })
        except Exception as e:
            logger.error(f"Corrida {i} falló: {e}")
            results_summary.append({
                "lr": lr,
                "checkpoint_dir": str(config["checkpoint_dir"]),
                "status": "failed",
                "error": str(e),
            })

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"Sweep completo en {t_total/3600:.2f} horas")
    logger.info("=" * 70)

    summary_path = PROJECT_ROOT / "results" / "v3_sweep" / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "lr_candidates": LR_CANDIDATES,
            "n_epochs": N_EPOCHS_SWEEP,
            "total_time_hours": t_total / 3600,
            "runs": results_summary,
        }, f, indent=2, default=str)
    logger.info(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    run_sweep()
