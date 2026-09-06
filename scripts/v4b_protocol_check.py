"""
scripts/v4b_protocol_check.py

V4b: ¿el protocolo estabilizado de Xu et al. 2022 cambia el modo de falla
de `mse_plus_squim`, sin reentrenar el proxy?

V4 (Fase 3, 30-31/08/2026) corrió el régimen máximamente inestable —
init random, lr 2e-4, update por minibatch — y Squim se gameó desde la
época 1 en las 4 corridas (estimaba PESQ 4.0-4.4 mientras el PESQ real
caía por debajo de Noisy). Xu et al. 2022 (`Papers/Deep Noise Suppression
Maximizing Non-Differentiable PESQ...`, sección IV-B) usa tres elementos
estabilizadores; dos de ellos NO requieren reentrenar el proxy y nunca
los probamos:

  1. Arranque desde un modelo ya convergido con MSE (no init random).
  2. lr 10x más bajo en la etapa perceptual (Xu usa 2e-5).
  3. Gradient accumulation a UN update de pesos por época. -- DESCARTADO
     acá: con 3 épocas serían 3 updates totales partiendo de V1
     convergido, no se movería nada y el nulo sería trivial.
     Incompatible con este presupuesto, no es una omisión por descuido.

El cuarto elemento de Xu (alternancia <1-1> reentrenando el PESQNet) ya
fue evaluado y descartado por alcance -- ver docs/decisions.md 30/08/2026.

DISEÑO (2x2, tres celdas; la cuarta -- init random + lr 2e-5 -- es inútil
porque casi no entrenaría):

                    | lr = 2e-4              | lr = 2e-5
    init random     | alpha_090 (YA CORRIDO) | --
    init V1 conv.   | v4b_lr2e4              | v4b_main

`v4b_lr2e4` desconfunde "arranque pre-entrenado" de "lr bajo" -- la misma
lección que dejó V3b (épocas y lr confundidos, ver CLAUDE.md) -- y sirve
de control positivo: es el brazo más propenso a gamear. Si ni ése gamea,
el arranque pre-entrenado solo ya es protector.

`v4b_placebo` es el control que el propio Xu usa (su "placebo setup",
alpha=1): mismo protocolo de fine-tuning, sin término perceptual. Sin él,
cualquier mejora sería inatribuible entre "el término Squim" y "seguir
entrenando un modelo ya convergido con lr bajo".

3 épocas por corrida para igualar el presupuesto de `alpha_090`, la celda
del 2x2 que ya está corrida. `save_every_n_epochs=1` porque `best.pt` se
elige por `val_loss`, que para `mse_plus_squim` INCLUYE el término Squim:
más gameado = menor val_loss = "mejor" checkpoint. Hay que evaluar las
tres épocas por separado, no el best.

Presupuesto: ~8.2h (main ~3.3h + placebo ~1.6h + lr2e4 ~3.3h), medido
sobre el sweep de Fase 3 (~65 min/época con Squim, ~32 min/época sin).

USO:
    setsid nohup python -m scripts.v4b_protocol_check > logs/v4b_protocol.log 2>&1 < /dev/null &
    # OJO: `$!` NO devuelve el PID real con este patrón (setsid forkea).
    # Verificar con: pgrep -af v4b_protocol_check
    # Ver CLAUDE.md, "Gotcha del PID capturado con $!".

    # Después:
    python -m scripts.v4b_protocol_check_evaluate
"""
import json
import logging
import time
from pathlib import Path

from training.config import CONFIG_V1
from training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_EPOCHS = 3
ALPHA = 0.9          # misma celda que alpha_090 del sweep de Fase 3
LR_XU = 2e-5         # lr de la 2da etapa de fine-tuning de Xu et al. 2022
LR_PROJECT = 2e-4    # lr histórico del proyecto (V1..V3, y el sweep de V4)
INIT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "v1" / "best.pt"
RUN_DIR = PROJECT_ROOT / "checkpoints" / "v4b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _base_config():
    """Base compartida: dataset EN completo (mismo que V1), lr fijo sin
    scheduler, arranque desde V1 convergido, checkpoint por época.

    `cudnn_deterministic: False` para poder comparar contra `alpha_090`,
    que se corrió así en el sweep de Fase 3 -- la comparabilidad con la
    celda ya existente del 2x2 pesa más que la bit-exactitud acá, y estas
    corridas siguen siendo exploratorias. Se documenta como limitación.
    """
    return {
        "train_dir": CONFIG_V1["train_dir"],
        "val_dir": CONFIG_V1["val_dir"],
        "init_checkpoint": INIT_CHECKPOINT,
        "batch_size": 4,
        "n_epochs": N_EPOCHS,
        "seed": 42,
        "train_shuffle": True,
        "val_shuffle": False,
        "cudnn_deterministic": False,
        "save_every_n_epochs": 1,  # esquiva el sesgo de selección de best.pt
        # sin scheduler_step/scheduler_gamma -> lr fijo
    }


# Orden deliberado: `main` primero (es el titular; si gamea fuerte en
# época 1 ya sabemos el desenlace), después el placebo, y el control
# positivo al final.
RUNS = [
    ("v4b_main", {
        "loss": "mse_plus_squim", "squim_alpha": ALPHA,
        "squim_scale": None, "lr": LR_XU,
    }),
    ("v4b_placebo", {
        "loss": "mse_magnitude", "lr": LR_XU,
    }),
    ("v4b_lr2e4", {
        "loss": "mse_plus_squim", "squim_alpha": ALPHA,
        "squim_scale": None, "lr": LR_PROJECT,
    }),
]


def _run(nombre, extra_config, results_summary):
    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"Corrida: {nombre}")
    logger.info("=" * 70)

    config = {**_base_config(), **extra_config}
    config["checkpoint_dir"] = RUN_DIR / nombre

    try:
        Trainer(config).fit()
        elapsed = time.time() - t0
        logger.info(f"{nombre} completa en {elapsed/60:.1f} min")
        results_summary.append({
            "nombre": nombre,
            "checkpoint_dir": str(config["checkpoint_dir"]),
            "loss": config["loss"],
            "lr": config["lr"],
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


def main():
    if not INIT_CHECKPOINT.exists():
        raise SystemExit(f"No existe {INIT_CHECKPOINT} -- V4b arranca desde V1 convergido.")

    t_start = time.time()
    logger.info(f"V4b: protocolo estabilizado de Xu, {len(RUNS)} corridas x {N_EPOCHS} épocas")
    logger.info(f"init_checkpoint = {INIT_CHECKPOINT}")

    results_summary = []
    for nombre, extra in RUNS:
        _run(nombre, extra, results_summary)

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"V4b completo en {t_total/3600:.2f} horas")
    logger.info("=" * 70)

    summary_path = PROJECT_ROOT / "results" / "v4b" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "n_epochs": N_EPOCHS,
            "alpha": ALPHA,
            "init_checkpoint": str(INIT_CHECKPOINT),
            "total_time_hours": t_total / 3600,
            "runs": results_summary,
        }, f, indent=2, default=str)
    logger.info(f"Resumen en {summary_path}")
    logger.info("Siguiente paso: python -m scripts.v4b_protocol_check_evaluate")


if __name__ == "__main__":
    main()
