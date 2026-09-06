"""
scripts/squim_warmstart_check.py

Paso 2 del mini-sweep de calibración de V4 (docs/PLAN_V4.md Fase 3):
prueba warm-start (1 época mse_magnitude + 2 épocas mse_plus_squim)
SOLO sobre el ganador de Paso 1 (o el alpha que se pase por --alpha),
y lo compara contra el resultado sin warm-start del mismo alpha (ya
evaluado en Paso 1, no se re-corre).

Por defecto lee el ganador de results/v4_sweep/paso1_comparison.json.
Correr scripts.squim_scale_sweep_evaluate ANTES que este script.

Implementación: reutiliza init_checkpoint (ya existe en trainer.py,
usado desde V3) -- no hace falta código nuevo en el trainer.

USO:
    python -m scripts.squim_warmstart_check
    python -m scripts.squim_warmstart_check --alpha 0.9   # override manual
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from training.config import CONFIG_V1
from training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "checkpoints" / "v4_sweep"
RESULTS_DIR = PROJECT_ROOT / "results" / "v4_sweep"
TEST_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"

N_EPOCHS_WARMUP = 1
N_EPOCHS_SQUIM = 2  # total 3, mismo presupuesto que Paso 1 (ver docs/decisions.md 30/08/2026)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _pick_winner_alpha() -> float:
    paso1_path = RESULTS_DIR / "paso1_comparison.json"
    if not paso1_path.exists():
        raise SystemExit(
            f"No existe {paso1_path}. Correr scripts.squim_scale_sweep_evaluate primero, "
            "o pasar --alpha explícitamente."
        )
    with open(paso1_path) as f:
        paso1 = json.load(f)
    validos = paso1["validos_por_criterio"]
    if not validos:
        raise SystemExit(
            "Ningún candidato de Paso 1 pasó el criterio -- no hay ganador para warm-start. "
            "Ver docs/PLAN_V4.md, esto es un resultado documentable en sí mismo."
        )
    candidatos = {c["nombre"]: c for c in paso1["candidatos"]}
    winner_nombre = max(validos, key=lambda n: candidatos[n]["pesq_nb"])
    alpha = int(winner_nombre.replace("alpha_", "")) / 100
    logger.info(f"Ganador de Paso 1: {winner_nombre} (alpha={alpha})")
    return alpha


def run_warmstart(alpha: float):
    alpha_str = f"{alpha:.2f}".replace(".", "")
    warmup_dir = SWEEP_DIR / f"warmstart_alpha_{alpha_str}_warmup"
    main_dir = SWEEP_DIR / f"warmstart_alpha_{alpha_str}"

    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"Warmup: {N_EPOCHS_WARMUP} época(s) mse_magnitude")
    logger.info("=" * 70)
    warmup_config = {
        "train_dir": CONFIG_V1["train_dir"],
        "val_dir": CONFIG_V1["val_dir"],
        "loss": "mse_magnitude",
        "batch_size": 4,
        "lr": 2e-4,
        "n_epochs": N_EPOCHS_WARMUP,
        "seed": 42,
        "train_shuffle": True,
        "val_shuffle": False,
        "cudnn_deterministic": False,
        "checkpoint_dir": warmup_dir,
    }
    Trainer(warmup_config).fit()

    logger.info("=" * 70)
    logger.info(f"Squim: {N_EPOCHS_SQUIM} épocas mse_plus_squim, alpha={alpha}, "
                f"init_checkpoint={warmup_dir / 'best.pt'}")
    logger.info("=" * 70)
    main_config = {
        "train_dir": CONFIG_V1["train_dir"],
        "val_dir": CONFIG_V1["val_dir"],
        "loss": "mse_plus_squim",
        "squim_alpha": alpha,
        "squim_scale": None,
        "init_checkpoint": warmup_dir / "best.pt",
        "batch_size": 4,
        "lr": 2e-4,
        "n_epochs": N_EPOCHS_SQUIM,
        "seed": 42,
        "train_shuffle": True,
        "val_shuffle": False,
        "cudnn_deterministic": False,
        "checkpoint_dir": main_dir,
    }
    Trainer(main_config).fit()

    elapsed = time.time() - t0
    logger.info(f"Paso 2 completo en {elapsed/60:.1f} min")
    return main_dir


def evaluate_checkpoint(checkpoint_dir: Path) -> dict:
    output_json = RESULTS_DIR / f"{checkpoint_dir.name}_v1_en.json"
    cmd = [
        sys.executable, "-m", "evaluation.evaluate_variant",
        "--variant", checkpoint_dir.name,
        "--checkpoint", str(checkpoint_dir / "best.pt"),
        "--test_dir", str(TEST_DIR),
        "--metadata", str(METADATA),
        "--output", str(output_json),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(output_json) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override manual del alpha ganador de Paso 1")
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else _pick_winner_alpha()

    main_dir = run_warmstart(alpha)

    logger.info(f"Evaluando {main_dir.name} sobre test_v1_en")
    eval_result = evaluate_checkpoint(main_dir)
    g = eval_result["global"]
    warmstart_row = {
        "nombre": main_dir.name,
        "alpha": alpha,
        "pesq_nb": g["pesq_nb"]["est_mean"],
        "pesq_wb": g["pesq_wb"]["est_mean"],
        "stoi": g["stoi"]["est_mean"],
        "sisdr": g["sisdr"]["est_mean"],
    }

    # Comparar contra el resultado sin warm-start del mismo alpha (Paso 1, ya evaluado)
    alpha_str = f"{alpha:.2f}".replace(".", "")
    no_warmstart_path = RESULTS_DIR / f"alpha_{alpha_str}_v1_en.json"
    no_warmstart_row = None
    if no_warmstart_path.exists():
        with open(no_warmstart_path) as f:
            no_ws = json.load(f)
        g2 = no_ws["global"]
        no_warmstart_row = {
            "nombre": f"alpha_{alpha_str} (sin warm-start, Paso 1)",
            "pesq_nb": g2["pesq_nb"]["est_mean"],
            "pesq_wb": g2["pesq_wb"]["est_mean"],
            "stoi": g2["stoi"]["est_mean"],
            "sisdr": g2["sisdr"]["est_mean"],
        }

    output_path = RESULTS_DIR / "paso2_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "alpha": alpha,
            "con_warmstart": warmstart_row,
            "sin_warmstart": no_warmstart_row,
        }, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info(f"RESULTADOS PASO 2 -- alpha={alpha}, test_v1_en")
    logger.info("=" * 90)
    header = f"{'nombre':<38} | {'PESQ-NB':<8} | {'PESQ-WB':<8} | {'STOI':<7} | {'SI-SDR':<8}"
    logger.info(header)
    logger.info("-" * 90)
    if no_warmstart_row:
        logger.info(f"{no_warmstart_row['nombre']:<38} | {no_warmstart_row['pesq_nb']:<8.3f} | "
                    f"{no_warmstart_row['pesq_wb']:<8.3f} | {no_warmstart_row['stoi']:<7.3f} | "
                    f"{no_warmstart_row['sisdr']:<8.2f}")
    logger.info(f"{warmstart_row['nombre']:<38} | {warmstart_row['pesq_nb']:<8.3f} | "
                f"{warmstart_row['pesq_wb']:<8.3f} | {warmstart_row['stoi']:<7.3f} | "
                f"{warmstart_row['sisdr']:<8.2f}")
    logger.info("=" * 90)
    logger.info(f"\nResumen guardado en {output_path}")
    logger.info("Registrar decisión final (con/sin warm-start) en docs/decisions.md.")


if __name__ == "__main__":
    main()
