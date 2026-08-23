"""
scripts/lr_sweep_v3b_evaluate.py

Evalúa cada checkpoint del sweep de V3b (scripts/lr_sweep_v3b.py) sobre
ambos test sets sellados:
- test_v1_en (mide catastrophic forgetting en inglés)
- test_v2_es (mide ganancia en español)

Genera tabla comparativa con score compuesto para elegir el lr ganador.
Usa evaluation/evaluate_variant.py vía subprocess con --checkpoint/--output
explícitos, porque los checkpoints del sweep no siguen la convención
checkpoints/<variant>/best.pt.

USO:
    python -m scripts.lr_sweep_v3b_evaluate
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "checkpoints" / "v3_sweep"
RESULTS_DIR = PROJECT_ROOT / "results" / "v3_sweep"

# Baselines para calcular deltas — de results/v1_v1_en.json y results/v1_v2_es.json
V1_ON_TEST_EN_PESQ_NB = 2.650
V1_ON_TEST_ES_PESQ_NB = 2.330

# Peso para el score compuesto (lambda)
FORGETTING_WEIGHT = 1.0  # score = ΔPESQ_ES + λ × ΔPESQ_EN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(checkpoint_dir: Path, test_type: str) -> dict:
    """Evalúa un checkpoint del sweep sobre un test set, vía evaluate_variant.py.

    Args:
        checkpoint_dir: directorio del checkpoint (con best.pt), ej.
            checkpoints/v3_sweep/lr_5e-6/
        test_type: 'v1_en' o 'v2_es'
    """
    if test_type == "v1_en":
        test_dir = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
        metadata = PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"
    elif test_type == "v2_es":
        test_dir = PROJECT_ROOT / "data" / "test_sealed" / "v2_es"
        metadata = PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json"
    else:
        raise ValueError(f"test_type inválido: {test_type}")

    output_json = RESULTS_DIR / f"{checkpoint_dir.name}_{test_type}.json"

    logger.info(f"  Evaluando {checkpoint_dir.name} sobre {test_type}")

    cmd = [
        sys.executable, "-m", "evaluation.evaluate_variant",
        "--variant", checkpoint_dir.name,
        "--checkpoint", str(checkpoint_dir / "best.pt"),
        "--test_dir", str(test_dir),
        "--metadata", str(metadata),
        "--output", str(output_json),
    ]

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    with open(output_json) as f:
        return json.load(f)


def extract_pesq_nb(eval_result: dict) -> float:
    """PESQ-NB absoluto del modelo evaluado (no el delta).

    La estructura real de evaluate_variant.py anida las métricas bajo
    "global" -> "<metrica>" -> "est_mean" (verificado contra
    results/v1_v1_en.json). No hay una clave "pesq_nb" de primer nivel.
    """
    return eval_result["global"]["pesq_nb"]["est_mean"]


def compute_score(pesq_es: float, pesq_en: float, lambda_forget: float = FORGETTING_WEIGHT) -> float:
    """
    Score compuesto: ΔPESQ_ES + λ × ΔPESQ_EN

    ΔPESQ_ES: ganancia vs V1 sobre test_v2_es (positivo = mejor)
    ΔPESQ_EN: cambio vs V1 sobre test_v1_en (negativo = forgetting)
    """
    delta_es = pesq_es - V1_ON_TEST_ES_PESQ_NB
    delta_en = pesq_en - V1_ON_TEST_EN_PESQ_NB
    return delta_es + lambda_forget * delta_en


def main():
    if not SWEEP_DIR.exists():
        logger.error(f"No existe {SWEEP_DIR}. Correr scripts.lr_sweep_v3b primero.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dirs = sorted([d for d in SWEEP_DIR.iterdir() if d.is_dir() and d.name.startswith("lr_")])
    logger.info(f"Encontrados {len(checkpoint_dirs)} checkpoints del sweep")

    results_table = []

    for ckpt_dir in checkpoint_dirs:
        if not (ckpt_dir / "best.pt").exists():
            logger.warning(f"No hay best.pt en {ckpt_dir}, se salta")
            continue

        lr_str = ckpt_dir.name.replace("lr_", "")
        lr = float(lr_str)

        logger.info(f"Evaluando lr={lr:.2e}")

        history_path = ckpt_dir / "history.json"
        val_loss_min = None
        best_epoch = None
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
                val_losses = history.get("val_loss", [])
                if val_losses:
                    val_loss_min = min(val_losses)
                    best_epoch = val_losses.index(val_loss_min) + 1

        result_es = evaluate_checkpoint(ckpt_dir, "v2_es")
        result_en = evaluate_checkpoint(ckpt_dir, "v1_en")

        pesq_es = extract_pesq_nb(result_es)
        pesq_en = extract_pesq_nb(result_en)

        delta_es = pesq_es - V1_ON_TEST_ES_PESQ_NB
        delta_en = pesq_en - V1_ON_TEST_EN_PESQ_NB
        score = compute_score(pesq_es, pesq_en)

        results_table.append({
            "lr": lr,
            "lr_str": f"{lr:.0e}",
            "val_loss_min": val_loss_min,
            "best_epoch": best_epoch,
            "pesq_es": pesq_es,
            "pesq_en": pesq_en,
            "delta_pesq_es": delta_es,
            "delta_pesq_en": delta_en,
            "score": score,
        })

    output_path = RESULTS_DIR / "sweep_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "lambda_forgetting": FORGETTING_WEIGHT,
            "v1_baseline_pesq_es": V1_ON_TEST_ES_PESQ_NB,
            "v1_baseline_pesq_en": V1_ON_TEST_EN_PESQ_NB,
            "results": results_table,
        }, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("RESULTADOS DEL SWEEP")
    logger.info("=" * 80)
    header = f"{'lr':<10} | {'val_loss':<10} | {'best_ep':<8} | {'ΔPESQ_ES':<10} | {'ΔPESQ_EN':<10} | {'Score':<8}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results_table:
        vl = f"{r['val_loss_min']:.4f}" if r["val_loss_min"] is not None else "N/A"
        logger.info(
            f"{r['lr_str']:<10} | "
            f"{vl:<10} | "
            f"{str(r['best_epoch']):<8} | "
            f"{r['delta_pesq_es']:+.3f}    | "
            f"{r['delta_pesq_en']:+.3f}    | "
            f"{r['score']:+.3f}"
        )
    logger.info("=" * 80)

    if results_table:
        winner = max(results_table, key=lambda r: r["score"])
        logger.info(f"\nGanador: lr={winner['lr_str']} con score {winner['score']:+.3f}")
        logger.info(f"   ΔPESQ_ES: {winner['delta_pesq_es']:+.3f}")
        logger.info(f"   ΔPESQ_EN: {winner['delta_pesq_en']:+.3f}")
        logger.info(f"\nUsar este lr para V3b final (10 épocas, con scheduler activado).")


if __name__ == "__main__":
    main()
