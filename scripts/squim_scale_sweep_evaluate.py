"""
scripts/squim_scale_sweep_evaluate.py

Evalúa los checkpoints de Paso 1 del sweep de V4 (scripts/squim_scale_sweep.py)
sobre test_v1_en (250 pares sellados, EN -- V4 es rama EN, no toca español).

Aplica el criterio de selección de docs/PLAN_V4.md Fase 3: el candidato
que más mejora PESQ-NB REAL (no el estimado por Squim) sin que SI-SDR o
STOI reales caigan por debajo del control (mse_magnitude, mismo
presupuesto de épocas). "Real" es la palabra clave -- evita medir el
régimen que se supone hay que vigilar (Squim gameando su propia métrica).

Usa evaluation/evaluate_variant.py vía subprocess (mismo patrón que
scripts/lr_sweep_v3b_evaluate.py).

USO:
    python -m scripts.squim_scale_sweep_evaluate
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "checkpoints" / "v4_sweep"
RESULTS_DIR = PROJECT_ROOT / "results" / "v4_sweep"
TEST_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(checkpoint_dir: Path) -> dict:
    """Evalúa un checkpoint del sweep sobre test_v1_en, vía evaluate_variant.py."""
    output_json = RESULTS_DIR / f"{checkpoint_dir.name}_v1_en.json"

    logger.info(f"  Evaluando {checkpoint_dir.name}")

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


def epoch1_instability(history_path: Path) -> str:
    """Chequeo rápido de inestabilidad en época 1 (NaN o salto grande en
    val_loss) -- insumo para la decisión de Paso 2 (warm-start)."""
    if not history_path.exists():
        return "sin history.json"
    with open(history_path) as f:
        history = json.load(f)
    val_loss = history.get("val_loss", [])
    if not val_loss:
        return "sin val_loss"
    if any(v != v for v in val_loss):  # NaN check
        return "NaN detectado"
    if len(val_loss) >= 2 and val_loss[1] > val_loss[0] * 1.5:
        return f"salto ep1->ep2: {val_loss[0]:.4f} -> {val_loss[1]:.4f}"
    return "sin señales de inestabilidad"


def main():
    if not SWEEP_DIR.exists():
        logger.error(f"No existe {SWEEP_DIR}. Correr scripts.squim_scale_sweep primero.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in SWEEP_DIR.iterdir() if d.is_dir()])
    logger.info(f"Encontrados {len(run_dirs)} checkpoints del sweep")

    results_table = []
    control_row = None

    for run_dir in run_dirs:
        if not (run_dir / "best.pt").exists():
            logger.warning(f"No hay best.pt en {run_dir}, se salta")
            continue

        logger.info(f"Evaluando {run_dir.name}")
        eval_result = evaluate_checkpoint(run_dir)
        g = eval_result["global"]

        row = {
            "nombre": run_dir.name,
            "pesq_nb": g["pesq_nb"]["est_mean"],
            "pesq_wb": g["pesq_wb"]["est_mean"],
            "stoi": g["stoi"]["est_mean"],
            "sisdr": g["sisdr"]["est_mean"],
            "epoch1_check": epoch1_instability(run_dir / "history.json"),
        }
        results_table.append(row)
        if run_dir.name == "control_mse":
            control_row = row

    if control_row is None:
        logger.error("No se encontró el checkpoint de control (control_mse). Abortando selección.")
        return

    # Criterio de selección: mejora PESQ-NB real sin caer SI-SDR/STOI por debajo del control
    candidatos = [r for r in results_table if r["nombre"] != "control_mse"]
    validos = [
        r for r in candidatos
        if r["pesq_nb"] > control_row["pesq_nb"]
        and r["sisdr"] >= control_row["sisdr"]
        and r["stoi"] >= control_row["stoi"]
    ]

    output_path = RESULTS_DIR / "paso1_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "control": control_row,
            "candidatos": candidatos,
            "validos_por_criterio": [v["nombre"] for v in validos],
        }, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info("RESULTADOS PASO 1 -- test_v1_en")
    logger.info("=" * 90)
    header = f"{'nombre':<16} | {'PESQ-NB':<8} | {'PESQ-WB':<8} | {'STOI':<7} | {'SI-SDR':<8} | epoch1"
    logger.info(header)
    logger.info("-" * 90)
    for r in [control_row] + candidatos:
        logger.info(
            f"{r['nombre']:<16} | {r['pesq_nb']:<8.3f} | {r['pesq_wb']:<8.3f} | "
            f"{r['stoi']:<7.3f} | {r['sisdr']:<8.2f} | {r['epoch1_check']}"
        )
    logger.info("=" * 90)

    if validos:
        winner = max(validos, key=lambda r: r["pesq_nb"])
        logger.info(f"\nGanador (pasa el criterio, mejor PESQ-NB): {winner['nombre']}")
        logger.info(f"  PESQ-NB: {winner['pesq_nb']:.3f} (control: {control_row['pesq_nb']:.3f})")
        logger.info(f"  Chequeo época 1: {winner['epoch1_check']}")
        if "sin señales" in winner["epoch1_check"]:
            logger.info("  -> Sin inestabilidad en época 1: correr Paso 2 igual, pero se puede "
                        "documentar que el warm-start probablemente no sea necesario.")
        else:
            logger.info("  -> Señal de inestabilidad en época 1: Paso 2 (warm-start) es prioritario.")
    else:
        logger.info("\nNingún candidato pasa el criterio (mejora PESQ-NB sin caer SI-SDR/STOI "
                    "por debajo del control). Ver docs/PLAN_V4.md: esto también es un resultado "
                    "documentable, no hay que forzar un ganador.")


if __name__ == "__main__":
    main()
