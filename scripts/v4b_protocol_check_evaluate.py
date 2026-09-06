"""
scripts/v4b_protocol_check_evaluate.py

Evalúa las corridas de V4b (scripts/v4b_protocol_check.py) sobre
test_v1_en, UNA ÉPOCA POR VEZ.

Por qué por época y no `best.pt`: `Trainer` elige `best.pt` por `val_loss`
mínima, y para `mse_plus_squim` esa `val_loss` incluye el término
`(1-alpha)*(-pesq_hat)`. Más gameado = mayor pesq_hat = menor val_loss =
"mejor" checkpoint. El criterio de selección está contaminado por
exactamente lo que queremos medir. En el sweep de Fase 3 esto se ve
directo: `alpha_070` guardó como best la época 3, la de `val_squim` más
inflado (-4.43), y `alpha_080` guardó la época 2. Acá evaluamos
`epoch_01/02/03.pt` por separado y miramos la trayectoria.

Además reporta el DIAGNÓSTICO DE GAMING por época: la brecha entre el
PESQ que Squim cree que produjo el modelo (`pesq_hat = -val_squim`, del
history.json) y el PESQ-WB real medido sobre el test sellado. Squim
estima WB-PESQ, acotado por construcción en [1, 4.64] vía sigmoide
(Kumar et al. 2023, ec. 5) -- por eso la comparación honesta es contra
PESQ-WB, no NB. En V4 esa brecha llegó a ~3 puntos (Squim estimaba
4.0-4.4, el real daba 1.07-1.36).

Compara contra dos referencias ya existentes:
  - `v4b_placebo` a la misma época: aísla el término Squim del efecto de
    seguir fine-tuneando V1 con lr bajo (el "placebo setup" de Xu).
  - `alpha_090` del sweep de Fase 3: la celda init-random/lr-2e-4 del 2x2.

USO:
    python -m scripts.v4b_protocol_check_evaluate
    python -m scripts.v4b_protocol_check_evaluate --skip_eval  # solo re-arma
                                                              # tablas de JSONs ya generados
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "checkpoints" / "v4b"
RESULTS_DIR = PROJECT_ROOT / "results" / "v4b"
TEST_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"
ALPHA_090_RESULT = PROJECT_ROOT / "results" / "v4_sweep" / "alpha_090_v1_en.json"

RUNS = ["v4b_main", "v4b_placebo", "v4b_lr2e4"]
SQUIM_CEILING = 4.64  # techo estructural de la cabeza PESQ de Squim (Kumar 2023, ec. 5)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(run_name: str, ckpt_path: Path, skip_eval: bool) -> dict | None:
    """Corre evaluate_variant sobre un checkpoint puntual. Devuelve el
    dict `global` del JSON, o None si el checkpoint no existe."""
    if not ckpt_path.exists():
        return None

    tag = f"{run_name}_{ckpt_path.stem}"
    output_json = RESULTS_DIR / f"{tag}_v1_en.json"

    if not (skip_eval and output_json.exists()):
        cmd = [
            sys.executable, "-m", "evaluation.evaluate_variant",
            "--variant", tag,
            "--checkpoint", str(ckpt_path),
            "--test_dir", str(TEST_DIR),
            "--metadata", str(METADATA),
            "--output", str(output_json),
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    with open(output_json) as f:
        return json.load(f)


def pesq_hat_trajectory(run_name: str) -> list[float | None]:
    """pesq_hat estimado por Squim por época (= -val_squim del history).
    Devuelve [] si no hay history, y None por época si esa corrida no usa
    Squim (val_squim == 0.0, como en el placebo)."""
    history_path = RUN_DIR / run_name / "history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        h = json.load(f)
    return [(-v if v != 0.0 else None) for v in h.get("val_squim", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_eval", action="store_true",
                        help="No re-evaluar: re-armar tablas desde los JSON ya generados.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filas = []
    for run_name in RUNS:
        traj = pesq_hat_trajectory(run_name)
        for epoch in (1, 2, 3):
            ckpt = RUN_DIR / run_name / f"epoch_{epoch:02d}.pt"
            logger.info(f"Evaluando {run_name} época {epoch}")
            res = evaluate_checkpoint(run_name, ckpt, args.skip_eval)
            if res is None:
                logger.warning(f"  {ckpt} no existe -- salteado")
                continue
            g = res["global"]
            filas.append({
                "run": run_name,
                "epoch": epoch,
                "pesq_nb": g["pesq_nb"]["est_mean"],
                "pesq_nb_delta": g["pesq_nb"]["delta_mean"],
                "pesq_wb": g["pesq_wb"]["est_mean"],
                "pesq_wb_delta": g["pesq_wb"]["delta_mean"],
                "stoi_delta": g["stoi"]["delta_mean"],
                "sisdr_delta": g["sisdr"]["delta_mean"],
                "pesq_hat": traj[epoch - 1] if len(traj) >= epoch else None,
            })

    if not filas:
        raise SystemExit("No se evaluó ningún checkpoint. ¿Corrió scripts.v4b_protocol_check?")

    noisy_nb = None
    noisy_wb = None
    for run_name in RUNS:
        for epoch in (1, 2, 3):
            p = RESULTS_DIR / f"{run_name}_epoch_{epoch:02d}_v1_en.json"
            if p.exists():
                with open(p) as f:
                    g = json.load(f)["global"]
                noisy_nb = g["pesq_nb"]["noisy_mean"]
                noisy_wb = g["pesq_wb"]["noisy_mean"]
                break
        if noisy_nb is not None:
            break

    # ---- Tabla 1: trayectoria por época ----
    logger.info("\n" + "=" * 104)
    logger.info(f"V4b -- TRAYECTORIA POR ÉPOCA sobre test_v1_en   (Noisy: PESQ-NB {noisy_nb:.3f} / PESQ-WB {noisy_wb:.3f})")
    logger.info("=" * 104)
    logger.info(f"{'run':<14} | {'ep':<3} | {'PESQ-NB':<8} {'Δ':<8} | {'PESQ-WB':<8} {'Δ':<8} | "
                f"{'ΔSTOI':<7} | {'ΔSI-SDR':<8} | {'pesq_hat':<9} | brecha")
    logger.info("-" * 104)
    for f_ in filas:
        ph = f_["pesq_hat"]
        ph_str = f"{ph:.3f}" if ph is not None else "--"
        # Brecha de gaming: lo que Squim cree menos el WB real medido.
        brecha = f"{ph - f_['pesq_wb']:+.2f}" if ph is not None else "--"
        logger.info(f"{f_['run']:<14} | {f_['epoch']:<3} | {f_['pesq_nb']:<8.3f} {f_['pesq_nb_delta']:<+8.3f} | "
                    f"{f_['pesq_wb']:<8.3f} {f_['pesq_wb_delta']:<+8.3f} | {f_['stoi_delta']:<+7.3f} | "
                    f"{f_['sisdr_delta']:<+8.2f} | {ph_str:<9} | {brecha}")
    logger.info("=" * 104)
    logger.info(f"'brecha' = pesq_hat (lo que Squim cree) - PESQ-WB real. Techo de Squim: {SQUIM_CEILING}.")
    logger.info("En V4 (init random, lr 2e-4) esa brecha llegó a ~+3.0 puntos.")

    # ---- Tabla 2: el 2x2 (última época de cada celda) ----
    def ultima(run_name):
        candidatos = [f_ for f_ in filas if f_["run"] == run_name]
        return candidatos[-1] if candidatos else None

    alpha090 = None
    if ALPHA_090_RESULT.exists():
        with open(ALPHA_090_RESULT) as f:
            g = json.load(f)["global"]
        alpha090 = {
            "pesq_nb_delta": g["pesq_nb"]["delta_mean"],
            "stoi_delta": g["stoi"]["delta_mean"],
            "sisdr_delta": g["sisdr"]["delta_mean"],
        }

    logger.info("\n" + "=" * 104)
    logger.info("V4b -- 2x2: ¿qué elemento del protocolo protege?  (Δ PESQ-NB vs Noisy, última época)")
    logger.info("=" * 104)
    logger.info(f"{'':<26} | {'lr = 2e-4':<24} | {'lr = 2e-5':<24}")
    logger.info("-" * 104)
    a090 = f"{alpha090['pesq_nb_delta']:+.3f} (alpha_090, Fase 3)" if alpha090 else "-- (falta JSON)"
    logger.info(f"{'init random':<26} | {a090:<24} | {'-- (no aplica)':<24}")
    m_lr = ultima("v4b_lr2e4")
    m_main = ultima("v4b_main")
    s_lr = f"{m_lr['pesq_nb_delta']:+.3f} (v4b_lr2e4)" if m_lr else "-- (no corrido)"
    s_main = f"{m_main['pesq_nb_delta']:+.3f} (v4b_main)" if m_main else "-- (no corrido)"
    logger.info(f"{'init V1 convergido':<26} | {s_lr:<24} | {s_main:<24}")
    logger.info("=" * 104)

    # ---- Tabla 3: aislamiento del término Squim (vs placebo) ----
    m_pl = ultima("v4b_placebo")
    if m_pl and m_main:
        logger.info("\n" + "=" * 104)
        logger.info("V4b -- ¿APORTA EL TÉRMINO SQUIM?  (main vs placebo, mismo protocolo, misma época)")
        logger.info("=" * 104)
        for etiqueta, m in (("placebo (alpha=1, sin Squim)", m_pl), ("main   (alpha=0.9, con Squim)", m_main)):
            logger.info(f"  {etiqueta:<32} ΔPESQ-NB {m['pesq_nb_delta']:+.3f}  "
                        f"ΔPESQ-WB {m['pesq_wb_delta']:+.3f}  ΔSTOI {m['stoi_delta']:+.3f}  "
                        f"ΔSI-SDR {m['sisdr_delta']:+.2f}")
        d = m_main["pesq_nb_delta"] - m_pl["pesq_nb_delta"]
        logger.info("-" * 104)
        logger.info(f"  Aporte atribuible SOLO al término Squim: {d:+.3f} PESQ-NB")
        logger.info("  (positivo = el término perceptual aporta sobre seguir fine-tuneando con lr bajo;")
        logger.info("   ~0 = no aporta, consistente con Xu [24]/[31] para proxy fijo;")
        logger.info("   negativo = degrada, como en V4)")
        logger.info("=" * 104)

    out = RESULTS_DIR / "v4b_comparison.json"
    with open(out, "w") as f:
        json.dump({
            "noisy_pesq_nb": noisy_nb,
            "noisy_pesq_wb": noisy_wb,
            "squim_ceiling": SQUIM_CEILING,
            "filas": filas,
            "alpha_090_ref": alpha090,
        }, f, indent=2)
    logger.info(f"\nResumen guardado en {out}")
    logger.info("Registrar el cierre en docs/decisions.md (entrada de V4b).")


if __name__ == "__main__":
    main()
