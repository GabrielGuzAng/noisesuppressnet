"""
evaluation/monitor_correlation.py

KPI propio del proyecto (tercer eje del aporte): correlación entre el PESQ real
y el PESQ estimado por Squim, **apareada archivo por archivo** sobre un test set
sellado.

Diseño: docs/PLAN_V4.md, "Fase 4". Dos regímenes de medición, no uno:

  1. REFERENCIA — correlación sobre el audio NOISY de los pares sellados. Es el
     régimen en que Squim fue validado por sus autores (Kumar 2023, MAE nominal
     0,142 para WB-PESQ); PLAN_V4 esperaba ~0,90 LCC. Sirve de ancla externa: si
     este número cae lejos de eso, el sospechoso es este script, no el hallazgo.
  2. RÉGIMEN REAL DE USO — la misma correlación sobre la SALIDA del modelo. Es
     donde el proxy se usa de verdad, y donde puede romperse.

El tercer régimen —limpio contra gameado— no es un modo de cálculo distinto: es
el régimen (2) corrido sobre modelos que nunca vieron Squim en su loss y sobre
modelos entrenados contra él, y después comparado con `--summarize`. Esa
comparación es EXPLORATORIA en la escala de informe/LEDGER.md: la hipótesis
salió de los resultados de V4/V4b, no la precede.

Por qué correlación y no la brecha que ya está medida
-----------------------------------------------------
Lo que V4/V4b midieron es la BRECHA `pesq_hat − PESQ-WB real` (+2,36 y +2,77
contra un MAE nominal de 0,142): un estimando de NIVEL, o sea sesgo. El KPI
declarado en el anteproyecto (§6.3, con umbral de alerta r < 0,30) es
CORRELACIÓN: un estimando de FORMA. No son sustitutos — una correlación alta
convive con un sesgo constante grande, que es justamente la firma del gaming.
Por eso este script reporta sesgo y correlación en la MISMA fila: separarlos es
lo que permitió confundir un estimando con el otro.

Además, aquella brecha estaba SIN APAREAR: `pesq_hat` salía de `val_squim` del
`history.json` (media de Squim sobre las 2.000 mixturas de validación) y el
PESQ-WB real de los 250 pares del sellado. Acá los dos números salen del mismo
archivo, y el `pesq_hat` por archivo queda guardado en `per_pair` — hasta ahora
no existía en ningún lado (informe/LEDGER.md §11.a).

Verificación posterior, no vigilancia
-------------------------------------
Este instrumento se construyó en septiembre de 2026, después de que las
decisiones de V4 y V4b se tomaran con la brecha. Produce el `r` prometido, no la
vigilancia prometida: el KPI nunca alertó de nada, porque no existía. El JSON de
salida lleva ese caveat en `posterior_verification`.

Decisiones que no se reabren
----------------------------
- Se compara contra PESQ-**WB**, nunca NB: la cabeza de Squim estima WB-PESQ,
  acotada en [1, 4.64] por sigmoide (CLAUDE.md, docs/decisions.md).
- El PESQ real NO se recalcula: se reusa por `pair_id` de
  `results/<variante>_<sellado>.json`, para que el número sea idéntico al ya
  publicado. `--verify-pesq N` recalcula N pares y verifica que la reinferencia
  reproduzca el audio que produjo esos números.

USO:
    # Régimen (1) + (2) sobre una variante (default: sellado v1_en)
    python -m evaluation.monitor_correlation --variant v1

    # Checkpoint fuera de la convención checkpoints/<variante>/best.pt
    python -m evaluation.monitor_correlation \
        --variant alpha_070 \
        --checkpoint checkpoints/v4_sweep/alpha_070/best.pt \
        --real-pesq-json results/v4_sweep/alpha_070_v1_en.json \
        --output results/v4_sweep/alpha_070_v1_en_correlation.json

    # Solo el régimen (1), sin modelo
    python -m evaluation.monitor_correlation --variant noisy --checkpoint none \
        --real-pesq-json results/v1_v1_en.json

    # Smoke test en CPU sobre 10 pares
    python -m evaluation.monitor_correlation --variant v1 --device cpu --limit 10

    # Régimen (3): tabla limpio vs gameado
    python -m evaluation.monitor_correlation --summarize results/*_correlation.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import stats
from torchaudio.pipelines import SQUIM_OBJECTIVE

from evaluation.evaluate_variant import infer_pair, load_variant
from evaluation.metrics import compute_metrics
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

# Mismos valores que analysis/reanalysis_stats.py, para que los IC de este
# archivo se lean con la misma vara que los del reanálisis.
SEED = 42
N_RESAMPLES = 10000

# Techo estructural de la cabeza PESQ de Squim (sigmoide sobre [1, 4.64]).
# Kumar 2023 reporta MAE 0,142 para WB-PESQ en el régimen en que la validaron.
SQUIM_PESQ_CEILING = 4.64
SQUIM_NOMINAL_MAE = 0.142

# Umbral de alerta declarado en el anteproyecto §6.3. Se evalúa y se reporta,
# pero es una verificación POSTERIOR: ver el docstring de módulo.
ALERT_THRESHOLD_R = 0.30

# Referencia esperada para el régimen (1), de PLAN_V4 Fase 4.
REFERENCE_EXPECTED_LCC = 0.90

TEST_SETS = {
    "v1_en": ("data/test_sealed/v1_en", "seal_test_metadata/test_v1_metadata.json"),
    "v2_es": ("data/test_sealed/v2_es", "seal_test_metadata/test_v2_metadata.json"),
    "v3_mls_es": ("data/test_sealed/v3_mls_es",
                  "seal_test_metadata/test_v3_mls_es_metadata.json"),
}

logger = logging.getLogger(__name__)


# ─────────────────────────── utilidades ───────────────────────────

def _display_path(p: Path) -> str:
    """Path relative to PROJECT_ROOT for logging, absolute if it lives outside."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def resolve_test_set(name: str) -> tuple[Path, Path]:
    """Map a sealed-set name to its (directory, metadata) paths.

    Args:
        name: One of the keys of ``TEST_SETS``.

    Returns:
        Tuple of absolute paths (test directory, metadata JSON).
    """
    if name not in TEST_SETS:
        raise ValueError(f"Sellado desconocido: {name}. Conocidos: {sorted(TEST_SETS)}")
    test_dir, metadata = TEST_SETS[name]
    return PROJECT_ROOT / test_dir, PROJECT_ROOT / metadata


def infer_exposure(checkpoint_path: Path | None) -> str:
    """Derive whether the checkpoint was trained against Squim, from its config.

    Reads ``<checkpoint_dir>/config.json`` and looks at the training loss. This
    is what makes the clean-vs-gamed split a property of the run rather than a
    hardcoded list of variant names.

    Returns:
        'squim' if the training loss included the Squim term, 'clean' if it did
        not, 'unknown' if there is no config next to the checkpoint.
    """
    if checkpoint_path is None:
        return "none"
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return "unknown"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    # V1 no tiene clave "loss" en su config: el default del trainer era
    # mse_magnitude, sin componente perceptual.
    loss_name = config.get("loss", "mse_magnitude")
    return "squim" if "squim" in loss_name else "clean"


# ─────────────────────────── estadística ───────────────────────────

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.pearsonr(x, y)[0])


def _mean(d: np.ndarray) -> float:
    return float(np.mean(d))


def pearson_with_ci(x: np.ndarray, y: np.ndarray, seed: int = SEED,
                    n_resamples: int = N_RESAMPLES) -> dict:
    """Pearson r with a paired bootstrap CI and an analytic Fisher-z CI.

    The bootstrap resamples pair indices jointly (``paired=True``), which is the
    right unit here: the two series are two measurements of the same file.
    The Fisher-z interval is carried alongside as a cheap cross-check — if the
    two disagree badly, the bootstrap is the one to distrust.

    Returns:
        Dict with r, p-value, both CIs, and the bootstrap method that was used.
    """
    r, p_value = stats.pearsonr(x, y)
    n = len(x)

    method = "BCa"
    try:
        res = stats.bootstrap(
            (x, y), _pearson_r, paired=True, vectorized=False,
            method="BCa", n_resamples=n_resamples,
            confidence_level=0.95, random_state=np.random.default_rng(seed),
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("BCa devolvió un intervalo no finito")
    except (ValueError, RuntimeError) as e:
        # BCa degenera si la aceleración no está definida (p. ej. un estadístico
        # constante sobre las remuestras jackknife). Se cae a percentil y queda
        # registrado en el JSON cuál de los dos se usó.
        logger.warning("BCa falló para Pearson (%s); se usa percentil", e)
        res = stats.bootstrap(
            (x, y), _pearson_r, paired=True, vectorized=False,
            method="percentile", n_resamples=n_resamples,
            confidence_level=0.95, random_state=np.random.default_rng(seed),
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        method = "percentile"

    if n > 3 and abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        fisher = {"lo": float(np.tanh(z - 1.96 * se)), "hi": float(np.tanh(z + 1.96 * se))}
    else:
        fisher = {"lo": None, "hi": None}

    return {
        "r": float(r),
        "p_value": float(p_value),
        "ci95": {"lo": lo, "hi": hi, "method": method,
                 "n_resamples": n_resamples, "seed": seed},
        "ci95_fisher_z": fisher,
    }


def bootstrap_ci_mean(d: np.ndarray, seed: int = SEED,
                      n_resamples: int = N_RESAMPLES) -> dict:
    """BCa bootstrap CI for the mean of a one-sample array.

    Same recipe as ``analysis/reanalysis_stats.py::bootstrap_ci``, kept local so
    that ``evaluation/`` does not depend on ``analysis/``.
    """
    try:
        res = stats.bootstrap(
            (d,), _mean, method="BCa", n_resamples=n_resamples,
            confidence_level=0.95, random_state=np.random.default_rng(seed),
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        method = "BCa"
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("BCa devolvió un intervalo no finito")
    except (ValueError, RuntimeError) as e:
        logger.warning("BCa falló para la media (%s); se usa percentil", e)
        res = stats.bootstrap(
            (d,), _mean, method="percentile", n_resamples=n_resamples,
            confidence_level=0.95, random_state=np.random.default_rng(seed),
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        method = "percentile"
    return {"point": float(np.mean(d)), "lo": lo, "hi": hi, "method": method,
            "n_resamples": n_resamples, "seed": seed}


def correlation_row(pesq_hat: np.ndarray, pesq_real: np.ndarray,
                    seed: int = SEED, n_resamples: int = N_RESAMPLES) -> dict:
    """Build one reporting row: correlation and bias together, never apart.

    Args:
        pesq_hat: Squim's estimate, per pair.
        pesq_real: Real PESQ-WB, per pair, same order.

    Returns:
        Dict with Pearson (r, CI, p), Spearman (rho, p), bias with CI, MAE,
        RMSE, the two means, and n.
    """
    if len(pesq_hat) != len(pesq_real):
        raise ValueError("Las dos series tienen distinto largo: no están apareadas")
    if len(pesq_hat) < 4:
        raise ValueError(f"n={len(pesq_hat)}: muy pocos pares para correlacionar")

    # Apareado: la media de las diferencias es idéntica a la diferencia de las
    # medias. Se calcula sobre las diferencias porque así el IC también sale
    # apareado, que es el punto de todo el script.
    diff = pesq_hat - pesq_real
    rho, rho_p = stats.spearmanr(pesq_hat, pesq_real)
    pearson = pearson_with_ci(pesq_hat, pesq_real, seed=seed, n_resamples=n_resamples)

    return {
        "n": int(len(pesq_hat)),
        "pearson_r": pearson["r"],
        "pearson_p_value": pearson["p_value"],
        "pearson_ci95": pearson["ci95"],
        "pearson_ci95_fisher_z": pearson["ci95_fisher_z"],
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "bias_mean": float(np.mean(diff)),
        "bias_ci95": bootstrap_ci_mean(diff, seed=seed, n_resamples=n_resamples),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mean_pesq_hat": float(np.mean(pesq_hat)),
        "mean_pesq_wb_real": float(np.mean(pesq_real)),
        "bias_vs_nominal_mae": float(np.mean(diff) / SQUIM_NOMINAL_MAE),
        "alert_threshold_r": ALERT_THRESHOLD_R,
        "below_alert_threshold": bool(pearson["r"] < ALERT_THRESHOLD_R),
    }


# ─────────────────────────── medición ───────────────────────────

def load_real_pesq(results_path: Path) -> dict[int, dict]:
    """Index the published per-pair PESQ-WB of a results JSON by pair_id.

    Returns:
        Tuple of (mapping pair_id -> {'noisy': float, 'est': float}, header
        dict with the file's variant/epoch/test set so the caller can check
        that the checkpoint and the published numbers are the same run).
    """
    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_pair = {
        record["pair_id"]: {"noisy": record["pesq_wb_noisy"], "est": record["pesq_wb_est"]}
        for record in data["all_pairs"]
    }
    return by_pair, {
        "variant": data.get("variant"),
        "checkpoint_epoch": data.get("checkpoint_epoch"),
        "test_set": data.get("test_set"),
        "n_pairs_evaluated": data.get("n_pairs_evaluated"),
    }


def squim_pesq(squim_model, waveform: torch.Tensor, device: torch.device) -> float:
    """Squim's PESQ-WB estimate for one waveform.

    Args:
        waveform: 1-D float tensor at 16 kHz.

    Returns:
        The estimate as a float. cuDNN stays enabled here (unlike in the
        trainer, where it has to be off for the backward pass) because this is
        inference only.
    """
    with torch.no_grad():
        wav = waveform.to(device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        _, pesq_hat, _ = squim_model(wav)
        return float(pesq_hat.squeeze())


def measure(variant: str, checkpoint_path: Path | None, test_dir: Path,
            metadata_path: Path, real_pesq_path: Path, device: torch.device,
            limit: int | None = None, verify_pesq: int = 5,
            exposure: str | None = None, allow_epoch_mismatch: bool = False,
            seed: int = SEED, n_resamples: int = N_RESAMPLES) -> dict:
    """Run both regimes over one checkpoint and one sealed set.

    Args:
        variant: Label for the output; need not match a checkpoints/ directory.
        checkpoint_path: Checkpoint to evaluate, or None for reference-only.
        real_pesq_path: Published results JSON that supplies the real PESQ-WB.
        limit: Process only the first N pairs (smoke test).
        verify_pesq: Recompute real PESQ-WB on the first N pairs and check it
            against the published value.
        allow_epoch_mismatch: Skip the checkpoint/results consistency check.

    Returns:
        The full report dict, as written to the output JSON.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"CORRELACIÓN PESQ real vs Squim: {variant.upper()} sobre {test_dir.name}")
    print(f"{'=' * 70}")
    print(f"Device:        {device}")
    print(f"Sellado:       {_display_path(test_dir)}")
    print(f"PESQ real de:  {_display_path(real_pesq_path)}  (reusado, no recalculado)")

    real_pesq, real_header = load_real_pesq(real_pesq_path)
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    pair_meta = {p["id"]: p for p in metadata["pairs"]}

    if Path(real_header["test_set"]).name != test_dir.name:
        raise ValueError(
            f"El JSON de PESQ real es de '{real_header['test_set']}' y el sellado "
            f"pedido es '{test_dir.name}'. No están apareados.")

    # Cargar modelo (o no, si es el régimen de referencia solo)
    model, ckpt = None, None
    if checkpoint_path is not None:
        model, ckpt = load_variant(variant, device, checkpoint_path=str(checkpoint_path))
        # Esta comprobación existe por un caso concreto: best.pt de v4b_placebo es
        # la época 1 (mínimo de val MSE) y sus resultados publicados están por
        # época. Aparear el checkpoint equivocado con el PESQ publicado
        # corrompería el KPI entero sin dar ninguna señal.
        if ckpt["epoch"] != real_header["checkpoint_epoch"] and not allow_epoch_mismatch:
            raise ValueError(
                f"El checkpoint es de la época {ckpt['epoch']} y el PESQ publicado "
                f"de la época {real_header['checkpoint_epoch']} "
                f"({_display_path(real_pesq_path)}). Apuntá a los resultados de la "
                f"misma época, o pasá --allow-epoch-mismatch si sabés lo que hacés.")

    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)

    print("Cargando Squim (SQUIM_OBJECTIVE)...")
    squim_model = SQUIM_OBJECTIVE.get_model().to(device)
    squim_model.eval()
    for p in squim_model.parameters():
        p.requires_grad_(False)

    pair_ids = sorted(pair_meta.keys())
    if limit is not None:
        pair_ids = pair_ids[:limit]
        print(f"⚠ --limit {limit}: esto es un smoke test, no un resultado")

    per_pair = []
    pesq_reuse_deltas = []
    print(f"\nProcesando {len(pair_ids)} pares...")

    for i, pair_id in enumerate(pair_ids):
        pair_dir = test_dir / f"pair_{pair_id:04d}"
        if not pair_dir.exists():
            logger.warning("Falta %s, se saltea", pair_dir.name)
            continue
        if pair_id not in real_pesq:
            logger.warning("pair_%04d no está en el JSON de PESQ real, se saltea", pair_id)
            continue

        noisy, sr_n = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, sr_c = torchaudio.load(str(pair_dir / "clean.wav"))
        if not sr_n == sr_c == SAMPLE_RATE:
            raise ValueError(f"SR incorrecto en pair_{pair_id:04d}: {sr_n}, {sr_c}")

        record = {
            "pair_id": pair_id,
            "bucket_idx": pair_meta[pair_id]["bucket_idx"],
            "snr_db": pair_meta[pair_id]["snr_db"],
            "noise_category": pair_meta[pair_id]["noise_category"],
            "pesq_wb_real_noisy": real_pesq[pair_id]["noisy"],
            "pesq_hat_noisy": squim_pesq(squim_model, noisy.squeeze(0), device),
        }

        if model is not None:
            enhanced, _ = infer_pair(model, stft, noisy.squeeze(0), device)
            record["pesq_wb_real_est"] = real_pesq[pair_id]["est"]
            record["pesq_hat_est"] = squim_pesq(squim_model, enhanced, device)

            # Auditoría del reuso: la reinferencia tiene que reproducir el audio
            # que produjo el PESQ publicado. Si no, reusar ese número es mentira.
            if i < verify_pesq:
                recomputed = compute_metrics(clean.squeeze(0), enhanced)["PESQ-WB"]
                published = real_pesq[pair_id]["est"]
                if np.isfinite(recomputed) and np.isfinite(published):
                    pesq_reuse_deltas.append(abs(recomputed - published))

        per_pair.append(record)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pair_ids)}")

    # ─── Regímenes ───
    regimes = {}

    def _series(hat_key: str, real_key: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Paired, finite-only series for one regime."""
        hat, real = [], []
        for r in per_pair:
            h, v = r.get(hat_key), r.get(real_key)
            if h is None or v is None:
                continue
            if np.isfinite(h) and np.isfinite(v):
                hat.append(h)
                real.append(v)
        return np.asarray(hat), np.asarray(real), len(per_pair) - len(hat)

    hat_n, real_n, dropped_n = _series("pesq_hat_noisy", "pesq_wb_real_noisy")
    regimes["reference_noisy"] = {
        "description": "Squim vs PESQ-WB real sobre el audio noisy del sellado. "
                       "Régimen en que Squim fue validado por sus autores.",
        "evidence_level": "descriptivo (diseñado antes del dato, PLAN_V4 Fase 4)",
        "expected_lcc": REFERENCE_EXPECTED_LCC,
        "n_dropped_non_finite": dropped_n,
        **correlation_row(hat_n, real_n, seed=seed, n_resamples=n_resamples),
    }

    if model is not None:
        hat_e, real_e, dropped_e = _series("pesq_hat_est", "pesq_wb_real_est")
        regimes["model_output"] = {
            "description": "Squim vs PESQ-WB real sobre la salida del modelo. "
                           "Régimen real de uso del proxy.",
            "evidence_level": "descriptivo (diseñado antes del dato, PLAN_V4 Fase 4)",
            "n_dropped_non_finite": dropped_e,
            **correlation_row(hat_e, real_e, seed=seed, n_resamples=n_resamples),
        }

    report = {
        "variant": variant,
        "checkpoint": None if checkpoint_path is None else _display_path(checkpoint_path),
        "checkpoint_epoch": None if ckpt is None else ckpt["epoch"],
        "squim_exposure": exposure or infer_exposure(checkpoint_path),
        "test_set": _display_path(test_dir),
        "real_pesq_source": _display_path(real_pesq_path),
        "real_pesq_recomputed": False,
        "metric": "PESQ-WB",
        "metric_note": "PESQ-WB y nunca NB: la cabeza de Squim estima WB-PESQ, "
                       f"acotada en [1, {SQUIM_PESQ_CEILING}] por sigmoide.",
        "squim_bundle": "torchaudio.pipelines.SQUIM_OBJECTIVE",
        "squim_pesq_ceiling": SQUIM_PESQ_CEILING,
        "squim_nominal_mae_wb": SQUIM_NOMINAL_MAE,
        "n_pairs": len(per_pair),
        "limit_applied": limit,
        "device": str(device),
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "posterior_verification": (
            "Instrumento construido en septiembre de 2026, después de que las "
            "decisiones de V4 y V4b se tomaran con la brecha pesq_hat - PESQ-WB. "
            "Da el r comprometido, no la vigilancia comprometida: el KPI nunca "
            "alertó de nada porque no existía. Ver informe/LEDGER.md §11.a."),
        "regimes": regimes,
        "per_pair": per_pair,
    }

    if pesq_reuse_deltas:
        report["pesq_reuse_check"] = {
            "n_recomputed": len(pesq_reuse_deltas),
            "max_abs_delta": float(max(pesq_reuse_deltas)),
            "mean_abs_delta": float(np.mean(pesq_reuse_deltas)),
            "note": "PESQ-WB recalculado sobre los primeros pares y comparado "
                    "contra el publicado. Verifica que la reinferencia reproduce "
                    "el audio que generó los números que se reusan.",
        }

    return report


def print_report(report: dict) -> None:
    """Print the per-variant regimes as a table."""
    print(f"\n{'=' * 70}")
    print(f"RESULTADO ({report['variant'].upper()}, exposición a Squim: "
          f"{report['squim_exposure']})")
    print(f"{'=' * 70}")
    header = (f"{'Régimen':<18} {'n':>4} {'Pearson r':>10} {'IC95':>18} "
              f"{'Spearman':>9} {'sesgo':>8} {'MAE':>7}")
    print(header)
    print("-" * len(header))
    for name, row in report["regimes"].items():
        ci = f"[{row['pearson_ci95']['lo']:+.3f}, {row['pearson_ci95']['hi']:+.3f}]"
        print(f"{name:<18} {row['n']:>4} {row['pearson_r']:>10.3f} {ci:>18} "
              f"{row['spearman_rho']:>9.3f} {row['bias_mean']:>+8.3f} {row['mae']:>7.3f}")

    ref = report["regimes"].get("reference_noisy")
    if ref is not None:
        print(f"\nAncla externa — régimen de referencia: r = {ref['pearson_r']:.3f} "
              f"contra ~{REFERENCE_EXPECTED_LCC:.2f} esperado (PLAN_V4 Fase 4).")
        if abs(ref["pearson_r"] - REFERENCE_EXPECTED_LCC) > 0.15:
            print("  ⚠ Lejos de lo esperado. Sospechar del script antes que del hallazgo.")

    check = report.get("pesq_reuse_check")
    if check is not None:
        print(f"\nAuditoría del reuso de PESQ: {check['n_recomputed']} pares "
              f"recalculados, desvío máximo {check['max_abs_delta']:.2e}")
        if check["max_abs_delta"] > 1e-3:
            print("  ⚠ La reinferencia NO reproduce el audio publicado. "
                  "Los números reusados no corresponden a este checkpoint.")


# ─────────────────────────── régimen (3) ───────────────────────────

def summarize(report_paths: list[Path]) -> dict:
    """Assemble the clean-vs-gamed comparison from per-variant reports.

    This is the exploratory regime: the hypothesis came out of the V4/V4b
    results, so the level in informe/LEDGER.md's scale is 'exploratorio'. It
    only groups and prints what the per-variant runs already measured — there is
    no test here, and there should not be: the checkpoints are not independent
    draws (several share an initialisation) and n per group is single digits.
    """
    rows = []
    for path in report_paths:
        with path.open(encoding="utf-8") as f:
            report = json.load(f)
        regime = report["regimes"].get("model_output")
        if regime is None:
            continue  # reference-only run, no model to group
        rows.append({
            "variant": report["variant"],
            "squim_exposure": report["squim_exposure"],
            "test_set": Path(report["test_set"]).name,
            "checkpoint": report["checkpoint"],
            "checkpoint_epoch": report["checkpoint_epoch"],
            "n": regime["n"],
            "pearson_r": regime["pearson_r"],
            "pearson_ci95_lo": regime["pearson_ci95"]["lo"],
            "pearson_ci95_hi": regime["pearson_ci95"]["hi"],
            "spearman_rho": regime["spearman_rho"],
            "bias_mean": regime["bias_mean"],
            "mae": regime["mae"],
            "mean_pesq_hat": regime["mean_pesq_hat"],
            "mean_pesq_wb_real": regime["mean_pesq_wb_real"],
            "source": _display_path(path),
        })

    rows.sort(key=lambda r: (r["test_set"], r["squim_exposure"], r["variant"]))

    by_group = {}
    for row in rows:
        key = (row["test_set"], row["squim_exposure"])
        by_group.setdefault(key, []).append(row)

    groups = [
        {
            "test_set": test_set,
            "squim_exposure": exposure,
            "n_checkpoints": len(group),
            "variants": [r["variant"] for r in group],
            "pearson_r_mean": float(np.mean([r["pearson_r"] for r in group])),
            "pearson_r_min": float(np.min([r["pearson_r"] for r in group])),
            "pearson_r_max": float(np.max([r["pearson_r"] for r in group])),
            "bias_mean": float(np.mean([r["bias_mean"] for r in group])),
        }
        for (test_set, exposure), group in sorted(by_group.items())
    ]

    return {
        "question": "¿Squim pierde poder DISCRIMINANTE bajo optimización "
                    "adversaria, o solo se DESCALIBRA? Pearson alto con sesgo "
                    "explotado = descalibración; Pearson derrumbado = pérdida "
                    "de discriminación.",
        "evidence_level": "exploratorio (la hipótesis salió de los resultados de "
                          "V4/V4b; escala en informe/LEDGER.md §1)",
        "caveat": "Sin test estadístico a propósito: los checkpoints no son "
                  "sorteos independientes (varios comparten inicialización) y n "
                  "por grupo es de un dígito. Se muestra el patrón, sin p-valor.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "groups": groups,
        "rows": rows,
    }


def print_summary(summary: dict) -> None:
    """Print the exploratory clean-vs-gamed table."""
    print(f"\n{'=' * 96}")
    print("RÉGIMEN (3) — LIMPIO CONTRA GAMEADO  [exploratorio]")
    print(f"{'=' * 96}")
    header = (f"{'sellado':<11} {'exposición':<11} {'variante':<22} {'n':>4} "
              f"{'Pearson r':>10} {'IC95':>18} {'Spearman':>9} {'sesgo':>8} {'MAE':>7}")
    print(header)
    print("-" * len(header))
    for row in summary["rows"]:
        ci = f"[{row['pearson_ci95_lo']:+.3f}, {row['pearson_ci95_hi']:+.3f}]"
        print(f"{row['test_set']:<11} {row['squim_exposure']:<11} {row['variant']:<22} "
              f"{row['n']:>4} {row['pearson_r']:>10.3f} {ci:>18} "
              f"{row['spearman_rho']:>9.3f} {row['bias_mean']:>+8.3f} {row['mae']:>7.3f}")

    print(f"\n{'-' * 96}")
    print(f"{'sellado':<11} {'exposición':<11} {'k':>3} {'r medio':>9} "
          f"{'r min':>8} {'r max':>8} {'sesgo medio':>12}")
    print("-" * 96)
    for g in summary["groups"]:
        print(f"{g['test_set']:<11} {g['squim_exposure']:<11} {g['n_checkpoints']:>3} "
              f"{g['pearson_r_mean']:>9.3f} {g['pearson_r_min']:>8.3f} "
              f"{g['pearson_r_max']:>8.3f} {g['bias_mean']:>+12.3f}")
    print(f"\n{summary['caveat']}")


# ─────────────────────────── CLI ───────────────────────────

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="KPI propio: correlación apareada entre PESQ real y Squim.")
    parser.add_argument("--variant", type=str,
                        help="Etiqueta de la variante (v1, v2, alpha_070, ...). "
                             "No tiene que existir como checkpoints/<variant>/.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al checkpoint. Default: "
                             "checkpoints/<variant>/best.pt. 'none' corre solo "
                             "el régimen de referencia, sin modelo.")
    parser.add_argument("--test-set", type=str, default="v1_en",
                        choices=sorted(TEST_SETS),
                        help="Sellado sobre el que medir (default: v1_en)")
    parser.add_argument("--real-pesq-json", type=str, default=None,
                        help="JSON de resultados del que se reusa el PESQ-WB "
                             "real por par. Default: "
                             "results/<variant>_<test_set>.json")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON de salida. Default: "
                             "results/<variant>_<test_set>_correlation.json")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N pares (smoke test)")
    parser.add_argument("--verify-pesq", type=int, default=5,
                        help="Recalcular PESQ-WB en los primeros N pares para "
                             "auditar el reuso (default: 5, 0 lo desactiva)")
    parser.add_argument("--exposure", type=str, default=None,
                        choices=["clean", "squim"],
                        help="Override de la exposición a Squim. Por default se "
                             "deduce del config.json del checkpoint.")
    parser.add_argument("--allow-epoch-mismatch", action="store_true",
                        help="Permitir que la época del checkpoint no coincida "
                             "con la del JSON de PESQ real")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES,
                        help=f"Remuestras del bootstrap (default: {N_RESAMPLES})")
    parser.add_argument("--summarize", type=str, nargs="+", default=None,
                        help="Modo régimen (3): arma la tabla limpio vs gameado "
                             "a partir de varios *_correlation.json")
    args = parser.parse_args()

    if args.summarize:
        paths = [Path(p).resolve() for p in args.summarize]
        summary = summarize(paths)
        print_summary(summary)
        out = Path(args.output).resolve() if args.output else (
            PROJECT_ROOT / "results" / "squim_correlation_summary.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resumen guardado en: {_display_path(out)}\n")
        return

    if not args.variant:
        parser.error("--variant es obligatorio salvo en modo --summarize")

    test_dir, metadata_path = resolve_test_set(args.test_set)

    if args.checkpoint == "none":
        checkpoint_path = None
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        checkpoint_path = PROJECT_ROOT / "checkpoints" / args.variant / "best.pt"
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    if args.real_pesq_json:
        real_pesq_path = Path(args.real_pesq_json).resolve()
    else:
        real_pesq_path = PROJECT_ROOT / "results" / f"{args.variant}_{args.test_set}.json"
    if not real_pesq_path.exists():
        raise FileNotFoundError(
            f"No existe el JSON de PESQ real: {real_pesq_path}. Pasá "
            f"--real-pesq-json apuntando a los resultados publicados de este "
            f"checkpoint sobre {args.test_set}.")

    report = measure(
        variant=args.variant,
        checkpoint_path=checkpoint_path,
        test_dir=test_dir,
        metadata_path=metadata_path,
        real_pesq_path=real_pesq_path,
        device=_resolve_device(args.device),
        limit=args.limit,
        verify_pesq=args.verify_pesq,
        exposure=args.exposure,
        allow_epoch_mismatch=args.allow_epoch_mismatch,
        n_resamples=args.n_resamples,
    )
    print_report(report)

    # El nombre incluye el sellado, no solo la variante: PLAN_V4 decía
    # results/<variante>_correlation.json, pero lo escribió cuando el único
    # sellado era v1_en. Con tres sellados, esa convención hace que la segunda
    # corrida de una variante pise la primera sin avisar -- el mismo bug que ya
    # se corrigió en evaluate_variant.py.
    output_path = Path(args.output).resolve() if args.output else (
        PROJECT_ROOT / "results" / f"{args.variant}_{args.test_set}_correlation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Guardado en: {_display_path(output_path)}\n")


if __name__ == "__main__":
    main()
