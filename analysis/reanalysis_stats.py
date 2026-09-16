"""Statistical reanalysis of the sealed-test-set evaluation results.

Recomputes paired-difference statistics (sign test, BCa bootstrap CIs, Spearman
monotonicity, Holm-adjusted p-values) directly from ``results/*_v*.json`` and
``seal_test_metadata/*.json``, per the contract in section 4 of
``docs/PROMPTS_REANALISIS.md``. Produces ``results/reanalysis_stats.json`` and a
human-readable ``docs/reanalisis_estadistico.md`` report.

This module does not train or evaluate models: it is pure CPU post-processing
over already-computed evaluation outputs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED: int = 42
N_RESAMPLES: int = 10000
BROKEN_THRESHOLD: float = -0.2
SENSITIVITY_THRESHOLDS: tuple[float, ...] = (-0.1, -0.2, -0.3, -0.5)

PRIMARY_METRIC = "pesq_nb"

VARIANTS = ("v1", "v2", "v3", "v3b", "v3e")
TEST_SETS = ("v1_en", "v2_es")

BASELINE_QUANTILE: float = 0.75

PRIMARY_CONTROL: str = "v4b_placebo_epoch_03"
CONTROL_PATHS: dict[str, str] = {
    "v4b_placebo_epoch_01": "results/v4b/v4b_placebo_epoch_01_v1_en.json",
    "v4b_placebo_epoch_02": "results/v4b/v4b_placebo_epoch_02_v1_en.json",
    "v4b_placebo_epoch_03": "results/v4b/v4b_placebo_epoch_03_v1_en.json",
    "v2": "results/v2_v1_en.json",
}
TREATMENTS: tuple[str, ...] = ("v3", "v3b", "v3e")


def load_pairs(results_path: Path) -> dict[int, dict]:
    """Load a results JSON and index its all_pairs records by pair_id.

    Args:
        results_path: Path to a ``results/{variant}_{testset}.json`` file.

    Returns:
        Mapping from ``pair_id`` to the corresponding record in ``all_pairs``.
    """
    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {record["pair_id"]: record for record in data["all_pairs"]}


def paired_diff(
    base: dict[int, dict], other: dict[int, dict], field: str
) -> tuple[np.ndarray, list[int]]:
    """Return (other - base) for `field` over the common pair_ids, ascending.

    Args:
        base: Baseline pair_id -> record mapping.
        other: Comparison pair_id -> record mapping.
        field: Key to read from each record (e.g. ``"pesq_nb_est"``).

    Returns:
        Tuple of (differences array, sorted list of pair_ids used).

    Raises:
        ValueError: If the pair_id sets of `base` and `other` differ.
    """
    base_ids = set(base.keys())
    other_ids = set(other.keys())
    if base_ids != other_ids:
        missing_in_other = sorted(base_ids - other_ids)
        missing_in_base = sorted(other_ids - base_ids)
        raise ValueError(
            "pair_id sets differ between base and other: "
            f"missing_in_other={missing_in_other}, missing_in_base={missing_in_base}"
        )
    pair_ids = sorted(base_ids)
    d = np.array(
        [other[pid][field] - base[pid][field] for pid in pair_ids], dtype=np.float64
    )
    return d, pair_ids


def sign_test(d: np.ndarray) -> dict:
    """Two-sided sign test over a vector of paired differences.

    Returns:
        Dict with 'n_pos', 'n_neg', 'n_zero', 'prop_improve', 'p_value'.
    """
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))
    n_nonzero = n_pos + n_neg
    prop_improve = n_pos / n_nonzero if n_nonzero > 0 else float("nan")
    result = stats.binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided")
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "prop_improve": prop_improve,
        "p_value": float(result.pvalue),
    }


def bootstrap_ci(
    d: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    seed: int = SEED,
    n_resamples: int = N_RESAMPLES,
) -> dict:
    """BCa bootstrap confidence interval for a statistic over paired differences.

    A fresh ``np.random.default_rng(seed)`` instance is created for this call
    only, so results do not depend on the order in which families/tests run.

    Returns:
        Dict with 'point', 'lo', 'hi', 'method', 'n_resamples', 'seed'.
    """
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (d,),
        statistic,
        method="BCa",
        n_resamples=n_resamples,
        confidence_level=0.95,
        random_state=rng,
    )
    return {
        "point": float(statistic(d)),
        "lo": float(res.confidence_interval.low),
        "hi": float(res.confidence_interval.high),
        "method": "BCa",
        "n_resamples": n_resamples,
        "seed": seed,
    }


def spearman_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided Spearman rank correlation.

    Returns:
        Dict with 'rho', 'p_value', 'n'.
    """
    rho, p_value = stats.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p_value), "n": int(len(x))}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Implements: sort ascending, p_adj[(k)] = min(1, max_{j<=k} (m-j+1) * p[(j)])
    (the running max enforces monotonicity), then restore the input order.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for k, idx in enumerate(order):
        raw = (m - k) * p_values[idx]
        running_max = max(running_max, raw)
        adjusted_sorted[k] = min(1.0, running_max)
    adjusted = [0.0] * m
    for k, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[k]
    return adjusted


def prop_broken(d: np.ndarray, threshold: float = BROKEN_THRESHOLD) -> float:
    """Fraction of paired differences strictly below `threshold`."""
    return float(np.mean(d < threshold))


def broken_set(
    d: np.ndarray, pair_ids: list[int], threshold: float = BROKEN_THRESHOLD
) -> set[int]:
    """pair_ids whose paired difference is strictly below `threshold`."""
    return {pid for pid, value in zip(pair_ids, d) if value < threshold}


def _quantile_cutoff(values: np.ndarray, quantile: float) -> float:
    """Single source of truth for the quantile cutoff used by `baseline_stratum`."""
    return float(np.quantile(values, quantile))


def baseline_stratum(
    base: dict[int, dict],
    pair_ids: list[int],
    quantile: float = BASELINE_QUANTILE,
    field: str = f"{PRIMARY_METRIC}_est",
) -> np.ndarray:
    """Boolean mask over `pair_ids`: True where the baseline value is >= the quantile."""
    values = np.array([base[pid][field] for pid in pair_ids], dtype=np.float64)
    cutoff = _quantile_cutoff(values, quantile)
    return values >= cutoff


def mcnemar_test(broken_a: np.ndarray, broken_b: np.ndarray) -> dict:
    """Paired test for two binary outcomes over the same units.

    b = #(broken in a, not in b); c = #(broken in b, not in a).
    p_value = binomtest(b, b + c, 0.5) two-sided; 1.0 when b + c == 0.

    Returns:
        Dict with 'b', 'c', 'n_discordant', 'p_value'.
    """
    a = np.asarray(broken_a, dtype=bool)
    b_arr = np.asarray(broken_b, dtype=bool)
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(b_arr & ~a))
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            stats.binomtest(b, n_discordant, 0.5, alternative="two-sided").pvalue
        )
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_value}


def overlap_test(set_a: set[int], set_b: set[int], n_total: int) -> dict:
    """Upper-tail hypergeometric test for set overlap.

    p_value = hypergeom.sf(observed - 1, n_total, len(set_a), len(set_b))

    Returns:
        Dict with 'observed', 'expected', 'n_a', 'n_b', 'p_value'.
    """
    n_a = len(set_a)
    n_b = len(set_b)
    observed = len(set_a & set_b)
    expected = n_a * n_b / n_total
    p_value = float(stats.hypergeom.sf(observed - 1, n_total, n_a, n_b))
    return {
        "observed": observed,
        "expected": float(expected),
        "n_a": n_a,
        "n_b": n_b,
        "p_value": p_value,
    }


def _all_numeric_nan_count(record: dict) -> int:
    """Count NaN values among numeric fields of a single all_pairs record."""
    count = 0
    for value in record.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and np.isnan(value):
                count += 1
    return count


def check_integrity(results_dir: Path, metadata_dir: Path) -> dict:
    """Check pair_id consistency, NaNs, and EN/ES metadata alignment.

    Returns:
        Dict with 'n_files', 'pair_ids_consistent', 'n_nan', 'snr_match_en_es',
        'noise_file_match_en_es', 'issues'.
    """
    issues: list[str] = []
    expected_ids = set(range(250))
    n_files = 0
    n_nan = 0
    pair_ids_consistent = True

    for variant in VARIANTS:
        for test_set in TEST_SETS:
            path = results_dir / f"{variant}_{test_set}.json"
            if not path.exists():
                issues.append(f"missing results file: {path}")
                pair_ids_consistent = False
                continue
            n_files += 1
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            pairs = data["all_pairs"]
            ids = {record["pair_id"] for record in pairs}
            if ids != expected_ids:
                pair_ids_consistent = False
                missing = sorted(expected_ids - ids)
                extra = sorted(ids - expected_ids)
                issues.append(
                    f"{path.name}: pair_id set mismatch (missing={missing}, extra={extra})"
                )
            for record in pairs:
                n_nan += _all_numeric_nan_count(record)

    v1_meta_path = metadata_dir / "test_v1_metadata.json"
    v2_meta_path = metadata_dir / "test_v2_metadata.json"
    with v1_meta_path.open(encoding="utf-8") as f:
        v1_meta = {p["id"]: p for p in json.load(f)["pairs"]}
    with v2_meta_path.open(encoding="utf-8") as f:
        v2_meta = {p["id"]: p for p in json.load(f)["pairs"]}

    meta_ids_v1 = set(v1_meta.keys())
    meta_ids_v2 = set(v2_meta.keys())
    if meta_ids_v1 != expected_ids:
        pair_ids_consistent = False
        issues.append(f"{v1_meta_path.name}: pair id set != 0..249")
    if meta_ids_v2 != expected_ids:
        pair_ids_consistent = False
        issues.append(f"{v2_meta_path.name}: pair id set != 0..249")

    common_ids = sorted(meta_ids_v1 & meta_ids_v2)
    snr_match_en_es = sum(
        1 for i in common_ids if v1_meta[i]["snr_db"] == v2_meta[i]["snr_db"]
    )
    noise_file_match_en_es = sum(
        1 for i in common_ids if v1_meta[i]["noise_file"] == v2_meta[i]["noise_file"]
    )
    bucket_mismatches = [
        i for i in common_ids if v1_meta[i]["bucket_idx"] != v2_meta[i]["bucket_idx"]
    ]
    if bucket_mismatches:
        issues.append(
            f"bucket_idx mismatch between v1/v2 metadata for pair_ids: {bucket_mismatches}"
        )

    return {
        "n_files": n_files,
        "pair_ids_consistent": pair_ids_consistent,
        "n_nan": n_nan,
        "snr_match_en_es": snr_match_en_es,
        "noise_file_match_en_es": noise_file_match_en_es,
        "issues": issues,
    }


def _mean_stat(d: np.ndarray) -> float:
    return float(np.mean(d))


def _summarize_paired_diff(d: np.ndarray, seed: int) -> dict:
    """Shared descriptive block for F2/F3: mean, median, skew, sign counts, CI."""
    sign = sign_test(d)
    ci = bootstrap_ci(d, _mean_stat, seed=seed)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "skew": float(stats.skew(d)),
        "n_pos": sign["n_pos"],
        "n_neg": sign["n_neg"],
        "n_zero": sign["n_zero"],
        "ci95": {"lo": ci["lo"], "hi": ci["hi"], "method": ci["method"]},
    }


def family_f1(results_dir: Path, seed: int = SEED) -> dict:
    """F1: SNR x language interaction, via Spearman monotonicity of PESQ-NB delta."""
    v1_en = load_pairs(results_dir / "v1_v1_en.json")
    v1_es = load_pairs(results_dir / "v1_v2_es.json")

    ids_en = sorted(v1_en.keys())
    ids_es = sorted(v1_es.keys())

    snr_en = np.array([v1_en[i]["snr_db"] for i in ids_en], dtype=np.float64)
    delta_en = np.array(
        [v1_en[i][f"{PRIMARY_METRIC}_delta"] for i in ids_en], dtype=np.float64
    )
    f1a = spearman_test(delta_en, snr_en)

    snr_es = np.array([v1_es[i]["snr_db"] for i in ids_es], dtype=np.float64)
    delta_es = np.array(
        [v1_es[i][f"{PRIMARY_METRIC}_delta"] for i in ids_es], dtype=np.float64
    )
    f1b = spearman_test(delta_es, snr_es)

    diff_delta, pair_ids = paired_diff(v1_es, v1_en, f"{PRIMARY_METRIC}_delta")
    snr_common = np.array([v1_en[i]["snr_db"] for i in pair_ids], dtype=np.float64)
    f1c = spearman_test(diff_delta, snr_common)

    p_raw = [f1a["p_value"], f1b["p_value"], f1c["p_value"]]
    p_adj = holm(p_raw)

    tests = [
        {
            "id": "F1.a",
            "label": "Delta PESQ-NB de V1 vs snr_db, en v1_v1_en",
            "test": "spearman",
            "statistic": {"rho": f1a["rho"], "n": f1a["n"]},
            "p_raw": f1a["p_value"],
            "p_holm": p_adj[0],
        },
        {
            "id": "F1.b",
            "label": "Delta PESQ-NB de V1 vs snr_db, en v1_v2_es",
            "test": "spearman",
            "statistic": {"rho": f1b["rho"], "n": f1b["n"]},
            "p_raw": f1b["p_value"],
            "p_holm": p_adj[1],
        },
        {
            "id": "F1.c",
            "label": "(Delta_EN - Delta_ES) vs snr_db, apareado por pair_id",
            "test": "spearman",
            "statistic": {"rho": f1c["rho"], "n": f1c["n"]},
            "p_raw": f1c["p_value"],
            "p_holm": p_adj[2],
        },
    ]

    return {
        "label": "Interaccion SNR x idioma",
        "estimand": "monotonia de Delta PESQ-NB respecto al SNR",
        "correction": "holm",
        "exploratory": False,
        "tests": tests,
    }


def family_f2(results_dir: Path, seed: int = SEED) -> dict:
    """F2: Spanish adaptation gain (v3/v3b/v3e vs v1) on v2_es, PESQ-NB est."""
    v1_es = load_pairs(results_dir / "v1_v2_es.json")

    p_raw = []
    per_variant = {}
    for variant in ("v3", "v3b", "v3e"):
        other = load_pairs(results_dir / f"{variant}_v2_es.json")
        d, _ = paired_diff(v1_es, other, f"{PRIMARY_METRIC}_est")
        sign = sign_test(d)
        summary = _summarize_paired_diff(d, seed=seed)
        p_raw.append(sign["p_value"])
        per_variant[variant] = {"d": d, "sign": sign, "summary": summary}

    p_adj = holm(p_raw)

    tests = []
    for idx, variant in enumerate(("v3", "v3b", "v3e")):
        info = per_variant[variant]
        tests.append(
            {
                "id": f"F2.{variant}",
                "label": f"{variant}_v2_es - v1_v2_es, PESQ-NB est",
                "test": "sign+bootstrap_ci",
                "statistic": {
                    "n_pos": info["sign"]["n_pos"],
                    "n_neg": info["sign"]["n_neg"],
                    "n_zero": info["sign"]["n_zero"],
                    "prop_improve": info["sign"]["prop_improve"],
                },
                "p_raw": info["sign"]["p_value"],
                "p_holm": p_adj[idx],
                "mean": info["summary"]["mean"],
                "median": info["summary"]["median"],
                "skew": info["summary"]["skew"],
                "n_pos": info["summary"]["n_pos"],
                "n_neg": info["summary"]["n_neg"],
                "n_zero": info["summary"]["n_zero"],
                "ci95": info["summary"]["ci95"],
            }
        )

    return {
        "label": "Adaptacion (ganancia en ES)",
        "estimand": "P(mejora) y media del cambio",
        "correction": "holm",
        "exploratory": False,
        "tests": tests,
    }


def family_f3(results_dir: Path, seed: int = SEED) -> dict:
    """F3: English forgetting cost (v3/v3b/v3e vs v1) on v1_en, PESQ-NB est."""
    v1_en = load_pairs(results_dir / "v1_v1_en.json")

    p_raw = []
    per_variant = {}
    for variant in ("v3", "v3b", "v3e"):
        other = load_pairs(results_dir / f"{variant}_v1_en.json")
        d, _ = paired_diff(v1_en, other, f"{PRIMARY_METRIC}_est")
        sign = sign_test(d)
        summary = _summarize_paired_diff(d, seed=seed)

        prop_broken_by_thr = {
            str(thr): prop_broken(d, thr) for thr in SENSITIVITY_THRESHOLDS
        }
        prop_broken_ci = bootstrap_ci(
            d, lambda arr: prop_broken(arr, BROKEN_THRESHOLD), seed=seed
        )
        p5 = float(np.percentile(d, 5))
        p10 = float(np.percentile(d, 10))

        p_raw.append(sign["p_value"])
        per_variant[variant] = {
            "sign": sign,
            "summary": summary,
            "tail": {
                "prop_broken": prop_broken_by_thr,
                "prop_broken_ci95": {
                    "lo": prop_broken_ci["lo"],
                    "hi": prop_broken_ci["hi"],
                },
                "p5": p5,
                "p10": p10,
            },
        }

    p_adj = holm(p_raw)

    tests = []
    for idx, variant in enumerate(("v3", "v3b", "v3e")):
        info = per_variant[variant]
        tests.append(
            {
                "id": f"F3.{variant}",
                "label": f"{variant}_v1_en - v1_v1_en, PESQ-NB est",
                "test": "binomial+bootstrap_ci",
                "statistic": {
                    "n_pos": info["sign"]["n_pos"],
                    "n_neg": info["sign"]["n_neg"],
                    "n_zero": info["sign"]["n_zero"],
                    "prop_improve": info["sign"]["prop_improve"],
                },
                "p_raw": info["sign"]["p_value"],
                "p_holm": p_adj[idx],
                "mean": info["summary"]["mean"],
                "median": info["summary"]["median"],
                "skew": info["summary"]["skew"],
                "n_pos": info["summary"]["n_pos"],
                "n_neg": info["summary"]["n_neg"],
                "n_zero": info["summary"]["n_zero"],
                "ci95": info["summary"]["ci95"],
                "tail": info["tail"],
            }
        )

    return {
        "label": "Olvido (costo en EN)",
        "estimand": "P(rotura), P5, media",
        "correction": "holm",
        "exploratory": False,
        "tests": tests,
    }


def family_f4(results_dir: Path, seed: int = SEED) -> dict:
    """F4: exploratory per-bucket profile for every variant x test_set, PESQ-NB delta.

    Descriptive only: mean per bucket with BCa CI on the bucket-level values of
    `{m}_delta`. No hypothesis test, no correction.

    Note (design choice, contract underspecified the field): the contract's F3
    row uses PESQ-NB *est* diffs against V1 to measure forgetting, but F4 is
    described as "the profile table that feeds the figure" and the existing
    dashboard's per-bucket figure plots delta PESQ-NB (noisy vs enhanced), not
    a diff against V1. Delta was chosen to match that existing convention.
    """
    entries = []
    for test_set in TEST_SETS:
        for variant in VARIANTS:
            pairs = load_pairs(results_dir / f"{variant}_{test_set}.json")
            ids = sorted(pairs.keys())
            buckets = sorted({pairs[i]["bucket_idx"] for i in ids})
            for bucket_idx in buckets:
                bucket_ids = [i for i in ids if pairs[i]["bucket_idx"] == bucket_idx]
                delta = np.array(
                    [pairs[i][f"{PRIMARY_METRIC}_delta"] for i in bucket_ids],
                    dtype=np.float64,
                )
                ci = bootstrap_ci(delta, _mean_stat, seed=seed)
                entries.append(
                    {
                        "variant": variant,
                        "test_set": test_set,
                        "bucket_idx": bucket_idx,
                        "n_pairs": len(bucket_ids),
                        "mean": ci["point"],
                        "ci95": {"lo": ci["lo"], "hi": ci["hi"], "method": ci["method"]},
                    }
                )

    return {
        "label": "Perfil por bucket",
        "estimand": "medias por bucket de PESQ-NB delta (est - noisy)",
        "correction": None,
        "exploratory": True,
        "entries": entries,
    }


def _mannwhitney_bca_diff(
    values_a: np.ndarray, values_b: np.ndarray, seed: int = SEED
) -> dict:
    """Mann-Whitney U test plus a BCa CI on the difference of means (a - b).

    Returns:
        Dict with 'n_a', 'n_b', 'mean_a', 'mean_b', 'mean_diff', 'ci95', 'p_raw'.
    """
    n_a = len(values_a)
    n_b = len(values_b)

    if n_a > 0 and n_b > 0:
        p_raw = float(
            stats.mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue
        )
    else:
        p_raw = float("nan")

    def _mean_diff_stat(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    if n_a >= 2 and n_b >= 2:
        rng = np.random.default_rng(seed)
        res = stats.bootstrap(
            (values_a, values_b),
            _mean_diff_stat,
            method="BCa",
            n_resamples=N_RESAMPLES,
            confidence_level=0.95,
            random_state=rng,
        )
        ci_lo = float(res.confidence_interval.low)
        ci_hi = float(res.confidence_interval.high)
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")

    mean_a = float(np.mean(values_a)) if n_a else float("nan")
    mean_b = float(np.mean(values_b)) if n_b else float("nan")

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_a - mean_b,
        "ci95": {"lo": ci_lo, "hi": ci_hi, "method": "BCa"},
        "p_raw": p_raw,
    }


def family_f5(results_dir: Path, seed: int = SEED) -> dict:
    """F5: tail-of-forgetting control against regression to the mean.

    Compares the treatments' (v3/v3b/v3e) broken-file fraction on v1_en against a
    paired placebo control (McNemar, F5.a), checks whether the treatments' broken
    sets overlap more than chance (hypergeometric, F5.b), measures the association
    between V1's baseline PESQ-NB and being broken (Mann-Whitney + BCa CI, F5.c),
    and repeats the F5.a comparison restricted to the top quartile of V1's baseline
    PESQ-NB (McNemar, F5.d).
    """
    v1_en = load_pairs(results_dir / "v1_v1_en.json")

    treatment_diffs: dict[str, tuple[np.ndarray, list[int]]] = {}
    for variant in TREATMENTS:
        other = load_pairs(results_dir / f"{variant}_v1_en.json")
        d, pair_ids = paired_diff(v1_en, other, f"{PRIMARY_METRIC}_est")
        treatment_diffs[variant] = (d, pair_ids)

    control_diffs: dict[str, tuple[np.ndarray, list[int]]] = {}
    for name, rel_path in CONTROL_PATHS.items():
        other = load_pairs(PROJECT_ROOT / rel_path)
        d, pair_ids = paired_diff(v1_en, other, f"{PRIMARY_METRIC}_est")
        control_diffs[name] = (d, pair_ids)

    control_d, control_pair_ids = control_diffs[PRIMARY_CONTROL]
    control_broken = control_d < BROKEN_THRESHOLD
    n_total = len(control_pair_ids)

    f5a_tests = []
    f5a_p_raw = []
    treatment_broken_full: dict[str, np.ndarray] = {}
    for variant in TREATMENTS:
        d, pair_ids = treatment_diffs[variant]
        if pair_ids != control_pair_ids:
            raise ValueError(
                f"pair_id ordering mismatch between treatment {variant!r} and "
                f"control {PRIMARY_CONTROL!r}"
            )
        treatment_broken = d < BROKEN_THRESHOLD
        treatment_broken_full[variant] = treatment_broken
        mcnemar = mcnemar_test(treatment_broken, control_broken)
        f5a_p_raw.append(mcnemar["p_value"])
        f5a_tests.append(
            {
                "id": f"F5.a.{variant}",
                "label": f"{variant}_v1_en roto vs {PRIMARY_CONTROL} roto "
                          f"(McNemar, mismos pair_id)",
                "test": "mcnemar",
                "statistic": {
                    "b": mcnemar["b"],
                    "c": mcnemar["c"],
                    "n_discordant": mcnemar["n_discordant"],
                    "prop_treatment": prop_broken(d),
                    "prop_control": prop_broken(control_d),
                },
                "p_raw": mcnemar["p_value"],
            }
        )

    broken_sets = {
        variant: broken_set(d, pair_ids)
        for variant, (d, pair_ids) in treatment_diffs.items()
    }

    f5b_tests = []
    f5b_p_raw = []
    for a, b in combinations(TREATMENTS, 2):
        overlap = overlap_test(broken_sets[a], broken_sets[b], n_total)
        f5b_p_raw.append(overlap["p_value"])
        f5b_tests.append(
            {
                "id": f"F5.b.{a}_{b}",
                "label": f"Solapamiento de broken_set({a}) y broken_set({b})",
                "test": "hypergeometric",
                "statistic": {
                    "observed": overlap["observed"],
                    "expected": overlap["expected"],
                    "n_a": overlap["n_a"],
                    "n_b": overlap["n_b"],
                },
                "p_raw": overlap["p_value"],
            }
        )

    baseline_mask = baseline_stratum(v1_en, control_pair_ids, quantile=BASELINE_QUANTILE)
    v1_scores_all = np.array(
        [v1_en[pid][f"{PRIMARY_METRIC}_est"] for pid in control_pair_ids],
        dtype=np.float64,
    )
    baseline_cutoff = _quantile_cutoff(v1_scores_all, BASELINE_QUANTILE)
    n_stratum = int(np.sum(baseline_mask))

    f5d_tests = []
    f5d_p_raw = []
    for variant in TREATMENTS:
        d, _ = treatment_diffs[variant]
        treatment_broken_stratum = treatment_broken_full[variant][baseline_mask]
        control_broken_stratum = control_broken[baseline_mask]
        mcnemar = mcnemar_test(treatment_broken_stratum, control_broken_stratum)
        f5d_p_raw.append(mcnemar["p_value"])
        f5d_tests.append(
            {
                "id": f"F5.d.{variant}",
                "label": f"{variant}_v1_en roto vs {PRIMARY_CONTROL} roto "
                          f"(McNemar, estrato PESQ-NB V1 >= cuantil "
                          f"{BASELINE_QUANTILE})",
                "test": "mcnemar_stratified",
                "statistic": {
                    "b": mcnemar["b"],
                    "c": mcnemar["c"],
                    "n_discordant": mcnemar["n_discordant"],
                    "prop_treatment": prop_broken(d[baseline_mask]),
                    "prop_control": prop_broken(control_d[baseline_mask]),
                    "n_stratum": n_stratum,
                    "baseline_cutoff": baseline_cutoff,
                },
                "p_raw": mcnemar["p_value"],
            }
        )

    p_raw_all = f5a_p_raw + f5b_p_raw + f5d_p_raw
    p_holm_all = holm(p_raw_all)
    for test, p_holm in zip(f5a_tests + f5b_tests + f5d_tests, p_holm_all):
        test["p_holm"] = p_holm

    broken_all_three = set.intersection(*broken_sets.values())
    union_broken = set.union(*broken_sets.values())
    never_broken_ids = [pid for pid in control_pair_ids if pid not in union_broken]
    rest_of_triple_ids = [
        pid for pid in control_pair_ids if pid not in broken_all_three
    ]

    def _v1_scores(ids: list[int]) -> np.ndarray:
        return np.array(
            [v1_en[pid][f"{PRIMARY_METRIC}_est"] for pid in ids], dtype=np.float64
        )

    union_scores = _v1_scores(sorted(union_broken))
    never_scores = _v1_scores(never_broken_ids)
    triple_scores = _v1_scores(sorted(broken_all_three))
    rest_scores = _v1_scores(rest_of_triple_ids)

    confounder = {
        "id": "F5.c",
        "test": "mannwhitney+bootstrap_ci",
        "corrected": False,
        "primary": {
            "group": "union_broken_vs_never_broken",
            **_mannwhitney_bca_diff(union_scores, never_scores, seed=seed),
        },
        "sensitivity": [
            {
                "group": "triple_vs_rest",
                **_mannwhitney_bca_diff(triple_scores, rest_scores, seed=seed),
            },
            {
                "group": "triple_vs_never_broken",
                **_mannwhitney_bca_diff(triple_scores, never_scores, seed=seed),
            },
        ],
    }

    quartile_edges = np.quantile(v1_scores_all, [0.25, 0.5, 0.75])
    quartile_idx = np.digitize(v1_scores_all, quartile_edges, right=False) + 1

    stratified_profile = []
    for variant in TREATMENTS:
        d, _ = treatment_diffs[variant]
        for q in (1, 2, 3, 4):
            mask_q = quartile_idx == q
            stratified_profile.append(
                {
                    "series": variant,
                    "role": "treatment",
                    "quartile": q,
                    "n": int(np.sum(mask_q)),
                    "prop_broken": prop_broken(d[mask_q]),
                }
            )
    for name in CONTROL_PATHS:
        d, _ = control_diffs[name]
        for q in (1, 2, 3, 4):
            mask_q = quartile_idx == q
            stratified_profile.append(
                {
                    "series": name,
                    "role": "control",
                    "quartile": q,
                    "n": int(np.sum(mask_q)),
                    "prop_broken": prop_broken(d[mask_q]),
                }
            )

    sensitivity = []
    for variant in TREATMENTS:
        d, _ = treatment_diffs[variant]
        sensitivity.append(
            {
                "series": variant,
                "role": "treatment",
                "prop_broken": {
                    str(thr): prop_broken(d, thr) for thr in SENSITIVITY_THRESHOLDS
                },
            }
        )
    for name in CONTROL_PATHS:
        d, _ = control_diffs[name]
        sensitivity.append(
            {
                "series": name,
                "role": "control",
                "prop_broken": {
                    str(thr): prop_broken(d, thr) for thr in SENSITIVITY_THRESHOLDS
                },
            }
        )

    return {
        "label": "Control de la cola (olvido vs regresion a la media)",
        "estimand": "exceso de rotura del tratamiento sobre el control pareado; "
                    "solapamiento de los conjuntos rotos entre tratamientos; "
                    "confusor medido (nivel basal de V1) y su control estratificado",
        "correction": "holm",
        "exploratory": False,
        "primary_control": PRIMARY_CONTROL,
        "threshold": BROKEN_THRESHOLD,
        "tests": f5a_tests + f5b_tests + f5d_tests,
        "confounder": confounder,
        "stratified_profile": stratified_profile,
        "sensitivity": sensitivity,
    }



# ---------------------------------------------------------------------------
# F6 -- control de idioma vs canal sobre test_v3_mls_es.
#
# Responde a Wang et al. 2022 (Interspeech), que concluyen que en estas tareas
# el canal de grabación domina y el idioma es despreciable. La comparación
# EN/ES del proyecto confundía ambos: LibriSpeech es audiolibro, Common Voice
# es crowdsourced. test_v3_mls_es fija el canal (audiolibro) y varía el idioma.
#
# TODA la inferencia de esta familia se agrupa por hablante. Son ~6 pares por
# cada uno de los 40 hablantes, así que los pares NO son independientes y un
# bootstrap sobre pares sería sobre-confiado: el N efectivo está más cerca de
# 40 que de 250. Los intervalos salen más anchos que los de F1-F5 y eso es
# correcto, no un defecto a corregir.
#
# Predicciones y umbrales fijados en el preregistro ANTES de que el test set
# existiera (hash en docs/preregistro_mls_es.sha256).
# ---------------------------------------------------------------------------

CHANNEL_TEST_SET = "v3_mls_es"
CHANNEL_METADATA = "test_v3_mls_es_metadata.json"

# Umbrales preregistrados. No se tocan: cambiarlos después de ver el dato
# invalidaría el ejercicio.
P1_LANGUAGE_RHO = -0.15   # rho <= este valor, con IC sin cruzar cero -> idioma
P1_CHANNEL_RHO = -0.10    # rho >= este valor -> canal
P2_LANGUAGE_MEAN = 0.10   # media > este valor Y prop > P2_LANGUAGE_PROP -> idioma
P2_LANGUAGE_PROP = 0.70
P2_CHANNEL_MEAN = 0.03    # media <= este valor O prop <= P2_CHANNEL_PROP -> canal
P2_CHANNEL_PROP = 0.55


def p1_outcome(rho: float, ci_hi: float) -> str:
    """Regla de decisión de P1, tal como quedó fijada en el preregistro.

    El orden de las ramas importa. El caso "rho suficientemente negativo pero el
    IC agrupado incluye cero" se resuelve como INDETERMINADO POR POTENCIA y
    explícitamente NO como canal: con 40 clusters el intervalo puede ser ancho
    sin que eso sea evidencia a favor del canal.
    """
    if rho <= P1_LANGUAGE_RHO:
        return "idioma" if ci_hi < 0 else "indeterminado por potencia"
    if rho >= P1_CHANNEL_RHO:
        return "canal"
    return "indeterminado"


def p2_outcome(mean_gain: float, prop_improve: float) -> str:
    """Regla de decisión de P2, tal como quedó fijada en el preregistro."""
    if prop_improve > P2_LANGUAGE_PROP and mean_gain > P2_LANGUAGE_MEAN:
        return "idioma"
    if prop_improve <= P2_CHANNEL_PROP or mean_gain <= P2_CHANNEL_MEAN:
        return "canal"
    return "parcial"


def load_speaker_ids(metadata_path: Path) -> dict[int, str]:
    """pair_id -> speaker_id, desde la metadata del sellado."""
    with open(metadata_path) as f:
        return {p["id"]: str(p["speaker_id"]) for p in json.load(f)["pairs"]}


def clustered_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    seed: int = SEED,
    n_resamples: int = N_RESAMPLES,
) -> dict:
    """IC95 percentil remuestreando CLUSTERS (hablantes), no observaciones.

    Se usa percentil y no BCa a propósito: BCa necesita jackknife sobre las
    unidades de remuestreo, y con 40 clusters su corrección de aceleración es
    inestable. El percentil sobre clusters es el estimador honesto acá.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    draws = []
    for _ in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        value = statistic(idx)
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws)
    return {
        "point": float(statistic(np.arange(len(values)))),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "method": "percentil, remuestreo por hablante",
        "n_clusters": int(len(unique)),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def family_f6(results_dir: Path, metadata_dir: Path, seed: int = SEED) -> dict:
    """Control de idioma vs canal, con inferencia agrupada por hablante."""
    speakers = load_speaker_ids(metadata_dir / CHANNEL_METADATA)
    mls = {v: load_pairs(results_dir / f"{v}_{CHANNEL_TEST_SET}.json")
           for v in ("v1", "v3e", "v5")}
    ids = sorted(mls["v1"])
    clusters = np.array([speakers[i] for i in ids])
    snr = np.array([mls["v1"][i]["snr_db"] for i in ids], dtype=np.float64)

    # --- P1: pendiente de V1 contra el SNR, en los tres test sets ---
    slopes = []
    for test_set, language, channel in (
        ("v1_en", "inglés", "audiolibro"),
        ("v2_es", "español", "crowdsourced"),
        (CHANNEL_TEST_SET, "español", "audiolibro"),
    ):
        pairs = load_pairs(results_dir / f"v1_{test_set}.json")
        order = sorted(pairs)
        x = np.array([pairs[i]["snr_db"] for i in order], dtype=np.float64)
        y = np.array([pairs[i][f"{PRIMARY_METRIC}_delta"] for i in order], dtype=np.float64)
        spear = spearman_test(x, y)
        # El p de scipy trata los 250 pares como independientes. En
        # test_v3_mls_es NO lo son (~6 por hablante), así que ahí queda
        # renombrado para que nadie lo lea como si fuera válido: la
        # inferencia de ese set es el IC agrupado, no este p.
        entry = {"test_set": test_set, "language": language, "channel": channel,
                 "rho": spear["rho"], "n": spear["n"]}
        if test_set == CHANNEL_TEST_SET:
            entry["p_value_unclustered"] = spear["p_value"]
            entry["p_value_note"] = ("sobre-confiado: ignora el agrupamiento por "
                                     "hablante. Usar ci95 para inferir.")
        else:
            entry["p_value"] = spear["p_value"]
        if test_set == CHANNEL_TEST_SET:
            entry["ci95"] = clustered_bootstrap_ci(
                y, clusters,
                lambda idx: stats.spearmanr(x[idx], y[idx]).correlation,
                seed=seed)
        slopes.append(entry)

    control = next(s for s in slopes if s["test_set"] == CHANNEL_TEST_SET)
    rho, hi = control["rho"], control["ci95"]["hi"]
    outcome_p1 = p1_outcome(rho, hi)

    # --- P2: ¿la ganancia del fine-tuning transfiere de un canal al otro? ---
    gain = np.array([mls["v3e"][i][f"{PRIMARY_METRIC}_est"]
                     - mls["v1"][i][f"{PRIMARY_METRIC}_est"] for i in ids])
    cv = {v: load_pairs(results_dir / f"{v}_v2_es.json") for v in ("v1", "v3e")}
    cv_ids = sorted(cv["v1"])
    gain_cv = float(np.mean([cv["v3e"][i][f"{PRIMARY_METRIC}_est"]
                             - cv["v1"][i][f"{PRIMARY_METRIC}_est"] for i in cv_ids]))
    mean_ci = clustered_bootstrap_ci(gain, clusters, lambda idx: float(gain[idx].mean()), seed=seed)
    prop_ci = clustered_bootstrap_ci(gain, clusters, lambda idx: float((gain[idx] > 0).mean()), seed=seed)
    mean_gain, prop = mean_ci["point"], prop_ci["point"]
    outcome_p2 = p2_outcome(mean_gain, prop)

    # --- Retención por receta: cuánto de la ganancia sobrevive al cambio de canal ---
    retention = []
    for variant in ("v3e", "v5"):
        cv_v = load_pairs(results_dir / f"{variant}_v2_es.json")
        g_cv = float(np.mean([cv_v[i][f"{PRIMARY_METRIC}_est"]
                              - cv["v1"][i][f"{PRIMARY_METRIC}_est"] for i in cv_ids]))
        g_ml = float(np.mean([mls[variant][i][f"{PRIMARY_METRIC}_est"]
                              - mls["v1"][i][f"{PRIMARY_METRIC}_est"] for i in ids]))
        retention.append({"variant": variant, "gain_crowdsourced": g_cv,
                          "gain_audiobook": g_ml,
                          "retained": g_ml / g_cv if g_cv else float("nan")})

    return {
        "label": "Control de idioma vs canal (test_v3_mls_es)",
        "estimand": "pendiente de la ganancia contra el SNR, y transferencia de la "
                    "ganancia del fine-tuning entre canales",
        "correction": None,
        "exploratory": False,
        "preregistered": "docs/preregistro_mls_es.sha256",
        "inference": "bootstrap percentil agrupado por hablante (los pares no son "
                     "independientes: ~6 por cada uno de 40 hablantes)",
        "n_speakers": int(len(np.unique(clusters))),
        "p1": {"id": "P1", "label": "¿La pendiente contra el SNR sigue al idioma "
                                    "o al canal?",
               "slopes": slopes, "outcome": outcome_p1,
               "thresholds": {"language_rho": P1_LANGUAGE_RHO,
                              "channel_rho": P1_CHANNEL_RHO}},
        "p2": {"id": "P2", "label": "¿La ganancia del fine-tuning transfiere entre canales?",
               "gain_crowdsourced": gain_cv,
               "gain_audiobook": mean_ci, "prop_improve": prop_ci,
               "outcome": outcome_p2,
               "thresholds": {"language_mean": P2_LANGUAGE_MEAN,
                              "language_prop": P2_LANGUAGE_PROP,
                              "channel_mean": P2_CHANNEL_MEAN,
                              "channel_prop": P2_CHANNEL_PROP}},
        "retention_by_recipe": retention,
    }


def _write_markdown(payload: dict, out_md: Path) -> None:
    """Render the JSON payload as a human-readable Spanish Markdown report."""
    lines: list[str] = []
    lines.append("# Reanalisis estadistico")
    lines.append("")
    lines.append(f"Generado: {payload['generated_at']}")
    lines.append(f"Seed: {payload['seed']} | n_resamples: {payload['n_resamples']} | "
                 f"metrica principal: {payload['primary_metric']} | "
                 f"umbral de rotura: {payload['broken_threshold']}")
    lines.append("")

    integrity = payload["integrity"]
    lines.append("## Integridad de los datos")
    lines.append("")
    lines.append(f"- Archivos de resultados leidos: {integrity['n_files']}")
    lines.append(f"- Conjuntos de pair_id consistentes (0..249): {integrity['pair_ids_consistent']}")
    lines.append(f"- NaN encontrados en campos numericos: {integrity['n_nan']}")
    lines.append(f"- Pares con snr_db coincidente entre metadata EN/ES: "
                 f"{integrity['snr_match_en_es']} / 250")
    lines.append(f"- Pares con noise_file coincidente entre metadata EN/ES: "
                 f"{integrity['noise_file_match_en_es']} / 250")
    if integrity["issues"]:
        lines.append("- Issues:")
        for issue in integrity["issues"]:
            lines.append(f"  - {issue}")
    else:
        lines.append("- Issues: ninguno")
    lines.append("")

    f1 = payload["families"]["F1"]
    lines.append("## F1 — Interaccion SNR x idioma")
    lines.append("")
    lines.append(f"Estimando: {f1['estimand']}. Correccion: {f1['correction']} "
                  f"(dentro de la familia, m={len(f1['tests'])}).")
    lines.append("")
    lines.append("| id | descripcion | rho | n | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|")
    for t in f1["tests"]:
        lines.append(
            f"| {t['id']} | {t['label']} | {t['statistic']['rho']:.4f} | "
            f"{t['statistic']['n']} | {t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")

    f2 = payload["families"]["F2"]
    lines.append("## F2 — Adaptacion (ganancia en ES)")
    lines.append("")
    lines.append(f"Estimando: {f2['estimand']}. Correccion: {f2['correction']} "
                  f"(dentro de la familia, m={len(f2['tests'])}).")
    lines.append("")
    lines.append("| id | comparacion | prop_improve | n_pos | n_neg | n_zero | "
                  "media | mediana | skew | IC95 media | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in f2["tests"]:
        ci = t["ci95"]
        lines.append(
            f"| {t['id']} | {t['label']} | {t['statistic']['prop_improve']:.4f} | "
            f"{t['statistic']['n_pos']} | {t['statistic']['n_neg']} | "
            f"{t['statistic']['n_zero']} | {t['mean']:.4f} | {t['median']:.4f} | "
            f"{t['skew']:.4f} | [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"{t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")

    f3 = payload["families"]["F3"]
    lines.append("## F3 — Olvido (costo en EN)")
    lines.append("")
    lines.append(f"Estimando: {f3['estimand']}. Correccion: {f3['correction']} "
                  f"(dentro de la familia, m={len(f3['tests'])}).")
    lines.append("")
    lines.append("| id | comparacion | prop_improve | n_pos | n_neg | n_zero | "
                  "media | mediana | skew | IC95 media | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in f3["tests"]:
        ci = t["ci95"]
        lines.append(
            f"| {t['id']} | {t['label']} | {t['statistic']['prop_improve']:.4f} | "
            f"{t['statistic']['n_pos']} | {t['statistic']['n_neg']} | "
            f"{t['statistic']['n_zero']} | {t['mean']:.4f} | {t['median']:.4f} | "
            f"{t['skew']:.4f} | [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"{t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")
    lines.append("Cola de la distribucion de diferencias (proporcion de pares por debajo "
                  "de cada umbral, P5 y P10):")
    lines.append("")
    lines.append("| id | prop_broken(-0.1) | prop_broken(-0.2) | prop_broken(-0.3) | "
                  "prop_broken(-0.5) | IC95 prop_broken(-0.2) | P5 | P10 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in f3["tests"]:
        tail = t["tail"]
        pb = tail["prop_broken"]
        ci = tail["prop_broken_ci95"]
        lines.append(
            f"| {t['id']} | {pb['-0.1']:.4f} | {pb['-0.2']:.4f} | {pb['-0.3']:.4f} | "
            f"{pb['-0.5']:.4f} | [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"{tail['p5']:.4f} | {tail['p10']:.4f} |"
        )
    lines.append("")

    f4 = payload["families"]["F4"]
    lines.append("## F4 — Perfil por bucket (EXPLORATORIO, sin correccion)")
    lines.append("")
    lines.append(f"Estimando: {f4['estimand']}. **No se corrige por Holm**: esta tabla no "
                  "se usa para afirmar significancia, es el perfil descriptivo que alimenta "
                  "la figura.")
    lines.append("")
    lines.append("| variante | test_set | bucket_idx | n_pares | media PESQ-NB delta | IC95 |")
    lines.append("|---|---|---|---|---|---|")
    for e in f4["entries"]:
        ci = e["ci95"]
        lines.append(
            f"| {e['variant']} | {e['test_set']} | {e['bucket_idx']} | {e['n_pairs']} | "
            f"{e['mean']:.4f} | [{ci['lo']:.4f}, {ci['hi']:.4f}] |"
        )
    lines.append("")

    f5 = payload["families"]["F5"]
    lines.append("## F5 — Control de la cola (olvido vs regresion a la media)")
    lines.append("")
    lines.append(f"Estimando: {f5['estimand']}. Correccion: {f5['correction']} "
                  f"(dentro de la familia, m=9: 3 McNemar + 3 hipergeometrico + "
                  f"3 McNemar estratificado; F5.c queda afuera, sin correccion). "
                  f"Control primario: {f5['primary_control']}. "
                  f"Umbral de rotura: {f5['threshold']}.")
    lines.append("")

    f5a = [t for t in f5["tests"] if t["test"] == "mcnemar"]
    lines.append("### F5.a — McNemar contra el control primario")
    lines.append("")
    lines.append("| id | descripcion | b | c | n_discordant | prop_treatment | "
                  "prop_control | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in f5a:
        s = t["statistic"]
        lines.append(
            f"| {t['id']} | {t['label']} | {s['b']} | {s['c']} | {s['n_discordant']} | "
            f"{s['prop_treatment']:.4f} | {s['prop_control']:.4f} | "
            f"{t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")

    f5b = [t for t in f5["tests"] if t["test"] == "hypergeometric"]
    lines.append("### F5.b — Solapamiento entre pares de tratamientos (hipergeometrica)")
    lines.append("")
    lines.append("| id | descripcion | observado | esperado | n_a | n_b | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in f5b:
        s = t["statistic"]
        lines.append(
            f"| {t['id']} | {t['label']} | {s['observed']} | {s['expected']:.4f} | "
            f"{s['n_a']} | {s['n_b']} | {t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")

    f5d = [t for t in f5["tests"] if t["test"] == "mcnemar_stratified"]
    lines.append("### F5.d — McNemar contra el control primario, estrato PESQ-NB V1 alto")
    lines.append("")
    lines.append("| id | descripcion | b | c | n_discordant | prop_treatment | "
                  "prop_control | n_estrato | corte basal | p crudo | p Holm |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in f5d:
        s = t["statistic"]
        lines.append(
            f"| {t['id']} | {t['label']} | {s['b']} | {s['c']} | {s['n_discordant']} | "
            f"{s['prop_treatment']:.4f} | {s['prop_control']:.4f} | {s['n_stratum']} | "
            f"{s['baseline_cutoff']:.4f} | {t['p_raw']:.4g} | {t['p_holm']:.4g} |"
        )
    lines.append("")

    conf = f5["confounder"]
    lines.append("### F5.c — Confusor medido (nivel basal de V1), SIN corregir")
    lines.append("")
    lines.append("| grupo | n_a | n_b | media_a | media_b | diferencia | IC95 BCa | p crudo |")
    lines.append("|---|---|---|---|---|---|---|---|")
    p = conf["primary"]
    lines.append(
        f"| {p['group']} | {p['n_a']} | {p['n_b']} | {p['mean_a']:.4f} | "
        f"{p['mean_b']:.4f} | {p['mean_diff']:.4f} | "
        f"[{p['ci95']['lo']:.4f}, {p['ci95']['hi']:.4f}] | {p['p_raw']:.4g} |"
    )
    for s in conf["sensitivity"]:
        lines.append(
            f"| {s['group']} | {s['n_a']} | {s['n_b']} | {s['mean_a']:.4f} | "
            f"{s['mean_b']:.4f} | {s['mean_diff']:.4f} | "
            f"[{s['ci95']['lo']:.4f}, {s['ci95']['hi']:.4f}] | {s['p_raw']:.4g} |"
        )
    lines.append("")
    lines.append(f"(corrected={conf['corrected']})")
    lines.append("")

    lines.append("### F5 — Perfil estratificado (7 series x 4 cuartiles de PESQ-NB basal de V1)")
    lines.append("")
    lines.append("| serie | rol | cuartil | n | prop_broken |")
    lines.append("|---|---|---|---|---|")
    for entry in f5["stratified_profile"]:
        lines.append(
            f"| {entry['series']} | {entry['role']} | {entry['quartile']} | "
            f"{entry['n']} | {entry['prop_broken']:.4f} |"
        )
    lines.append("")

    lines.append("### F5 — Sensibilidad de prop_broken por umbral (7 series x 4 umbrales)")
    lines.append("")
    lines.append("| serie | rol | prop_broken(-0.1) | prop_broken(-0.2) | "
                  "prop_broken(-0.3) | prop_broken(-0.5) |")
    lines.append("|---|---|---|---|---|---|")
    for entry in f5["sensitivity"]:
        pb = entry["prop_broken"]
        lines.append(
            f"| {entry['series']} | {entry['role']} | {pb['-0.1']:.4f} | "
            f"{pb['-0.2']:.4f} | {pb['-0.3']:.4f} | {pb['-0.5']:.4f} |"
        )
    lines.append("")

    f6 = payload["families"].get("F6")
    if f6:
        lines.append("")
        lines.append("## F6 - Control de idioma vs canal (preregistrado)")
        lines.append("")
        lines.append(f"Inferencia: {f6['inference']}")
        lines.append(f"Hablantes: {f6['n_speakers']} | Preregistro: {f6['preregistered']}")
        lines.append("")
        lines.append(f"### P1 - desenlace: **{f6['p1']['outcome'].upper()}**")
        lines.append("")
        lines.append("| test set | idioma | canal | rho | inferencia |")
        lines.append("|---|---|---|---|---|")
        for sl in f6["p1"]["slopes"]:
            if "ci95" in sl:
                inf = (f"IC95 agrupado [{sl['ci95']['lo']:+.3f}, {sl['ci95']['hi']:+.3f}]"
                       f" (p sin agrupar {sl['p_value_unclustered']:.2e}, sobre-confiado)")
            else:
                inf = f"p = {sl['p_value']:.2e}"
            lines.append(f"| `{sl['test_set']}` | {sl['language']} | {sl['channel']} |"
                         f" {sl['rho']:+.3f} | {inf} |")
        p2 = f6["p2"]
        lines.append("")
        lines.append(f"### P2 - desenlace: **{p2['outcome'].upper()}**")
        lines.append("")
        lines.append(f"- Ganancia de V3e sobre V1, canal crowdsourced: {p2['gain_crowdsourced']:+.3f}")
        lines.append(f"- Canal audiolibro: {p2['gain_audiobook']['point']:+.3f}"
                     f" (IC95 agrupado [{p2['gain_audiobook']['lo']:+.3f}, {p2['gain_audiobook']['hi']:+.3f}])")
        lines.append(f"- Proporcion que mejora: {p2['prop_improve']['point']:.3f}"
                     f" (IC95 agrupado [{p2['prop_improve']['lo']:.3f}, {p2['prop_improve']['hi']:.3f}])")
        lines.append("")
        lines.append("### Retencion de la ganancia entre canales, por receta")
        lines.append("")
        lines.append("| variante | crowdsourced | audiolibro | retiene |")
        lines.append("|---|---|---|---|")
        for r in f6["retention_by_recipe"]:
            lines.append(f"| {r['variant']} | {r['gain_crowdsourced']:+.3f} |"
                         f" {r['gain_audiobook']:+.3f} | {100*r['retained']:.0f}% |")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def run_all(
    results_dir: Path = PROJECT_ROOT / "results",
    metadata_dir: Path = PROJECT_ROOT / "seal_test_metadata",
    out_json: Path = PROJECT_ROOT / "results" / "reanalysis_stats.json",
    out_md: Path = PROJECT_ROOT / "docs" / "reanalisis_estadistico.md",
    seed: int = SEED,
) -> dict:
    """Run every family, write JSON + Markdown, return the JSON payload."""
    integrity = check_integrity(results_dir, metadata_dir)
    if integrity["n_nan"] > 0:
        logger.warning("Found %d NaN values in numeric fields", integrity["n_nan"])
    if not integrity["pair_ids_consistent"]:
        logger.warning("pair_id sets are not consistent across inputs: %s", integrity["issues"])

    inputs = {
        f"{variant}_{test_set}": f"results/{variant}_{test_set}.json"
        for variant in VARIANTS
        for test_set in TEST_SETS
    }

    families = {
        "F1": family_f1(results_dir, seed=seed),
        "F2": family_f2(results_dir, seed=seed),
        "F3": family_f3(results_dir, seed=seed),
        "F4": family_f4(results_dir, seed=seed),
        "F5": family_f5(results_dir, seed=seed),
        "F6": family_f6(results_dir, metadata_dir, seed=seed),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_resamples": N_RESAMPLES,
        "primary_metric": PRIMARY_METRIC,
        "broken_threshold": BROKEN_THRESHOLD,
        "inputs": inputs,
        "integrity": integrity,
        "families": families,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(payload, out_md)

    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
