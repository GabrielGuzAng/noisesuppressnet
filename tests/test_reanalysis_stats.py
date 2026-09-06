"""Independent validation suite for analysis/reanalysis_stats.py.

Written against the contract in docs/PROMPTS_REANALISIS.md section 4, without
reading the implementation first (stat_validator role, T2). Five layers:

1. Reference reproduction: recompute all_pairs aggregates with plain numpy and
   compare against the stored by_bucket/global values (anchors the pipeline
   against numbers already committed to the repo).
2. Pairing integrity: pair_id completeness, snr_db/bucket_idx agreement
   between the two sealed metadata files, and absence of NaN.
3. Known-answer tests on synthetic data with hand-computed expected results.
4. Calibration under the null: rejection rate of the sign test on data with
   no effect must fall inside the exact binomial interval.
5. Determinism: two runs of run_all() with the same seed must be byte-for-byte
   identical except for generated_at.

This file must not modify analysis/reanalysis_stats.py, must not weaken a
failing assert, and must not build layer-1 reference values using functions
from the module under test.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_DIR = PROJECT_ROOT / "seal_test_metadata"

VARIANTS = ["v1", "v2", "v3", "v3b", "v3e"]
TEST_SETS = ["v1_en", "v2_es"]
RESULT_FILES = [f"{variant}_{test_set}.json" for variant in VARIANTS for test_set in TEST_SETS]

PESQ_STOI_ATOL = 1e-12
SISDR_RTOL = 1e-5

try:
    import analysis.reanalysis_stats as rs

    IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - we want to report any import failure, not hide it
    rs = None
    IMPORT_ERROR = exc


def _require_module() -> None:
    if rs is None:
        pytest.fail(
            f"analysis.reanalysis_stats could not be imported: {IMPORT_ERROR!r}. "
            "Tests are written against the contract regardless; this is a hard "
            "failure, not a skip."
        )


@pytest.fixture(scope="session", autouse=True)
def _check_module_importable():
    _require_module()


# ---------------------------------------------------------------------------
# Helpers used ONLY to load raw JSON with plain stdlib/numpy, never through
# the module under test. This is what makes layer 1 an independent check.
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_results_data() -> dict[str, dict]:
    return {name: _load_json(RESULTS_DIR / name) for name in RESULT_FILES}


# ===========================================================================
# LAYER 1 — reference reproduction against by_bucket / global
# ===========================================================================

_BUCKET_METRICS = ["pesq_nb", "stoi", "sisdr"]
_GLOBAL_METRICS = ["pesq_nb", "pesq_wb", "stoi", "sisdr"]


def _bucket_cases():
    cases = []
    for fname in RESULT_FILES:
        for bucket_idx in range(5):
            for metric in _BUCKET_METRICS:
                cases.append((fname, bucket_idx, metric))
    return cases


def _global_cases():
    cases = []
    for fname in RESULT_FILES:
        for metric in _GLOBAL_METRICS:
            cases.append((fname, metric))
    return cases


@pytest.mark.parametrize("fname,bucket_idx,metric", _bucket_cases())
def test_layer1_bucket_delta_mean_matches_recomputation(fname, bucket_idx, metric):
    data = _load_json(RESULTS_DIR / fname)
    field = f"{metric}_delta"
    deltas = np.array(
        [p[field] for p in data["all_pairs"] if p["bucket_idx"] == bucket_idx],
        dtype=np.float64,
    )
    assert deltas.size > 0, f"{fname}: bucket {bucket_idx} has no pairs"
    recomputed_mean = float(np.mean(deltas))

    bucket_entry = next(b for b in data["by_bucket"] if b["bucket_idx"] == bucket_idx)
    stored_mean = bucket_entry[f"{metric}_delta_mean"]

    if metric == "sisdr":
        assert recomputed_mean == pytest.approx(stored_mean, rel=SISDR_RTOL)
    else:
        assert recomputed_mean == pytest.approx(stored_mean, abs=PESQ_STOI_ATOL)


@pytest.mark.parametrize("fname,metric", _global_cases())
def test_layer1_global_delta_mean_matches_recomputation(fname, metric):
    data = _load_json(RESULTS_DIR / fname)
    field = f"{metric}_delta"
    deltas = np.array([p[field] for p in data["all_pairs"]], dtype=np.float64)
    assert deltas.size == data["n_pairs_evaluated"]
    recomputed_mean = float(np.mean(deltas))

    stored_mean = data["global"][metric]["delta_mean"]

    if metric == "sisdr":
        assert recomputed_mean == pytest.approx(stored_mean, rel=SISDR_RTOL)
    else:
        assert recomputed_mean == pytest.approx(stored_mean, abs=PESQ_STOI_ATOL)


# ===========================================================================
# LAYER 2 — pairing integrity
# ===========================================================================


@pytest.mark.parametrize("fname", RESULT_FILES)
def test_layer2_pair_ids_are_exactly_0_to_249(fname):
    data = _load_json(RESULTS_DIR / fname)
    pair_ids = sorted(p["pair_id"] for p in data["all_pairs"])
    assert pair_ids == list(range(250)), (
        f"{fname}: pair_id set is not exactly 0..249 "
        f"(got {len(pair_ids)} entries, "
        f"missing={sorted(set(range(250)) - set(pair_ids))}, "
        f"extra={sorted(set(pair_ids) - set(range(250)))})"
    )


def test_layer2_snr_db_matches_between_en_and_es_metadata():
    meta_en = _load_json(METADATA_DIR / "test_v1_metadata.json")
    meta_es = _load_json(METADATA_DIR / "test_v2_metadata.json")
    en_by_id = {p["id"]: p for p in meta_en["pairs"]}
    es_by_id = {p["id"]: p for p in meta_es["pairs"]}

    assert set(en_by_id) == set(es_by_id) == set(range(250))

    mismatches = [
        pid
        for pid in range(250)
        if en_by_id[pid]["snr_db"] != es_by_id[pid]["snr_db"]
    ]
    assert mismatches == [], f"snr_db mismatch for pair_ids: {mismatches}"


def test_layer2_bucket_idx_matches_between_en_and_es_metadata():
    meta_en = _load_json(METADATA_DIR / "test_v1_metadata.json")
    meta_es = _load_json(METADATA_DIR / "test_v2_metadata.json")
    en_by_id = {p["id"]: p for p in meta_en["pairs"]}
    es_by_id = {p["id"]: p for p in meta_es["pairs"]}

    mismatches = [
        pid
        for pid in range(250)
        if en_by_id[pid]["bucket_idx"] != es_by_id[pid]["bucket_idx"]
    ]
    assert mismatches == [], f"bucket_idx mismatch for pair_ids: {mismatches}"


_NUMERIC_ALL_PAIRS_FIELDS = [
    "pair_id",
    "bucket_idx",
    "snr_db",
    "pesq_nb_noisy",
    "pesq_nb_est",
    "pesq_nb_delta",
    "pesq_wb_noisy",
    "pesq_wb_est",
    "pesq_wb_delta",
    "stoi_noisy",
    "stoi_est",
    "stoi_delta",
    "sisdr_noisy",
    "sisdr_est",
    "sisdr_delta",
]


@pytest.mark.parametrize("fname", RESULT_FILES)
def test_layer2_no_nan_in_numeric_fields(fname):
    data = _load_json(RESULTS_DIR / fname)
    nan_locations = [
        (p["pair_id"], field)
        for p in data["all_pairs"]
        for field in _NUMERIC_ALL_PAIRS_FIELDS
        if isinstance(p[field], float) and math.isnan(p[field])
    ]
    assert nan_locations == [], f"{fname}: NaN found at (pair_id, field) = {nan_locations}"


# ===========================================================================
# LAYER 3 — known-answer tests on synthetic data
# ===========================================================================


def test_layer3_bootstrap_ci_recovers_known_shift():
    rng_local = np.random.default_rng(123)
    shift = 2.0
    d = shift + rng_local.standard_normal(500) * 0.5

    result = rs.bootstrap_ci(d, np.mean, seed=42, n_resamples=2000)

    assert result["method"] == "BCa"
    assert result["n_resamples"] == 2000
    assert result["seed"] == 42
    assert result["point"] == pytest.approx(np.mean(d), abs=1e-9)
    assert result["lo"] < result["hi"]
    assert result["lo"] <= shift <= result["hi"], (
        f"true shift {shift} not inside CI [{result['lo']}, {result['hi']}]"
    )


def test_layer3_sign_test_known_proportion():
    d = np.array([1.0, 2.0, 3.0, 0.5, 4.0, 1.5, 2.5, -1.0, -2.0, -3.0, 0.0, 0.0])
    # 7 positive, 3 negative, 2 zero -> prop_improve = 7 / 10 = 0.7
    result = rs.sign_test(d)

    assert result["n_pos"] == 7
    assert result["n_neg"] == 3
    assert result["n_zero"] == 2
    assert result["prop_improve"] == pytest.approx(0.7, abs=1e-12)

    expected_p = stats.binomtest(7, 10, 0.5, alternative="two-sided").pvalue
    assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)


def test_layer3_spearman_monotonic_increasing():
    x = np.arange(10, dtype=np.float64)
    y = x**3  # strictly increasing, nonlinear -> rho must still be exactly 1.0

    result = rs.spearman_test(x, y)

    assert result["rho"] == pytest.approx(1.0, abs=1e-9)
    assert result["n"] == 10
    assert result["p_value"] < 0.01


def test_layer3_spearman_monotonic_decreasing():
    x = np.arange(10, dtype=np.float64)
    y = -(x**3)

    result = rs.spearman_test(x, y)

    assert result["rho"] == pytest.approx(-1.0, abs=1e-9)
    assert result["n"] == 10


def test_layer3_holm_single_pvalue_unchanged():
    result = rs.holm([0.03])
    assert result == pytest.approx([0.03], abs=1e-12)


def test_layer3_holm_matches_hand_computation_sorted_input():
    # m=4, already ascending: p_adj(k) = min(1, max_{j<=k} (m-j+1)*p(j))
    # j=1: 4*0.01=0.04 (max=0.04)
    # j=2: 3*0.02=0.06 (max=0.06)
    # j=3: 2*0.03=0.06 (max=0.06, running max wins over raw 0.06)
    # j=4: 1*0.04=0.04 (max=0.06, forced up by monotonicity)
    p_values = [0.01, 0.02, 0.03, 0.04]
    expected = [0.04, 0.06, 0.06, 0.06]
    result = rs.holm(p_values)
    assert result == pytest.approx(expected, abs=1e-12)


def test_layer3_holm_monotonicity_forces_intermediate_value_up():
    # m=3, input order [idx0, idx1, idx2] = [0.01, 0.04, 0.03]
    # sorted ascending: p(1)=0.01 (idx0), p(2)=0.03 (idx2), p(3)=0.04 (idx1)
    # j=1: 3*0.01=0.03 (max=0.03)
    # j=2: 2*0.03=0.06 (max=0.06)
    # j=3: 1*0.04=0.04 -> forced up to running max 0.06 (monotonicity correction)
    # mapped back to original order: idx0=0.03, idx1=0.06, idx2=0.06
    p_values = [0.01, 0.04, 0.03]
    expected = [0.03, 0.06, 0.06]
    result = rs.holm(p_values)
    assert result == pytest.approx(expected, abs=1e-12)


def test_layer3_holm_caps_at_one():
    # m=2, [0.6, 0.7]: j=1 -> 2*0.6=1.2 (capped to 1.0); j=2 -> max(1.2, 1*0.7)=1.2 (capped to 1.0)
    p_values = [0.6, 0.7]
    expected = [1.0, 1.0]
    result = rs.holm(p_values)
    assert result == pytest.approx(expected, abs=1e-12)


def test_layer3_paired_diff_raises_on_mismatched_pair_ids():
    base = {0: {"m": 1.0}, 1: {"m": 2.0}, 2: {"m": 3.0}}
    other = {0: {"m": 1.5}, 1: {"m": 2.5}, 3: {"m": 4.0}}
    with pytest.raises(ValueError):
        rs.paired_diff(base, other, "m")


def test_layer3_paired_diff_returns_other_minus_base_ascending():
    base = {2: {"m": 10.0}, 0: {"m": 1.0}, 1: {"m": 5.0}}
    other = {0: {"m": 2.0}, 1: {"m": 4.0}, 2: {"m": 20.0}}
    d, pair_ids = rs.paired_diff(base, other, "m")

    assert pair_ids == [0, 1, 2]
    assert list(d) == pytest.approx([1.0, -1.0, 10.0], abs=1e-12)


# ===========================================================================
# LAYER 4 — calibration under the null
# ===========================================================================


def test_layer4_sign_test_calibration_under_null():
    rng_local = np.random.default_rng(20260905)
    n_datasets = 200
    n_pairs = 250
    alpha = 0.05

    rejections = 0
    for _ in range(n_datasets):
        d = rng_local.standard_normal(n_pairs)  # symmetric around 0: null is true
        result = rs.sign_test(d)
        if result["p_value"] < alpha:
            rejections += 1

    lo, hi = stats.binom.interval(0.99, n_datasets, alpha)
    assert lo <= rejections <= hi, (
        f"rejection rate under the null ({rejections}/{n_datasets}) falls outside the "
        f"exact 99% binomial interval [{lo}, {hi}] for p=0.05, n=200 -- "
        "possible miscalibration of sign_test (wrong tails, wrong denominator, "
        "or a paired test applied where independence was assumed)"
    )


# ===========================================================================
# LAYER 5 — determinism of run_all()
# ===========================================================================


def _strip_generated_at(payload: dict) -> dict:
    stripped = json.loads(json.dumps(payload))
    stripped.pop("generated_at", None)
    return stripped


def test_layer5_run_all_is_deterministic_across_two_runs(tmp_path):
    out_dir_a = tmp_path / "run_a"
    out_dir_b = tmp_path / "run_b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()

    payload_a = rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=out_dir_a / "reanalysis_stats.json",
        out_md=out_dir_a / "reanalisis_estadistico.md",
        seed=42,
    )
    payload_b = rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=out_dir_b / "reanalysis_stats.json",
        out_md=out_dir_b / "reanalisis_estadistico.md",
        seed=42,
    )

    assert _strip_generated_at(payload_a) == _strip_generated_at(payload_b)

    written_a = _load_json(out_dir_a / "reanalysis_stats.json")
    written_b = _load_json(out_dir_b / "reanalysis_stats.json")
    assert _strip_generated_at(written_a) == _strip_generated_at(written_b)


# ===========================================================================
# LAYER 6 — composition (T2b, contract 4.4-bis / 4.5-bis)
#
# Every expected value below is recomputed from the raw results/*.json and
# seal_test_metadata/*.json files with plain numpy/scipy. family_f1..f4,
# check_integrity, paired_diff and load_pairs are only ever used to fetch the
# OBSERVED value being checked, never to build the expected one (except 6.2,
# where paired_diff itself is the function under test).
# ===========================================================================


def _raw_field_by_pair_id(fname: str, field: str) -> dict[int, float]:
    data = _load_json(RESULTS_DIR / fname)
    return {p["pair_id"]: p[field] for p in data["all_pairs"]}


def _entry_by_id(entries: list[dict], entry_id: str) -> dict:
    matches = [e for e in entries if e["id"] == entry_id]
    assert len(matches) == 1, (
        f"expected exactly one entry with id={entry_id!r}, found {len(matches)}"
    )
    return matches[0]


def _holm_reference(p_values: list[float]) -> list[float]:
    """Independent re-implementation of Holm-Bonferroni, per contract 4.2.

    Deliberately duplicated here instead of importing rs.holm: the point of
    this layer is to check the module's composition against an outside
    computation, not against itself.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    running_max = 0.0
    adjusted_sorted = []
    for rank, idx in enumerate(order, start=1):
        candidate = (m - rank + 1) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted_sorted.append(min(1.0, running_max))
    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return result


def _assert_no_numpy_types(obj, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_no_numpy_types(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _assert_no_numpy_types(value, f"{path}[{i}]")
    else:
        assert type(obj) not in (
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.intp,
            np.bool_,
        ), f"numpy scalar type leaked into JSON payload at {path}: {type(obj)}"


def test_layer6_api_constants_match_contract():
    assert tuple(rs.VARIANTS) == ("v1", "v2", "v3", "v3b", "v3e")
    assert tuple(rs.TEST_SETS) == ("v1_en", "v2_es")
    assert rs.PRIMARY_METRIC == "pesq_nb"


# ---------------------------------------------------------------------------
# 6.1 — F1: monotonicity of Delta PESQ-NB vs SNR
# ---------------------------------------------------------------------------


def test_layer6_1_family_f1_a_matches_recomputation():
    pairs = _load_json(RESULTS_DIR / "v1_v1_en.json")["all_pairs"]
    x = np.array([p["snr_db"] for p in pairs], dtype=np.float64)
    y = np.array([p["pesq_nb_delta"] for p in pairs], dtype=np.float64)
    rho, p_value = stats.spearmanr(x, y)

    f1 = rs.family_f1(RESULTS_DIR)
    entry = _entry_by_id(f1["tests"], "F1.a")

    assert entry["statistic"]["n"] == 250
    assert entry["statistic"]["rho"] == pytest.approx(float(rho), abs=1e-12)
    assert entry["p_raw"] == pytest.approx(float(p_value), rel=1e-9)


def test_layer6_1_family_f1_b_matches_recomputation():
    pairs = _load_json(RESULTS_DIR / "v1_v2_es.json")["all_pairs"]
    x = np.array([p["snr_db"] for p in pairs], dtype=np.float64)
    y = np.array([p["pesq_nb_delta"] for p in pairs], dtype=np.float64)
    rho, p_value = stats.spearmanr(x, y)

    f1 = rs.family_f1(RESULTS_DIR)
    entry = _entry_by_id(f1["tests"], "F1.b")

    assert entry["statistic"]["n"] == 250
    assert entry["statistic"]["rho"] == pytest.approx(float(rho), abs=1e-12)
    assert entry["p_raw"] == pytest.approx(float(p_value), rel=1e-9)


def test_layer6_1_family_f1_c_matches_recomputation():
    delta_en = _raw_field_by_pair_id("v1_v1_en.json", "pesq_nb_delta")
    delta_es = _raw_field_by_pair_id("v1_v2_es.json", "pesq_nb_delta")
    snr_en = _raw_field_by_pair_id("v1_v1_en.json", "snr_db")

    common_ids = sorted(set(delta_en) & set(delta_es))
    assert common_ids == list(range(250))

    diff = np.array([delta_en[i] - delta_es[i] for i in common_ids], dtype=np.float64)
    snr = np.array([snr_en[i] for i in common_ids], dtype=np.float64)
    rho, p_value = stats.spearmanr(snr, diff)

    f1 = rs.family_f1(RESULTS_DIR)
    entry = _entry_by_id(f1["tests"], "F1.c")

    assert entry["statistic"]["n"] == 250
    assert entry["statistic"]["rho"] == pytest.approx(float(rho), abs=1e-12)
    assert entry["p_raw"] == pytest.approx(float(p_value), rel=1e-9)


# ---------------------------------------------------------------------------
# 6.2 — paired_diff pairs by pair_id, not by dict insertion order
# ---------------------------------------------------------------------------


def test_layer6_2_paired_diff_ignores_insertion_order():
    base_forward = {0: {"m": 10.0}, 1: {"m": 20.0}, 2: {"m": 30.0}}
    other_forward = {0: {"m": 11.0}, 1: {"m": 19.0}, 2: {"m": 33.0}}
    base_reversed = dict(reversed(list(base_forward.items())))
    other_reversed = dict(reversed(list(other_forward.items())))

    d_forward, ids_forward = rs.paired_diff(base_forward, other_forward, "m")
    d_reversed, ids_reversed = rs.paired_diff(base_reversed, other_reversed, "m")

    assert ids_forward == [0, 1, 2]
    assert ids_reversed == [0, 1, 2]
    assert list(d_forward) == pytest.approx(list(d_reversed), abs=1e-12)
    assert list(d_forward) == pytest.approx([1.0, -1.0, 3.0], abs=1e-12)


# ---------------------------------------------------------------------------
# 6.3 — F2 / F3: paired diff of PESQ-NB _est_ between variant and V1 baseline
# ---------------------------------------------------------------------------


def _recompute_f2_or_f3(base_fname: str, variant_fname: str) -> dict:
    base = _raw_field_by_pair_id(base_fname, "pesq_nb_est")
    variant = _raw_field_by_pair_id(variant_fname, "pesq_nb_est")
    common_ids = sorted(set(base) & set(variant))
    assert common_ids == list(range(250))

    d = np.array([variant[i] - base[i] for i in common_ids], dtype=np.float64)
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))
    prop_improve = n_pos / (n_pos + n_neg)
    p_raw = stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue

    return {
        "d": d,
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "skew": float(stats.skew(d)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "prop_improve": prop_improve,
        "p_raw": float(p_raw),
    }


@pytest.mark.parametrize("variant", ["v3", "v3b", "v3e"])
def test_layer6_3_family_f2_matches_recomputation(variant):
    expected = _recompute_f2_or_f3("v1_v2_es.json", f"{variant}_v2_es.json")

    f2 = rs.family_f2(RESULTS_DIR)
    entry = _entry_by_id(f2["tests"], f"F2.{variant}")

    assert entry["mean"] == pytest.approx(expected["mean"], abs=1e-12)
    assert entry["median"] == pytest.approx(expected["median"], abs=1e-12)
    assert entry["skew"] == pytest.approx(expected["skew"], abs=1e-12)
    assert entry["n_pos"] == expected["n_pos"]
    assert entry["n_neg"] == expected["n_neg"]
    assert entry["n_zero"] == expected["n_zero"]
    assert entry["statistic"]["n_pos"] == expected["n_pos"]
    assert entry["statistic"]["n_neg"] == expected["n_neg"]
    assert entry["statistic"]["n_zero"] == expected["n_zero"]
    assert entry["statistic"]["prop_improve"] == pytest.approx(expected["prop_improve"], abs=1e-12)
    assert entry["p_raw"] == pytest.approx(expected["p_raw"], rel=1e-9)


@pytest.mark.parametrize("variant", ["v3", "v3b", "v3e"])
def test_layer6_3_family_f3_matches_recomputation(variant):
    expected = _recompute_f2_or_f3("v1_v1_en.json", f"{variant}_v1_en.json")

    f3 = rs.family_f3(RESULTS_DIR)
    entry = _entry_by_id(f3["tests"], f"F3.{variant}")

    assert entry["mean"] == pytest.approx(expected["mean"], abs=1e-12)
    assert entry["median"] == pytest.approx(expected["median"], abs=1e-12)
    assert entry["skew"] == pytest.approx(expected["skew"], abs=1e-12)
    assert entry["n_pos"] == expected["n_pos"]
    assert entry["n_neg"] == expected["n_neg"]
    assert entry["n_zero"] == expected["n_zero"]
    assert entry["statistic"]["n_pos"] == expected["n_pos"]
    assert entry["statistic"]["n_neg"] == expected["n_neg"]
    assert entry["statistic"]["n_zero"] == expected["n_zero"]
    assert entry["statistic"]["prop_improve"] == pytest.approx(expected["prop_improve"], abs=1e-12)
    assert entry["p_raw"] == pytest.approx(expected["p_raw"], rel=1e-9)


# ---------------------------------------------------------------------------
# 6.4 — F3 tail block: prop_broken at 4 thresholds, P5, P10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["v3", "v3b", "v3e"])
def test_layer6_4_family_f3_tail_matches_recomputation(variant):
    expected = _recompute_f2_or_f3("v1_v1_en.json", f"{variant}_v1_en.json")
    d = expected["d"]

    expected_prop_broken = {
        str(thr): float(np.mean(d < thr)) for thr in (-0.1, -0.2, -0.3, -0.5)
    }
    expected_p5 = float(np.percentile(d, 5))
    expected_p10 = float(np.percentile(d, 10))

    f3 = rs.family_f3(RESULTS_DIR)
    entry = _entry_by_id(f3["tests"], f"F3.{variant}")
    tail = entry["tail"]

    for thr_key, expected_value in expected_prop_broken.items():
        assert tail["prop_broken"][thr_key] == pytest.approx(expected_value, abs=1e-12), (
            f"{variant}: prop_broken at {thr_key} mismatch"
        )
    assert tail["p5"] == pytest.approx(expected_p5, abs=1e-12)
    assert tail["p10"] == pytest.approx(expected_p10, abs=1e-12)


# ---------------------------------------------------------------------------
# 6.5 — Holm correction is applied within each family, not across families
# ---------------------------------------------------------------------------


def test_layer6_5_holm_matches_hand_computation_within_each_family():
    f1 = rs.family_f1(RESULTS_DIR)
    f2 = rs.family_f2(RESULTS_DIR)
    f3 = rs.family_f3(RESULTS_DIR)

    for family_dict, ids in (
        (f1, ["F1.a", "F1.b", "F1.c"]),
        (f2, ["F2.v3", "F2.v3b", "F2.v3e"]),
        (f3, ["F3.v3", "F3.v3b", "F3.v3e"]),
    ):
        entries = [_entry_by_id(family_dict["tests"], entry_id) for entry_id in ids]
        p_raw = [e["p_raw"] for e in entries]
        expected_holm = _holm_reference(p_raw)
        observed_holm = [e["p_holm"] for e in entries]
        assert observed_holm == pytest.approx(expected_holm, abs=1e-12)


def test_layer6_5_holm_per_family_differs_from_holm_over_all_nine():
    f1 = rs.family_f1(RESULTS_DIR)
    f2 = rs.family_f2(RESULTS_DIR)
    f3 = rs.family_f3(RESULTS_DIR)

    all_ids = (
        [("F1", i) for i in ["F1.a", "F1.b", "F1.c"]]
        + [("F2", i) for i in ["F2.v3", "F2.v3b", "F2.v3e"]]
        + [("F3", i) for i in ["F3.v3", "F3.v3b", "F3.v3e"]]
    )
    family_dicts = {"F1": f1, "F2": f2, "F3": f3}
    all_entries = [
        _entry_by_id(family_dicts[fam]["tests"], entry_id) for fam, entry_id in all_ids
    ]

    combined_p_raw = [e["p_raw"] for e in all_entries]
    combined_holm_global = _holm_reference(combined_p_raw)
    stored_holm_per_family = [e["p_holm"] for e in all_entries]

    mismatches = [
        i
        for i, (a, b) in enumerate(zip(combined_holm_global, stored_holm_per_family))
        if abs(a - b) > 1e-9
    ]
    assert mismatches, (
        "Holm applied over all 9 p-values together gives the same result as Holm "
        "applied within each family separately -- this test can't tell family-wise "
        "correction apart from global correction, which means it isn't proof that "
        "the implementation corrects within families as the contract requires."
    )


# ---------------------------------------------------------------------------
# 6.6 — F4: 50 descriptive entries, bit-exact against by_bucket
# ---------------------------------------------------------------------------


def test_layer6_6_family_f4_has_50_entries_matching_the_full_grid():
    f4 = rs.family_f4(RESULTS_DIR)
    entries = f4["entries"]
    assert len(entries) == 50

    seen = set()
    for entry in entries:
        key = (entry["variant"], entry["test_set"], entry["bucket_idx"])
        assert key not in seen, f"duplicate entry for {key}"
        seen.add(key)
        assert entry["n_pairs"] == 50

    expected_keys = {
        (variant, test_set, bucket_idx)
        for variant in VARIANTS
        for test_set in TEST_SETS
        for bucket_idx in range(5)
    }
    assert seen == expected_keys


def test_layer6_6_family_f4_means_are_bit_exact_against_by_bucket():
    f4 = rs.family_f4(RESULTS_DIR)
    for entry in f4["entries"]:
        fname = f"{entry['variant']}_{entry['test_set']}.json"
        raw = _load_json(RESULTS_DIR / fname)
        bucket_entry = next(
            b for b in raw["by_bucket"] if b["bucket_idx"] == entry["bucket_idx"]
        )
        assert entry["mean"] == pytest.approx(
            bucket_entry["pesq_nb_delta_mean"], abs=1e-12
        ), f"{fname} bucket {entry['bucket_idx']}"


# ---------------------------------------------------------------------------
# 6.7 — check_integrity against an independent recomputation
# ---------------------------------------------------------------------------


def test_layer6_7_check_integrity_matches_independent_recomputation():
    pair_ids_consistent = True
    n_nan = 0
    for fname in RESULT_FILES:
        data = _load_json(RESULTS_DIR / fname)
        pair_ids = sorted(p["pair_id"] for p in data["all_pairs"])
        if pair_ids != list(range(250)):
            pair_ids_consistent = False
        for p in data["all_pairs"]:
            for field in _NUMERIC_ALL_PAIRS_FIELDS:
                value = p[field]
                if isinstance(value, float) and math.isnan(value):
                    n_nan += 1

    meta_en = _load_json(METADATA_DIR / "test_v1_metadata.json")
    meta_es = _load_json(METADATA_DIR / "test_v2_metadata.json")
    en_by_id = {p["id"]: p for p in meta_en["pairs"]}
    es_by_id = {p["id"]: p for p in meta_es["pairs"]}
    assert set(en_by_id) == set(es_by_id) == set(range(250))

    snr_match = sum(
        1 for pid in range(250) if en_by_id[pid]["snr_db"] == es_by_id[pid]["snr_db"]
    )
    noise_file_match = sum(
        1
        for pid in range(250)
        if en_by_id[pid]["noise_file"] == es_by_id[pid]["noise_file"]
    )

    observed = rs.check_integrity(RESULTS_DIR, METADATA_DIR)

    assert observed["n_files"] == len(RESULT_FILES)
    assert observed["pair_ids_consistent"] == pair_ids_consistent
    assert observed["n_nan"] == n_nan
    assert observed["snr_match_en_es"] == snr_match
    assert observed["noise_file_match_en_es"] == noise_file_match
    if pair_ids_consistent and n_nan == 0:
        assert observed["issues"] == []


# ---------------------------------------------------------------------------
# 6.8 — schema and types of the run_all() payload
# ---------------------------------------------------------------------------


def test_layer6_8_run_all_payload_schema_and_types(tmp_path):
    payload = rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=tmp_path / "reanalysis_stats.json",
        out_md=tmp_path / "reanalisis_estadistico.md",
        seed=42,
    )

    top_level_keys = {
        "generated_at",
        "seed",
        "n_resamples",
        "primary_metric",
        "broken_threshold",
        "inputs",
        "integrity",
        "families",
    }
    assert top_level_keys <= payload.keys()
    assert payload["primary_metric"] == "pesq_nb"
    assert payload["seed"] == 42
    assert payload["n_resamples"] == 10000
    assert payload["broken_threshold"] == pytest.approx(-0.2)

    families = payload["families"]
    assert set(families.keys()) >= {"F1", "F2", "F3", "F4"}

    expected_test_name = {
        "F1": "spearman",
        "F2": "sign+bootstrap_ci",
        "F3": "binomial+bootstrap_ci",
    }
    for fam_id, test_name in expected_test_name.items():
        fam = families[fam_id]
        assert fam["correction"] == "holm"
        assert fam["exploratory"] is False
        assert "tests" in fam
        for entry in fam["tests"]:
            required_keys = {"id", "label", "test", "statistic", "p_raw", "p_holm"}
            assert required_keys <= entry.keys()
            assert entry["test"] == test_name

    f4 = families["F4"]
    assert f4["correction"] is None
    assert f4["exploratory"] is True
    assert "entries" in f4
    assert "tests" not in f4
    for entry in f4["entries"]:
        required_keys = {"variant", "test_set", "bucket_idx", "n_pairs", "mean", "ci95"}
        assert required_keys <= entry.keys()

    # Must be plain-JSON-serializable without a default= fallback, and must not
    # contain any numpy scalar types anywhere in the tree.
    json.dumps(payload)
    _assert_no_numpy_types(payload)


# ---------------------------------------------------------------------------
# 6.9 — bootstrap_ci with non-trivial statistics (median, prop_broken-style)
# ---------------------------------------------------------------------------


def test_layer6_9_bootstrap_ci_with_median_statistic():
    # A single draw checking "does the 95% CI contain the true value" is
    # statistically invalid on its own: a correctly implemented 95% CI is
    # expected to miss the true value on about 1 in 20 draws, independent of
    # whether bootstrap_ci is correct. This mirrors the reasoning behind
    # layer 4: check coverage RATE over repeated trials against the exact
    # binomial envelope, not a single hit-or-miss draw.
    true_median = 3.0
    n_trials = 40
    nominal_miss_rate = 0.05
    misses = 0

    for trial in range(n_trials):
        rng_trial = np.random.default_rng(1000 + trial)
        d = true_median + rng_trial.standard_normal(200) * 0.5
        result = rs.bootstrap_ci(d, np.median, seed=42, n_resamples=1000)
        assert result["lo"] < result["hi"]
        assert result["lo"] <= result["point"] <= result["hi"]
        if not (result["lo"] <= true_median <= result["hi"]):
            misses += 1

    lo_bound, hi_bound = stats.binom.interval(0.99, n_trials, nominal_miss_rate)
    assert lo_bound <= misses <= hi_bound, (
        f"CI miss rate for the median statistic ({misses}/{n_trials}) falls outside "
        f"the exact 99% binomial envelope [{lo_bound}, {hi_bound}] for a nominal "
        f"5% miss rate -- possible miscalibration of bootstrap_ci for a non-mean "
        f"statistic"
    )


def test_layer6_9_bootstrap_ci_with_prop_broken_statistic():
    rng_local = np.random.default_rng(99)
    thr = -0.2
    n = 1000
    true_prop = 0.3
    n_below = int(n * true_prop)

    below = rng_local.uniform(-1.0, thr - 0.01, size=n_below)
    above = rng_local.uniform(thr + 0.01, 1.0, size=n - n_below)
    d = np.concatenate([below, above])
    rng_local.shuffle(d)

    def prop_broken(x):
        return np.mean(x < thr)

    result = rs.bootstrap_ci(d, prop_broken, seed=42, n_resamples=2000)
    observed_prop = float(np.mean(d < thr))

    assert result["lo"] < result["hi"]
    assert result["lo"] <= result["point"] <= result["hi"]
    assert result["point"] == pytest.approx(observed_prop, abs=1e-9)
    assert result["lo"] <= true_prop <= result["hi"]


# ---------------------------------------------------------------------------
# 6.10 — determinism of the generated Markdown report (not just the JSON)
# ---------------------------------------------------------------------------


def test_layer6_10_markdown_identical_across_runs_except_timestamp(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=dir_a / "reanalysis_stats.json",
        out_md=dir_a / "reanalisis_estadistico.md",
        seed=42,
    )
    rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=dir_b / "reanalysis_stats.json",
        out_md=dir_b / "reanalisis_estadistico.md",
        seed=42,
    )

    lines_a = (dir_a / "reanalisis_estadistico.md").read_text(encoding="utf-8").splitlines()
    lines_b = (dir_b / "reanalisis_estadistico.md").read_text(encoding="utf-8").splitlines()
    assert len(lines_a) == len(lines_b), "the two runs produced .md files of different length"

    timestamp_like = re.compile(r"\d{4}-\d{2}-\d{2}|generad", re.IGNORECASE)
    differing = [i for i, (a, b) in enumerate(zip(lines_a, lines_b)) if a != b]
    for i in differing:
        assert timestamp_like.search(lines_a[i]) and timestamp_like.search(lines_b[i]), (
            f"line {i} differs between the two deterministic runs and does not look "
            f"like a timestamp line: {lines_a[i]!r} vs {lines_b[i]!r}"
        )


# ===========================================================================
# LAYER 7 — F5, the tail-of-forgetting control family (T2c, contract
# 4.4-ter / 4.5-ter). Same principle as layer 6: every expected value below
# is recomputed from raw JSON with numpy/scipy, never through family_f5,
# mcnemar_test, overlap_test, prop_broken or broken_set (except where one of
# those primitives is itself the thing under test, as in 6.2's paired_diff).
# ===========================================================================

# Literal API constants from contract 4.4-ter, not inferred statistics.
PRIMARY_CONTROL = "v4b_placebo_epoch_03"
CONTROL_REL_PATHS = {
    "v4b_placebo_epoch_01": "results/v4b/v4b_placebo_epoch_01_v1_en.json",
    "v4b_placebo_epoch_02": "results/v4b/v4b_placebo_epoch_02_v1_en.json",
    "v4b_placebo_epoch_03": "results/v4b/v4b_placebo_epoch_03_v1_en.json",
    "v2": "results/v2_v1_en.json",
}
CONTROL_ABS_PATHS = {name: PROJECT_ROOT / rel for name, rel in CONTROL_REL_PATHS.items()}
TREATMENTS = ("v3", "v3b", "v3e")
TREATMENT_ABS_PATHS = {v: RESULTS_DIR / f"{v}_v1_en.json" for v in TREATMENTS}
F5_THRESHOLD = -0.2


def test_layer7_api_constants_match_contract():
    assert rs.PRIMARY_CONTROL == PRIMARY_CONTROL
    assert dict(rs.CONTROL_PATHS) == CONTROL_REL_PATHS
    assert tuple(rs.TREATMENTS) == TREATMENTS


# ---------------------------------------------------------------------------
# 7.1 — no regression on F1-F4: the whole previous suite must still pass
# unmodified. There is no separate assertion to write for this beyond
# re-running layers 1-6 as-is; see the report for the explicit confirmation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7.2 — layer-1/layer-2 anchoring applied to the four new control files
# ---------------------------------------------------------------------------


def _control_bucket_cases():
    cases = []
    for name in CONTROL_ABS_PATHS:
        for bucket_idx in range(5):
            for metric in _BUCKET_METRICS:
                cases.append((name, bucket_idx, metric))
    return cases


def _control_global_cases():
    cases = []
    for name in CONTROL_ABS_PATHS:
        for metric in _GLOBAL_METRICS:
            cases.append((name, metric))
    return cases


@pytest.mark.parametrize("name,bucket_idx,metric", _control_bucket_cases())
def test_layer7_2_control_bucket_delta_mean_matches_recomputation(name, bucket_idx, metric):
    data = _load_json(CONTROL_ABS_PATHS[name])
    field = f"{metric}_delta"
    deltas = np.array(
        [p[field] for p in data["all_pairs"] if p["bucket_idx"] == bucket_idx],
        dtype=np.float64,
    )
    assert deltas.size > 0
    recomputed_mean = float(np.mean(deltas))
    bucket_entry = next(b for b in data["by_bucket"] if b["bucket_idx"] == bucket_idx)
    stored_mean = bucket_entry[f"{metric}_delta_mean"]

    if metric == "sisdr":
        assert recomputed_mean == pytest.approx(stored_mean, rel=SISDR_RTOL)
    else:
        assert recomputed_mean == pytest.approx(stored_mean, abs=PESQ_STOI_ATOL)


@pytest.mark.parametrize("name,metric", _control_global_cases())
def test_layer7_2_control_global_delta_mean_matches_recomputation(name, metric):
    data = _load_json(CONTROL_ABS_PATHS[name])
    field = f"{metric}_delta"
    deltas = np.array([p[field] for p in data["all_pairs"]], dtype=np.float64)
    assert deltas.size == data["n_pairs_evaluated"]
    recomputed_mean = float(np.mean(deltas))
    stored_mean = data["global"][metric]["delta_mean"]

    if metric == "sisdr":
        assert recomputed_mean == pytest.approx(stored_mean, rel=SISDR_RTOL)
    else:
        assert recomputed_mean == pytest.approx(stored_mean, abs=PESQ_STOI_ATOL)


@pytest.mark.parametrize("name", list(CONTROL_ABS_PATHS))
def test_layer7_2_control_pair_ids_are_exactly_0_to_249(name):
    data = _load_json(CONTROL_ABS_PATHS[name])
    pair_ids = sorted(p["pair_id"] for p in data["all_pairs"])
    assert pair_ids == list(range(250))


@pytest.mark.parametrize("name", list(CONTROL_ABS_PATHS))
def test_layer7_2_control_no_nan_in_numeric_fields(name):
    data = _load_json(CONTROL_ABS_PATHS[name])
    nan_locations = [
        (p["pair_id"], field)
        for p in data["all_pairs"]
        for field in _NUMERIC_ALL_PAIRS_FIELDS
        if isinstance(p[field], float) and math.isnan(p[field])
    ]
    assert nan_locations == []


# ---------------------------------------------------------------------------
# Shared helpers for 7.3-7.7: paired diff of pesq_nb_est against the V1/EN
# baseline, and "broken" sets/masks at a given threshold.
# ---------------------------------------------------------------------------


def _diff_est_vs_v1_en_baseline(path: Path, metric_field: str = "pesq_nb_est") -> np.ndarray:
    base = _raw_field_by_pair_id("v1_v1_en.json", metric_field)
    other_data = _load_json(path)
    other = {p["pair_id"]: p[metric_field] for p in other_data["all_pairs"]}
    common_ids = sorted(set(base) & set(other))
    assert common_ids == list(range(250))
    return np.array([other[i] - base[i] for i in common_ids], dtype=np.float64)


def _broken_mask(d: np.ndarray, threshold: float = F5_THRESHOLD) -> np.ndarray:
    return d < threshold


def _broken_set_from_diff(d: np.ndarray, threshold: float = F5_THRESHOLD) -> set[int]:
    return {pid for pid, value in enumerate(d) if value < threshold}


# ---------------------------------------------------------------------------
# Primitives: prop_broken / broken_set edge-case (strict "<", not "<=")
# ---------------------------------------------------------------------------


def test_layer7_prop_broken_is_strictly_less_than():
    d = np.array([-0.5, -0.2, -0.19, 0.0, 0.3, -1.0])
    result = rs.prop_broken(d, threshold=-0.2)
    # exactly -0.2 must NOT count as broken: only -0.5 and -1.0 qualify
    assert result == pytest.approx(2 / 6, abs=1e-12)


def test_layer7_broken_set_is_strictly_less_than():
    d = np.array([-0.5, -0.2, -0.19, 0.0, 0.3, -1.0])
    pair_ids = [10, 11, 12, 13, 14, 15]
    result = rs.broken_set(d, pair_ids, threshold=-0.2)
    assert result == {10, 15}


# ---------------------------------------------------------------------------
# 7.3 — McNemar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("treatment", TREATMENTS)
def test_layer7_3_family_f5a_mcnemar_matches_recomputation(treatment):
    d_control = _diff_est_vs_v1_en_baseline(CONTROL_ABS_PATHS[PRIMARY_CONTROL])
    d_treatment = _diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment])

    broken_control = _broken_mask(d_control)
    broken_treatment = _broken_mask(d_treatment)

    b = int(np.sum(broken_treatment & ~broken_control))
    c = int(np.sum(~broken_treatment & broken_control))
    n_discordant = b + c
    if n_discordant == 0:
        expected_p = 1.0
    else:
        expected_p = float(stats.binomtest(b, n_discordant, 0.5, alternative="two-sided").pvalue)
    prop_treatment = float(np.mean(broken_treatment))
    prop_control = float(np.mean(broken_control))

    f5 = rs.family_f5(RESULTS_DIR)
    entry = _entry_by_id(f5["tests"], f"F5.a.{treatment}")

    assert entry["statistic"]["b"] == b
    assert entry["statistic"]["c"] == c
    assert entry["statistic"]["n_discordant"] == n_discordant
    assert entry["statistic"]["prop_treatment"] == pytest.approx(prop_treatment, abs=1e-12)
    assert entry["statistic"]["prop_control"] == pytest.approx(prop_control, abs=1e-12)
    assert entry["p_raw"] == pytest.approx(expected_p, rel=1e-9)


def test_layer7_3_mcnemar_test_zero_discordant_gives_p_one():
    broken = np.array([True, False, True, False])
    result = rs.mcnemar_test(broken, broken.copy())
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_layer7_3_mcnemar_test_b_and_c_are_not_swapped():
    # broken in a-not-b happens 3 times; broken in b-not-a happens once.
    broken_a = np.array([True, True, True, True, False, False, False])
    broken_b = np.array([True, False, False, False, False, True, False])

    result = rs.mcnemar_test(broken_a, broken_b)

    assert result["b"] == 3
    assert result["c"] == 1
    assert result["n_discordant"] == 4
    expected_p = float(stats.binomtest(3, 4, 0.5, alternative="two-sided").pvalue)
    assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)


# ---------------------------------------------------------------------------
# 7.4 — hypergeometric overlap, with an explicit off-by-one detector
# ---------------------------------------------------------------------------


def test_layer7_4_overlap_test_off_by_one():
    # n_a = n_b = 5 out of n_total = 20, with FULL overlap (observed = 5).
    # 5 is the maximum possible overlap here, so sf(5, ...) == 0 exactly
    # (no probability mass strictly above the maximum), while the CORRECT
    # p-value sf(observed - 1, ...) = sf(4, ...) is P(X == 5), which is
    # small but clearly nonzero. A sf(observed, ...) bug would report 0.0.
    set_a = set(range(5))
    set_b = set(range(5))
    n_total = 20

    result = rs.overlap_test(set_a, set_b, n_total)

    correct_p = float(stats.hypergeom.sf(5 - 1, n_total, 5, 5))
    off_by_one_p = float(stats.hypergeom.sf(5, n_total, 5, 5))

    assert off_by_one_p == pytest.approx(0.0, abs=1e-12)
    assert correct_p > 1e-6
    assert result["observed"] == 5
    assert result["expected"] == pytest.approx(5 * 5 / n_total, abs=1e-12)
    assert result["p_value"] == pytest.approx(correct_p, rel=1e-9)
    assert abs(result["p_value"] - off_by_one_p) > 1e-6


@pytest.mark.parametrize("pair", list(itertools.combinations(TREATMENTS, 2)))
def test_layer7_4_family_f5b_hypergeometric_matches_recomputation(pair):
    treatment_a, treatment_b = pair
    d_a = _diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment_a])
    d_b = _diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment_b])
    set_a = _broken_set_from_diff(d_a)
    set_b = _broken_set_from_diff(d_b)
    n_total = 250

    observed = len(set_a & set_b)
    n_a, n_b = len(set_a), len(set_b)
    expected = n_a * n_b / n_total
    p_value = float(stats.hypergeom.sf(observed - 1, n_total, n_a, n_b))

    f5 = rs.family_f5(RESULTS_DIR)
    entry = _entry_by_id(f5["tests"], f"F5.b.{treatment_a}_{treatment_b}")

    assert entry["statistic"]["observed"] == observed
    assert entry["statistic"]["n_a"] == n_a
    assert entry["statistic"]["n_b"] == n_b
    assert entry["statistic"]["expected"] == pytest.approx(expected, abs=1e-12)
    assert entry["p_raw"] == pytest.approx(p_value, rel=1e-9)


# ---------------------------------------------------------------------------
# 7.5 and 7.6 — SUPERSEDED by amendment 3 (contract 4.4-quater/4.5-quater),
# not weakened. The original versions of these two tests asserted a schema
# that the contract itself retired: F5["negative_control"] with F5.c as a
# "no rejection wanted" negative control, and Holm applied over exactly the
# 6 F5.a+F5.b p-values. Amendment 3 replaces negative_control with
# F5["confounder"] (a different primary group definition) and requires Holm
# over 9 p-values (F5.a+F5.b+F5.d). Keeping the old asserts as-is would fail
# against a CORRECT post-amendment implementation, which is not a regression
# to report -- it is the old test enforcing a contract that no longer
# applies. Same principle T2b used when it replaced a single-draw CI check
# with a coverage-rate check: a superseded design is replaced and declared,
# not weakened. See test_layer8_5_* and test_layer8_2_* below for their
# amendment-3-aware replacements.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7.7 — sensitivity sweep: 7 series (3 treatments + 4 controls), 4 thresholds
# unaffected by amendment 3 (F5["sensitivity"] keeps its pre-amendment shape;
# only F5["tests"], F5["confounder"] and the new F5["stratified_profile"] change)
# ---------------------------------------------------------------------------


def test_layer7_7_sensitivity_has_seven_entries_with_correct_roles():
    f5 = rs.family_f5(RESULTS_DIR)
    sensitivity = f5["sensitivity"]
    assert len(sensitivity) == 7

    expected_roles = {t: "treatment" for t in TREATMENTS}
    expected_roles.update({c: "control" for c in CONTROL_ABS_PATHS})

    assert {e["series"] for e in sensitivity} == set(expected_roles)
    for entry in sensitivity:
        assert entry["role"] == expected_roles[entry["series"]]


def test_layer7_7_sensitivity_prop_broken_matches_recomputation():
    f5 = rs.family_f5(RESULTS_DIR)
    sensitivity = {e["series"]: e for e in f5["sensitivity"]}

    for series, path in {**TREATMENT_ABS_PATHS, **CONTROL_ABS_PATHS}.items():
        d = _diff_est_vs_v1_en_baseline(path)
        for thr in (-0.1, -0.2, -0.3, -0.5):
            expected_value = float(np.mean(d < thr))
            observed_value = sensitivity[series]["prop_broken"][str(thr)]
            assert observed_value == pytest.approx(expected_value, abs=1e-12), (
                f"{series} @ {thr}"
            )


# ---------------------------------------------------------------------------
# 7.8 — SUPERSEDED by amendment 3, same reasoning as 7.5/7.6 above: the old
# schema check required F5["negative_control"] and len(f5["tests"]) == 6.
# Amendment 3 retires both. See test_layer8_7_family_f5_schema_matches_amendment_3.
# ---------------------------------------------------------------------------


# ===========================================================================
# LAYER 8 — amendment 3 (T2d, contract 4.4-quater / 4.5-quater): F5.c is
# redefined from "negative control" to "measured confounder" with a new
# primary group (union of broken_set across the three treatments vs. never
# broken), F5.d is added as a stratified McNemar control restricted to the
# top quartile of V1's baseline PESQ-NB, and Holm within F5 now runs over 9
# p-values (F5.a + F5.b + F5.d) instead of 6. Same principle as layers 6-7:
# every expected value here is recomputed from raw JSON with numpy/scipy,
# never through family_f5, baseline_stratum, mcnemar_test or overlap_test,
# except where one of those primitives is itself the thing under test (as
# with baseline_stratum in 8.3, mirroring how 6.2 tested paired_diff itself).
# ===========================================================================

BASELINE_QUANTILE = 0.75


def _v1_est_array(pair_ids: list[int]) -> np.ndarray:
    v1_est = _raw_field_by_pair_id("v1_v1_en.json", "pesq_nb_est")
    return np.array([v1_est[i] for i in pair_ids], dtype=np.float64)


def _quartile_labels(values: np.ndarray) -> np.ndarray:
    """Assign each value to quartile 1-4 using np.quantile edges at .25/.5/.75.

    Half-open on the low side, closed on the high side of each bin
    (value >= edge belongs to the higher bin) -- this is the convention
    verified against results/reanalysis_stats.json before writing any
    assert against it: it reproduces the exact bin sizes (63/62/62/63) and
    the exact q75 cutoff that F5.d independently reports as baseline_cutoff.
    """
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    labels = np.empty(values.shape[0], dtype=int)
    labels[values < q25] = 1
    labels[(values >= q25) & (values < q50)] = 2
    labels[(values >= q50) & (values < q75)] = 3
    labels[values >= q75] = 4
    return labels


def test_layer8_api_constants_match_contract():
    assert rs.BASELINE_QUANTILE == pytest.approx(BASELINE_QUANTILE, abs=1e-12)


# ---------------------------------------------------------------------------
# 8.1 — no regression. F1-F4 are exercised independently by layers 1-6 and
# don't change here; what's checked explicitly below is that they still
# match after the F5 extension, and that within F5, F5.a/F5.b's statistic
# and p_raw match the exact same amendment-invariant recomputation as
# layers 7.3/7.4 (there is no separate "pre-amendment snapshot" to diff
# against in this session -- matching independent ground truth here IS the
# unchanged-since-before check, because that ground truth doesn't depend on
# F5.c/F5.d/Holm at all).
# ---------------------------------------------------------------------------


def test_layer8_1_f1_and_f4_still_match_recomputation_after_f5_extension():
    pairs = _load_json(RESULTS_DIR / "v1_v1_en.json")["all_pairs"]
    x = np.array([p["snr_db"] for p in pairs], dtype=np.float64)
    y = np.array([p["pesq_nb_delta"] for p in pairs], dtype=np.float64)
    rho, p_value = stats.spearmanr(x, y)

    f1 = rs.family_f1(RESULTS_DIR)
    entry = _entry_by_id(f1["tests"], "F1.a")
    assert entry["statistic"]["rho"] == pytest.approx(float(rho), abs=1e-12)
    assert entry["p_raw"] == pytest.approx(float(p_value), rel=1e-9)

    f4 = rs.family_f4(RESULTS_DIR)
    assert len(f4["entries"]) == 50
    for f4_entry in f4["entries"]:
        fname = f"{f4_entry['variant']}_{f4_entry['test_set']}.json"
        raw = _load_json(RESULTS_DIR / fname)
        bucket_entry = next(
            b for b in raw["by_bucket"] if b["bucket_idx"] == f4_entry["bucket_idx"]
        )
        assert f4_entry["mean"] == pytest.approx(
            bucket_entry["pesq_nb_delta_mean"], abs=1e-12
        )


def test_layer8_1_f5a_f5b_statistic_and_p_raw_amendment_invariant():
    f5 = rs.family_f5(RESULTS_DIR)

    for treatment in TREATMENTS:
        d_control = _diff_est_vs_v1_en_baseline(CONTROL_ABS_PATHS[PRIMARY_CONTROL])
        d_treatment = _diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment])
        broken_control = _broken_mask(d_control)
        broken_treatment = _broken_mask(d_treatment)
        b = int(np.sum(broken_treatment & ~broken_control))
        c = int(np.sum(~broken_treatment & broken_control))

        entry = _entry_by_id(f5["tests"], f"F5.a.{treatment}")
        assert entry["statistic"]["b"] == b
        assert entry["statistic"]["c"] == c
        assert entry["statistic"]["n_discordant"] == b + c

    for treatment_a, treatment_b in itertools.combinations(TREATMENTS, 2):
        set_a = _broken_set_from_diff(_diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment_a]))
        set_b = _broken_set_from_diff(_diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment_b]))
        entry = _entry_by_id(f5["tests"], f"F5.b.{treatment_a}_{treatment_b}")
        assert entry["statistic"]["observed"] == len(set_a & set_b)
        assert entry["statistic"]["n_a"] == len(set_a)
        assert entry["statistic"]["n_b"] == len(set_b)


# ---------------------------------------------------------------------------
# 8.2 — Holm within F5 now runs over the 9 F5.a+F5.b+F5.d p-values
# ---------------------------------------------------------------------------


def test_layer8_2_holm_within_f5_is_over_nine_including_f5d():
    f5 = rs.family_f5(RESULTS_DIR)

    ids_9 = (
        [f"F5.a.{t}" for t in TREATMENTS]
        + [f"F5.b.{a}_{b}" for a, b in itertools.combinations(TREATMENTS, 2)]
        + [f"F5.d.{t}" for t in TREATMENTS]
    )
    entries_9 = [_entry_by_id(f5["tests"], entry_id) for entry_id in ids_9]
    p_raw_9 = [e["p_raw"] for e in entries_9]
    expected_holm_9 = _holm_reference(p_raw_9)
    observed_holm_9 = [e["p_holm"] for e in entries_9]
    assert observed_holm_9 == pytest.approx(expected_holm_9, abs=1e-12)

    ids_6 = ids_9[:6]  # F5.a + F5.b only, in the same order as the old layer-7 test
    entries_6 = [_entry_by_id(f5["tests"], entry_id) for entry_id in ids_6]
    p_raw_6 = [e["p_raw"] for e in entries_6]
    holm_over_6_only = _holm_reference(p_raw_6)
    observed_p_holm_for_those_6 = [e["p_holm"] for e in entries_6]

    mismatches = [
        i
        for i, (a, b) in enumerate(zip(holm_over_6_only, observed_p_holm_for_those_6))
        if abs(a - b) > 1e-9
    ]
    assert mismatches, (
        "Holm computed over just the 6 F5.a+F5.b p-values gives the same p_holm "
        "as the stored values -- but amendment 3 requires Holm over 9 "
        "(F5.a+F5.b+F5.d), so if this doesn't differ the m=9 correction isn't "
        "actually being applied"
    )

    assert len(f5["tests"]) == 9
    ids_in_tests = {e["id"] for e in f5["tests"]}
    assert "F5.c" not in ids_in_tests
    assert "p_holm" not in f5["confounder"]


# ---------------------------------------------------------------------------
# 8.3 — baseline_stratum: quantile/mask recomputation on real data, plus a
# synthetic tie-at-the-cutoff case that pins down ">=" and not ">"
# ---------------------------------------------------------------------------


def test_layer8_3_baseline_stratum_matches_recomputation_on_real_data():
    pair_ids = list(range(250))
    v1_full = _raw_field_by_pair_id("v1_v1_en.json", "pesq_nb_est")
    base = {pid: {"pesq_nb_est": v1_full[pid]} for pid in pair_ids}
    values = np.array([v1_full[pid] for pid in pair_ids], dtype=np.float64)
    cutoff = float(np.quantile(values, BASELINE_QUANTILE))
    expected_mask = values >= cutoff

    observed_mask = rs.baseline_stratum(
        base, pair_ids, quantile=BASELINE_QUANTILE, field="pesq_nb_est"
    )

    assert list(observed_mask) == list(expected_mask)
    assert int(np.sum(observed_mask)) == int(np.sum(expected_mask))


def test_layer8_3_baseline_stratum_uses_greater_equal_not_greater():
    # np.quantile(.75) of [1,2,3,4,5] with linear interpolation lands exactly
    # on the value 4.0 (position 0.75*(5-1) = 3, a whole index) -- an exact
    # tie at the cutoff, so ">=" and ">" disagree on pair_id 3.
    base = {i: {"m": float(v)} for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])}
    pair_ids = [0, 1, 2, 3, 4]
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cutoff = float(np.quantile(values, 0.75))
    assert cutoff == pytest.approx(4.0, abs=1e-12), (
        "test construction assumption broke: expected quantile 0.75 of "
        "[1,2,3,4,5] to land exactly on the value 4.0"
    )

    mask = rs.baseline_stratum(base, pair_ids, quantile=0.75, field="m")

    assert list(mask) == [False, False, False, True, True], (
        "baseline_stratum must use '>=' against the quantile cutoff, not '>' "
        "-- with this synthetic data (cutoff lands exactly on the value 4.0) "
        "the two disagree on whether pair_id 3 belongs to the stratum"
    )


# ---------------------------------------------------------------------------
# 8.4 — F5.d: stratified McNemar restricted to the top quartile of V1's
# baseline PESQ-NB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("treatment", TREATMENTS)
def test_layer8_4_family_f5d_stratified_mcnemar_matches_recomputation(treatment):
    pair_ids = list(range(250))
    v1_est = _v1_est_array(pair_ids)
    cutoff = float(np.quantile(v1_est, BASELINE_QUANTILE))
    stratum_mask = v1_est >= cutoff
    n_stratum = int(np.sum(stratum_mask))

    d_control_full = _diff_est_vs_v1_en_baseline(CONTROL_ABS_PATHS[PRIMARY_CONTROL])
    d_treatment_full = _diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[treatment])

    d_control = d_control_full[stratum_mask]
    d_treatment = d_treatment_full[stratum_mask]

    broken_control = d_control < F5_THRESHOLD
    broken_treatment = d_treatment < F5_THRESHOLD

    b = int(np.sum(broken_treatment & ~broken_control))
    c = int(np.sum(~broken_treatment & broken_control))
    n_discordant = b + c
    expected_p = (
        1.0
        if n_discordant == 0
        else float(stats.binomtest(b, n_discordant, 0.5, alternative="two-sided").pvalue)
    )
    prop_treatment = float(np.mean(broken_treatment))
    prop_control = float(np.mean(broken_control))

    f5 = rs.family_f5(RESULTS_DIR)
    entry = _entry_by_id(f5["tests"], f"F5.d.{treatment}")

    assert entry["test"] == "mcnemar_stratified"
    assert entry["statistic"]["n_stratum"] == n_stratum
    assert entry["statistic"]["baseline_cutoff"] == pytest.approx(cutoff, abs=1e-9)
    assert entry["statistic"]["b"] == b
    assert entry["statistic"]["c"] == c
    assert entry["statistic"]["n_discordant"] == n_discordant
    assert entry["statistic"]["prop_treatment"] == pytest.approx(prop_treatment, abs=1e-12)
    assert entry["statistic"]["prop_control"] == pytest.approx(prop_control, abs=1e-12)
    assert entry["p_raw"] == pytest.approx(expected_p, rel=1e-9)


# ---------------------------------------------------------------------------
# 8.5 — F5.c redefined as a measured confounder: primary group (union of
# broken across the three treatments vs. never broken) plus the two
# mandatory sensitivity definitions
# ---------------------------------------------------------------------------


def test_layer8_5_family_f5c_confounder_primary_matches_recomputation():
    broken_sets = {
        t: _broken_set_from_diff(_diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[t]))
        for t in TREATMENTS
    }
    union_broken = broken_sets["v3"] | broken_sets["v3b"] | broken_sets["v3e"]
    all_ids = set(range(250))
    never_broken = sorted(all_ids - union_broken)
    union_ids = sorted(union_broken)

    assert len(union_ids) > 0
    assert len(never_broken) > 0

    v1_est = _raw_field_by_pair_id("v1_v1_en.json", "pesq_nb_est")
    group_a = np.array([v1_est[i] for i in union_ids], dtype=np.float64)
    group_b = np.array([v1_est[i] for i in never_broken], dtype=np.float64)

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    mean_diff = mean_a - mean_b
    _, p_raw = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    rng = np.random.default_rng(rs.SEED)
    boot = stats.bootstrap(
        (group_a, group_b),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        method="BCa",
        n_resamples=rs.N_RESAMPLES,
        confidence_level=0.95,
        random_state=rng,
    )

    f5 = rs.family_f5(RESULTS_DIR)
    confounder = f5["confounder"]
    primary = confounder["primary"]

    assert confounder["id"] == "F5.c"
    assert confounder["test"] == "mannwhitney+bootstrap_ci"
    assert confounder["corrected"] is False
    assert "negative_control" not in f5

    assert primary["group"] == "union_broken_vs_never_broken"
    assert primary["n_a"] == len(union_ids)
    assert primary["n_b"] == len(never_broken)
    assert primary["mean_a"] == pytest.approx(mean_a, abs=1e-12)
    assert primary["mean_b"] == pytest.approx(mean_b, abs=1e-12)
    assert primary["mean_diff"] == pytest.approx(mean_diff, abs=1e-12)
    assert primary["p_raw"] == pytest.approx(float(p_raw), rel=1e-9)
    assert primary["ci95"]["lo"] == pytest.approx(boot.confidence_interval.low, abs=1e-6)
    assert primary["ci95"]["hi"] == pytest.approx(boot.confidence_interval.high, abs=1e-6)


def test_layer8_5_family_f5c_confounder_sensitivity_matches_recomputation():
    broken_sets = {
        t: _broken_set_from_diff(_diff_est_vs_v1_en_baseline(TREATMENT_ABS_PATHS[t]))
        for t in TREATMENTS
    }
    triple = broken_sets["v3"] & broken_sets["v3b"] & broken_sets["v3e"]
    union_broken = broken_sets["v3"] | broken_sets["v3b"] | broken_sets["v3e"]
    all_ids = set(range(250))
    never_broken = all_ids - union_broken
    rest = all_ids - triple

    assert len(triple) > 0, "triple-intersection set is empty; Mann-Whitney is undefined"

    v1_est = _raw_field_by_pair_id("v1_v1_en.json", "pesq_nb_est")

    def _group(ids):
        return np.array([v1_est[i] for i in sorted(ids)], dtype=np.float64)

    expected_groups = {
        "triple_vs_rest": (triple, rest),
        "triple_vs_never_broken": (triple, never_broken),
    }

    f5 = rs.family_f5(RESULTS_DIR)
    sensitivity_by_group = {e["group"]: e for e in f5["confounder"]["sensitivity"]}
    assert set(sensitivity_by_group) == set(expected_groups)

    for group_name, (ids_a, ids_b) in expected_groups.items():
        group_a = _group(ids_a)
        group_b = _group(ids_b)
        mean_diff = float(np.mean(group_a) - np.mean(group_b))
        entry = sensitivity_by_group[group_name]
        assert entry["n_a"] == len(ids_a)
        assert entry["n_b"] == len(ids_b)
        assert entry["mean_a"] == pytest.approx(float(np.mean(group_a)), abs=1e-12)
        assert entry["mean_b"] == pytest.approx(float(np.mean(group_b)), abs=1e-12)
        assert entry["mean_diff"] == pytest.approx(mean_diff, abs=1e-12)


# ---------------------------------------------------------------------------
# 8.6 — stratified_profile: 28 entries (7 series x 4 quartiles), prop_broken
# recomputed for all of them
# ---------------------------------------------------------------------------


def test_layer8_6_stratified_profile_matches_recomputation():
    pair_ids = list(range(250))
    v1_est = _v1_est_array(pair_ids)
    quartiles = _quartile_labels(v1_est)

    f5 = rs.family_f5(RESULTS_DIR)
    profile_by_key = {(e["series"], e["quartile"]): e for e in f5["stratified_profile"]}
    assert len(f5["stratified_profile"]) == 28

    for series, path in {**TREATMENT_ABS_PATHS, **CONTROL_ABS_PATHS}.items():
        d = _diff_est_vs_v1_en_baseline(path)
        broken = d < F5_THRESHOLD
        for q in (1, 2, 3, 4):
            mask = quartiles == q
            n = int(np.sum(mask))
            prop = float(np.mean(broken[mask]))
            entry = profile_by_key[(series, q)]
            assert entry["n"] == n, f"{series} quartile {q}"
            assert entry["prop_broken"] == pytest.approx(prop, abs=1e-12), (
                f"{series} quartile {q}"
            )


# ---------------------------------------------------------------------------
# 8.7 — full F5 schema per contract 4.5-quater: 9 tests, confounder (not
# negative_control), stratified_profile, no numpy types
# ---------------------------------------------------------------------------


def test_layer8_7_family_f5_schema_matches_amendment_3(tmp_path):
    payload = rs.run_all(
        results_dir=RESULTS_DIR,
        metadata_dir=METADATA_DIR,
        out_json=tmp_path / "reanalysis_stats.json",
        out_md=tmp_path / "reanalisis_estadistico.md",
        seed=42,
    )
    f5 = payload["families"]["F5"]

    assert f5["correction"] == "holm"
    assert f5["exploratory"] is False
    assert f5["primary_control"] == PRIMARY_CONTROL
    assert f5["threshold"] == pytest.approx(F5_THRESHOLD)
    assert "negative_control" not in f5
    assert "confounder" in f5
    assert "stratified_profile" in f5

    assert len(f5["tests"]) == 9
    counts_by_prefix = {"F5.a": 0, "F5.b": 0, "F5.d": 0}
    for entry in f5["tests"]:
        required_keys = {"id", "label", "test", "statistic", "p_raw", "p_holm"}
        assert required_keys <= entry.keys()
        if entry["id"].startswith("F5.a"):
            counts_by_prefix["F5.a"] += 1
            assert entry["test"] == "mcnemar"
            assert {"b", "c", "n_discordant", "prop_treatment", "prop_control"} <= entry[
                "statistic"
            ].keys()
        elif entry["id"].startswith("F5.b"):
            counts_by_prefix["F5.b"] += 1
            assert entry["test"] == "hypergeometric"
            assert {"observed", "expected", "n_a", "n_b"} <= entry["statistic"].keys()
        elif entry["id"].startswith("F5.d"):
            counts_by_prefix["F5.d"] += 1
            assert entry["test"] == "mcnemar_stratified"
            assert {
                "b",
                "c",
                "n_discordant",
                "prop_treatment",
                "prop_control",
                "n_stratum",
                "baseline_cutoff",
            } <= entry["statistic"].keys()
        else:
            pytest.fail(f"unexpected F5 test id: {entry['id']!r}")
    assert counts_by_prefix == {"F5.a": 3, "F5.b": 3, "F5.d": 3}

    confounder = f5["confounder"]
    required_confounder_keys = {"id", "test", "corrected", "primary", "sensitivity"}
    assert required_confounder_keys <= confounder.keys()
    assert confounder["id"] == "F5.c"
    assert confounder["test"] == "mannwhitney+bootstrap_ci"
    assert confounder["corrected"] is False
    assert "p_holm" not in confounder

    primary = confounder["primary"]
    required_primary_keys = {
        "group",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "mean_diff",
        "ci95",
        "p_raw",
    }
    assert required_primary_keys <= primary.keys()
    assert primary["group"] == "union_broken_vs_never_broken"
    assert {"lo", "hi", "method"} <= primary["ci95"].keys()
    assert primary["ci95"]["method"] == "BCa"

    sensitivity_groups = {e["group"] for e in confounder["sensitivity"]}
    assert sensitivity_groups == {"triple_vs_rest", "triple_vs_never_broken"}
    for entry in confounder["sensitivity"]:
        assert required_primary_keys <= entry.keys()

    profile = f5["stratified_profile"]
    assert len(profile) == 28
    expected_series_roles = {t: "treatment" for t in TREATMENTS}
    expected_series_roles.update({c: "control" for c in CONTROL_ABS_PATHS})
    seen = set()
    for entry in profile:
        required_keys = {"series", "role", "quartile", "n", "prop_broken"}
        assert required_keys <= entry.keys()
        assert entry["role"] == expected_series_roles[entry["series"]]
        assert entry["quartile"] in (1, 2, 3, 4)
        key = (entry["series"], entry["quartile"])
        assert key not in seen
        seen.add(key)
    assert seen == {(s, q) for s in expected_series_roles for q in (1, 2, 3, 4)}

    json.dumps(f5)
    _assert_no_numpy_types(f5)
