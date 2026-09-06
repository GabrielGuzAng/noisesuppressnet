"""Ad hoc statistical checks for the high-SNR Spanish bucket, for docs/donde_esta_el_hueco.html.

Recomputes, deterministically and only from files under ``results/`` and
``seal_test_metadata/``, the two figures the document cites for the [15, 20] dB
bucket of ``v1_v2_es`` (bucket_idx == 4):

1. Looping robustness check: whether excluding the pairs whose speech clip is
   shorter than the mixture duration (``speech_offset == 0`` in the sealed test
   metadata, so ``pad_or_crop`` looped the clip instead of cropping it) makes the
   bucket's mean PESQ-NB delta less negative. It does not.
2. Sign-test statistics for that same bucket: n_pos, n_neg, the two-sided
   binomial p-value, the median, and the skew, to show the sign test lacks
   power at n=50 even though the bootstrap CI on the mean (reported separately
   in results/reanalysis_stats.json, family F4) excludes zero.

This module does not touch results/reanalysis_stats.json or any other family
of the main reanalysis (analysis/reanalysis_stats.py): it only answers the two
narrow follow-up questions above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "v1_v2_es.json"
METADATA_PATH = PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json"
BUCKET_IDX = 4  # SNR bucket [15, 20] dB


def load_bucket_deltas(
    results_path: Path, metadata_path: Path, bucket_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load PESQ-NB deltas for one bucket of v1_v2_es, with their speech_offset.

    Args:
        results_path: Path to results/v1_v2_es.json.
        metadata_path: Path to seal_test_metadata/test_v2_metadata.json.
        bucket_idx: SNR bucket index to select.

    Returns:
        Tuple of (pesq_nb_delta array, speech_offset array), same order,
        one entry per pair in the bucket.
    """
    with results_path.open(encoding="utf-8") as f:
        results = json.load(f)
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    offset_by_id = {pair["id"]: pair["speech_offset"] for pair in metadata["pairs"]}
    bucket_pairs = [p for p in results["all_pairs"] if p["bucket_idx"] == bucket_idx]

    deltas = np.array([p["pesq_nb_delta"] for p in bucket_pairs], dtype=np.float64)
    offsets = np.array(
        [offset_by_id[p["pair_id"]] for p in bucket_pairs], dtype=np.int64
    )
    return deltas, offsets


def looping_robustness_check(deltas: np.ndarray, offsets: np.ndarray) -> dict:
    """Compare the bucket mean with and without looped (speech_offset == 0) pairs.

    Returns:
        Dict with pair counts and means for the full bucket, the looped
        subset, and the non-looped subset.
    """
    looped = offsets == 0
    return {
        "n_total": int(deltas.size),
        "n_looped": int(np.sum(looped)),
        "n_non_looped": int(np.sum(~looped)),
        "mean_all": float(np.mean(deltas)),
        "mean_looped": float(np.mean(deltas[looped])) if np.any(looped) else None,
        "mean_non_looped": float(np.mean(deltas[~looped])),
    }


def sign_test_stats(deltas: np.ndarray) -> dict:
    """Two-sided sign test, median, mean and skew over a delta vector.

    Returns:
        Dict with 'n_pos', 'n_neg', 'n_zero', 'p_value', 'median', 'mean', 'skew'.
    """
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    result = stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided")
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "p_value": float(result.pvalue),
        "median": float(np.median(deltas)),
        "mean": float(np.mean(deltas)),
        "skew": float(stats.skew(deltas)),
    }


def main() -> None:
    deltas, offsets = load_bucket_deltas(RESULTS_PATH, METADATA_PATH, BUCKET_IDX)

    looping = looping_robustness_check(deltas, offsets)
    print("Looping robustness check — v1_v2_es, bucket_idx =", BUCKET_IDX)
    print(f"  n_total={looping['n_total']} n_looped={looping['n_looped']} "
          f"n_non_looped={looping['n_non_looped']}")
    print(f"  mean_all={looping['mean_all']:.4f}")
    print(f"  mean_looped={looping['mean_looped']:.4f}")
    print(f"  mean_non_looped={looping['mean_non_looped']:.4f}")

    sign = sign_test_stats(deltas)
    print()
    print("Sign-test power check — v1_v2_es, bucket_idx =", BUCKET_IDX)
    print(f"  n_pos={sign['n_pos']} n_neg={sign['n_neg']} n_zero={sign['n_zero']}")
    print(f"  sign test p_value={sign['p_value']:.4f}")
    print(f"  median={sign['median']:.4f} mean={sign['mean']:.4f} skew={sign['skew']:.4f}")


if __name__ == "__main__":
    main()
