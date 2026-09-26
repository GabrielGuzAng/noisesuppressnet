"""asr/prepare_smoke.py

Builds the clean-speech smoke set that characterizes the ASR instrument before
any sealed material exists: a handful of utterances per corpus, decoded to 16 kHz
mono float32 WAV with the project's own loaders, plus their reference text.

RUNS IN THE PROJECT VENV (`.venv`): it reuses `load_resample_mono` from
`datasets.make_mixtures` (Common Voice) and `decode_mls_audio` from
`scripts.seal_test_set_mls_es` (MLS), so the audio reaching the ASR goes through
the exact decoders the sealed sets used. The ASR venv only ever reads 16 kHz WAV.

Material, disjoint from every sealed set by construction and verified here:

- Common Voice ES `dev` (`data/interim/cv26_es/dev_manifest.tsv`). `test_v2_es`
  was drawn from the `test` split only.
- MLS Spanish `dev`+`test` minus the 250 `speech_source` ids of
  `test_v3_mls_es`. The `train` split is not used.

Whole utterances, never 4 s crops: a crop's reference would be the transcript of
the full utterance, which makes WER invalid by construction.

The selected ids land in the manifest. Whoever builds `test_v4_asr` should
exclude them, so the instrument's characterization set stays out of it.

Usage::

    python -m asr.prepare_smoke            # 50 per corpus, seed 42
"""
import argparse
import csv
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from datasets.make_mixtures import load_resample_mono
from scripts.seal_test_set_mls_es import MLS_DIR, MLS_PARQUETS, decode_mls_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = PROJECT_ROOT / "data" / "asr_smoke"
CV_DEV_MANIFEST = PROJECT_ROOT / "data" / "interim" / "cv26_es" / "dev_manifest.tsv"
SEAL_V2_METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json"
SEAL_V3_METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_metadata.json"

SAMPLE_RATE = 16000
SEED = 42
N_PER_CORPUS = 50

logger = logging.getLogger("asr.prepare_smoke")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_cv_dev(n: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Sample `n` Common Voice ES dev clips and verify they are not in test_v2_es.

    Returns:
        (selected rows, disjointness report).
    """
    with open(CV_DEV_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows.sort(key=lambda r: r["filepath"])

    sealed = json.loads(SEAL_V2_METADATA.read_text())["pairs"]
    sealed_clips = {Path(p["speech_file"]).name for p in sealed}
    dev_clips = {Path(r["filepath"]).name for r in rows}

    picked = rng.sample(rows, n)
    report = {
        "candidate_pool": "cv26_es dev_manifest.tsv",
        "n_candidates": len(rows),
        "sealed_reference": str(SEAL_V2_METADATA.relative_to(PROJECT_ROOT)),
        "n_sealed_clips": len(sealed_clips),
        "overlap_pool_vs_sealed": len(dev_clips & sealed_clips),
        "overlap_selected_vs_sealed": len({Path(r["filepath"]).name for r in picked}
                                          & sealed_clips),
    }
    if report["overlap_selected_vs_sealed"]:
        raise RuntimeError(f"CV dev se solapa con test_v2_es: {report}")
    return picked, report


def select_mls(n: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Sample `n` MLS dev+test utterances outside test_v3_mls_es.

    Returns:
        (selected rows with split/row/id/transcript, disjointness report).
    """
    sealed = json.loads(SEAL_V3_METADATA.read_text())["pairs"]
    # Formato: mls_es/<split>/<id>
    sealed_ids = {p["speech_source"] for p in sealed}

    pool = []
    for name in MLS_PARQUETS:
        split = name.split("-")[0]
        table = pq.read_table(MLS_DIR / name,
                              columns=["id", "transcript", "speaker_id", "audio_duration"])
        for row, (uid, txt, spk, dur) in enumerate(zip(
                table.column("id").to_pylist(), table.column("transcript").to_pylist(),
                table.column("speaker_id").to_pylist(),
                table.column("audio_duration").to_pylist())):
            pool.append({"split": split, "row": row, "source": f"mls_es/{split}/{uid}",
                         "transcript": txt, "speaker_id": str(spk),
                         "duration_s": float(dur)})
    pool.sort(key=lambda r: r["source"])

    all_ids = {r["source"] for r in pool}
    candidates = [r for r in pool if r["source"] not in sealed_ids]
    picked = rng.sample(candidates, n)
    report = {
        "candidate_pool": "MLS es dev+test parquets",
        "n_pool": len(pool),
        "sealed_reference": str(SEAL_V3_METADATA.relative_to(PROJECT_ROOT)),
        "n_sealed_ids": len(sealed_ids),
        "n_sealed_ids_found_in_pool": len(sealed_ids & all_ids),
        "n_candidates_after_exclusion": len(candidates),
        "overlap_selected_vs_sealed": len({r["source"] for r in picked} & sealed_ids),
    }
    # Si los ids del sellado no aparecen en el pool, la diferencia de conjuntos
    # no excluyó nada y el formato de id está mal: abortar, no seguir.
    if report["n_sealed_ids_found_in_pool"] != len(sealed_ids):
        raise RuntimeError(f"Ids del sellado ausentes del pool MLS: {report}")
    if report["overlap_selected_vs_sealed"]:
        raise RuntimeError(f"MLS se solapa con test_v3_mls_es: {report}")
    return picked, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean smoke set for the ASR instrument")
    parser.add_argument("--n", type=int, default=N_PER_CORPUS, help="Utterances per corpus")
    parser.add_argument("--out", type=str, default=str(SMOKE_ROOT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    out_root = Path(args.out)
    wav_dir = out_root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Un generador por corpus: agregar o sacar un corpus no mueve la muestra del otro.
    cv_rows, cv_report = select_cv_dev(args.n, random.Random(SEED))
    mls_rows, mls_report = select_mls(args.n, random.Random(SEED))
    logger.info("Disjunción CV: %s", cv_report)
    logger.info("Disjunción MLS: %s", mls_report)

    utterances = []
    for r in cv_rows:
        uid = f"cv_dev/{Path(r['filepath']).stem}"
        wav = load_resample_mono(PROJECT_ROOT / r["filepath"])
        utterances.append((uid, "cv_es_dev", r["filepath"], r["sentence"], wav))
    for r in mls_rows:
        wav = decode_mls_audio(r["split"], r["row"])
        utterances.append((r["source"], "mls_es_devtest", r["source"], r["transcript"], wav))

    entries = []
    for uid, corpus, source, reference, wav in utterances:
        path = wav_dir / f"{uid.replace('/', '__')}.wav"
        sf.write(str(path), wav.numpy(), SAMPLE_RATE, subtype="FLOAT")
        entries.append({
            "utt_id": uid,
            "corpus": corpus,
            "source": source,
            "wav": str(path.relative_to(PROJECT_ROOT)),
            "wav_sha256": sha256_file(path),
            "duration_s": wav.numel() / SAMPLE_RATE,
            "reference": reference,
        })
    logger.info("%d utterances escritas en %s", len(entries), wav_dir)

    manifest = {
        "purpose": "Caracterización del instrumento ASR sobre voz limpia. No es un "
                   "resultado ni material sellado. Excluir estos ids de test_v4_asr.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_per_corpus": args.n,
        "sample_rate": SAMPLE_RATE,
        "audio_format": "WAV float32 mono, utterance completa (sin recorte)",
        "disjointness": {"cv_es_dev": cv_report, "mls_es_devtest": mls_report},
        "utterances": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Manifiesto: %s", out_root / "manifest.json")


if __name__ == "__main__":
    main()
