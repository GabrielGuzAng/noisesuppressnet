"""
scripts/seal_test_set_asr.py

Sella test_v4_asr: 500 pares (250 de Common Voice + 250 de MLS) para medir WER
downstream. Es el gemelo SIN RECORTAR de test_v2_es y test_v3_mls_es.

Diseño:

1. VOZ: la utterance COMPLETA de cada par de los dos sellados de referencia, sin
   el recorte de 4,0 s. Ese recorte invalida el WER sobre los sellados existentes:
   la referencia es la transcripción de la utterance entera y el audio es un
   fragmento en un offset aleatorio. Estrato CV = pares 0..249 de
   test_v2_metadata.json (249 clips únicos: common_voice_es_24398099.mp3 aparece en
   los pares 87 y 239 con condiciones de ruido distintas, y se conserva así para
   que el gemelo sea par por par). Estrato MLS = pares 0..249 de
   test_v3_mls_es_metadata.json.

2. RECORTE DE SILENCIO de cabeza y cola sobre la FUENTE, antes de normalizar y
   mezclar: umbral TRIM_TOP_DB relativo al máximo del clip, TRIM_MARGIN_S de margen
   a cada lado, idéntico en los dos estratos. Sin esto, el silencio de cola induce
   inserciones alucinadas ("gracias") del ASR que el realce modifica, y eso se
   confunde con el efecto a medir. Como se aplica a la fuente, clean y noisy lo
   heredan igual. El recorte aplicado queda en la metadata de cada par.

3. RUIDO: noise_file, noise_offset y snr_db se heredan verbatim del par de
   referencia; no se sortea nada. El ruido se lee circularmente desde noise_offset
   durante toda la utterance, así que los primeros 4 s de ruido coinciden con los
   del sellado de referencia y lo que excede se toma a continuación (con vuelta al
   inicio del archivo si se termina).

4. Normalización RMS de la voz y peak a 0,9 igual que los otros sellados, pero
   sobre la longitud real de cada utterance recortada.

La metadata incluye `transcript` (referencia para el WER) y `speaker_id` por par.

Uso:
    # Dry run (no toca data/test_sealed ni seal_test_metadata):
    python -m scripts.seal_test_set_asr --dry-run --out /tmp/algo
    # Sellado real (una sola vez):
    python -m scripts.seal_test_set_asr
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torchaudio

from scripts.seal_test_set_es import load_resample_mono
from scripts.seal_test_set_mls_es import MLS_DIR, MLS_PARQUETS, decode_mls_audio

# ── Configuración (NO cambiar después del primer sellado) ──
SAMPLE_RATE = 16000
N_PAIRS_PER_STRATUM = 250
TRIM_TOP_DB = 40.0          # dB por debajo del máximo del clip
TRIM_FRAME = 400            # 25 ms
TRIM_HOP = 160              # 10 ms
TRIM_MARGIN_S = 0.100       # margen conservado a cada lado

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CV_REFERENCE = PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json"
MLS_REFERENCE = PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_metadata.json"
CV_TEST_MANIFEST = PROJECT_ROOT / "data" / "interim" / "cv26_es" / "test_manifest.tsv"

TEST_OUT_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v4_asr"
HASH_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_v4_asr_hash.txt"
METADATA_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_v4_asr_metadata.json"


def load_reference(path: Path) -> list[dict]:
    """Load the pairs of a sealed reference set, sorted and checked by id."""
    with open(path) as f:
        pairs = sorted(json.load(f)["pairs"], key=lambda p: p["id"])
    if [p["id"] for p in pairs] != list(range(N_PAIRS_PER_STRATUM)):
        raise ValueError(f"{path} no tiene los ids 0..{N_PAIRS_PER_STRATUM - 1}")
    return pairs


def load_cv_manifest() -> dict[str, dict]:
    """Map CV filepath -> {transcript, speaker_id} from the official test split."""
    manifest = pd.read_csv(CV_TEST_MANIFEST, sep="\t")
    return {
        row.filepath: {"transcript": row.sentence, "speaker_id": row.client_id}
        for row in manifest.itertuples()
    }


def load_mls_locations() -> dict[str, tuple[str, int]]:
    """Map MLS utterance id -> (split, row) over the dev+test parquets."""
    locations = {}
    for name in MLS_PARQUETS:
        split = name.split("-")[0]
        ids = pq.read_table(MLS_DIR / name, columns=["id"]).column("id").to_pylist()
        for row, uid in enumerate(ids):
            locations[str(uid)] = (split, row)
    return locations


def trim_silence(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Trim leading/trailing silence, keeping TRIM_MARGIN_S on each side.

    Frames are 25 ms / 10 ms hop; a frame is silent when its RMS is more than
    TRIM_TOP_DB below the loudest frame RMS of the clip (librosa.effects.trim).

    Returns:
        The trimmed signal and the [start, end) sample indices kept.
    """
    _, (start, end) = librosa.effects.trim(
        x.numpy(), top_db=TRIM_TOP_DB, ref=np.max,
        frame_length=TRIM_FRAME, hop_length=TRIM_HOP,
    )
    margin = int(round(TRIM_MARGIN_S * SAMPLE_RATE))
    start, end = max(0, int(start) - margin), min(x.numel(), int(end) + margin)
    return x[start:end], start, end


def noise_from_offset(noise: torch.Tensor, length: int, offset: int) -> torch.Tensor:
    """Read `length` samples circularly starting at `offset`.

    For the first 4 s this reproduces the noise segment of the reference seal
    (whose offsets were always drawn so that offset + 4 s fits, or were 0 for
    noise shorter than 4 s, which was looped).
    """
    idx = (offset + torch.arange(length)) % noise.numel()
    return noise[idx]


def build_pair(speech: torch.Tensor, ref: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Trim, normalize and mix one pair with the inherited noise condition."""
    orig_len = speech.numel()
    speech, trim_start, trim_end = trim_silence(speech)
    n = speech.numel()

    noise_full = load_resample_mono(PROJECT_ROOT / ref["noise_file"])
    noise = noise_from_offset(noise_full, n, ref["noise_offset"])

    # RMS normalization (etapa 1 de Tan & Wang), igual que los otros sellados
    speech = speech * torch.sqrt(
        torch.tensor(n, dtype=torch.float32) / (speech.pow(2).sum() + 1e-8)
    )
    rms = lambda s: torch.sqrt(torch.mean(s ** 2) + 1e-9)
    scaled_noise = noise * (rms(speech) / (10 ** (ref["snr_db"] / 20)) / rms(noise))
    mixture, clean = speech + scaled_noise, speech

    peak = max(mixture.abs().max(), clean.abs().max(), torch.tensor(1e-8))
    mixture, clean = mixture * (0.9 / peak), clean * (0.9 / peak)

    info = {
        "orig_duration_s": round(orig_len / SAMPLE_RATE, 4),
        "trim_start_sample": trim_start,
        "trim_end_sample": trim_end,
        "duration_s": round(n / SAMPLE_RATE, 4),
        "noise_wraps": ref["noise_offset"] + n > noise_full.numel(),
    }
    return mixture, clean, info


def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dataset_hash(pair_dirs) -> tuple[str, int]:
    combined = hashlib.sha256()
    files = []
    for d in sorted(pair_dirs):
        files.extend(sorted(f for f in d.iterdir() if f.suffix == ".wav"))
    for f in files:
        combined.update(f.name.encode())
        combined.update(compute_file_hash(f).encode())
    return combined.hexdigest(), len(files)


def plan_pairs() -> list[dict]:
    """Resolve the 500 pairs: speech source, transcript and inherited noise."""
    cv_meta = load_cv_manifest()
    mls_loc = load_mls_locations()
    plan = []
    for ref in load_reference(CV_REFERENCE):
        cv = cv_meta.get(ref["speech_file"])
        if cv is None:
            raise KeyError(f"{ref['speech_file']} no está en {CV_TEST_MANIFEST}")
        plan.append({"stratum": "cv", "ref": ref, "speech": ref["speech_file"], **cv})
    for ref in load_reference(MLS_REFERENCE):
        mls_id = ref["speech_source"].rsplit("/", 1)[1]
        if mls_id not in mls_loc:
            raise KeyError(f"{mls_id} no está en los parquet de MLS")
        plan.append({
            "stratum": "mls", "ref": ref, "speech": ref["speech_source"],
            "mls_loc": mls_loc[mls_id],
            "transcript": ref["transcript"], "speaker_id": ref["speaker_id"],
        })
    return plan


def load_speech(item: dict) -> torch.Tensor:
    if item["stratum"] == "cv":
        return load_resample_mono(PROJECT_ROOT / item["speech"])
    return decode_mls_audio(*item["mls_loc"])


def seal(out_dir: Path, indices: list[int], dry_run: bool) -> list[dict]:
    """Write the pairs in `indices` to `out_dir`; hash + metadata only if sealing."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"ABORT: {out_dir} ya existe con contenido.\n"
            f"El test set sellado NO debe regenerarse."
        )
    if not dry_run and (HASH_FILE.exists() or METADATA_FILE.exists()):
        raise RuntimeError(f"ABORT: ya existe {HASH_FILE.name} o {METADATA_FILE.name}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_pairs()
    records = []
    for pair_idx in indices:
        item = plan[pair_idx]
        ref = item["ref"]
        mixture, clean, info = build_pair(load_speech(item), ref)

        pair_dir = out_dir / f"pair_{pair_idx:04d}"
        pair_dir.mkdir()
        torchaudio.save(str(pair_dir / "noisy.wav"), mixture.unsqueeze(0), SAMPLE_RATE)
        torchaudio.save(str(pair_dir / "clean.wav"), clean.unsqueeze(0), SAMPLE_RATE)

        records.append({
            "id": pair_idx,
            "stratum": item["stratum"],
            "source_pair_id": ref["id"],
            "bucket_idx": ref["bucket_idx"],
            "snr_db": ref["snr_db"],
            "speech_source": item["speech"],
            "speaker_id": item["speaker_id"],
            "transcript": item["transcript"],
            "noise_file": ref["noise_file"],
            "noise_category": ref["noise_category"],
            "noise_offset": ref["noise_offset"],
            **info,
        })
        if not dry_run and (pair_idx + 1) % 50 == 0:
            print(f"  {pair_idx + 1}/{len(plan)}")

    if dry_run:
        return records

    dataset_hash, n_files = compute_dataset_hash(sorted(out_dir.glob("pair_*")))
    metadata = {
        "language": "es",
        "purpose": "WER downstream — gemelo sin recortar de test_v2_es y test_v3_mls_es",
        "strata": {
            "cv": {"ids": [0, N_PAIRS_PER_STRATUM - 1],
                   "reference": str(CV_REFERENCE.relative_to(PROJECT_ROOT))},
            "mls": {"ids": [N_PAIRS_PER_STRATUM, 2 * N_PAIRS_PER_STRATUM - 1],
                    "reference": str(MLS_REFERENCE.relative_to(PROJECT_ROOT))},
        },
        "n_pairs_total": len(records),
        "sample_rate": SAMPLE_RATE,
        "target_duration_s": None,
        "silence_trim": {
            "method": "librosa.effects.trim, ref = max frame RMS del clip",
            "top_db": TRIM_TOP_DB, "frame_length": TRIM_FRAME,
            "hop_length": TRIM_HOP, "margin_s": TRIM_MARGIN_S,
        },
        "noise_read": "circular desde noise_offset por toda la utterance",
        "generated_at": datetime.now().isoformat(),
        "pairs": records,
    }
    HASH_FILE.write_text(
        "# Test Set v4 ASR (ES, CV + MLS, utterances completas) - Hash de integridad\n"
        f"# Generado: {metadata['generated_at']}\n"
        f"# Total pares: {len(records)}\n"
        f"# Total archivos WAV: {n_files}\n"
        "# Condiciones de ruido heredadas de test_v2_metadata.json y test_v3_mls_es_metadata.json\n"
        "# Hash SHA-256 acumulativo:\n"
        f"{dataset_hash}\n"
    )
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Pares sellados: {len(records)}  |  wav: {n_files}  |  SHA-256: {dataset_hash}")
    return records


def dry_run_indices() -> list[int]:
    """First pair of each SNR bucket in each stratum (5 CV + 5 MLS)."""
    out = []
    for offset, ref_path in ((0, CV_REFERENCE), (N_PAIRS_PER_STRATUM, MLS_REFERENCE)):
        seen = set()
        for p in load_reference(ref_path):
            if p["bucket_idx"] not in seen:
                seen.add(p["bucket_idx"])
                out.append(offset + p["id"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="directorio del dry run")
    parser.add_argument("--ids", type=int, nargs="*", help="pares del dry run (0..499)")
    args = parser.parse_args()

    if not args.dry_run:
        seal(TEST_OUT_DIR, list(range(2 * N_PAIRS_PER_STRATUM)), dry_run=False)
        return

    if args.out is None or args.out.resolve().is_relative_to(PROJECT_ROOT / "data"):
        raise SystemExit("--dry-run pide --out fuera de data/")
    records = seal(args.out, args.ids or dry_run_indices(), dry_run=True)
    for r in records:
        pair_dir = args.out / f"pair_{r['id']:04d}"
        noisy, _ = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair_dir / "clean.wav"))
        snr = 10 * torch.log10(clean.pow(2).sum() / (noisy - clean).pow(2).sum())
        r["snr_achieved_db"] = round(float(snr), 3)
        r["trimmed_head_s"] = round(r["trim_start_sample"] / SAMPLE_RATE, 3)
        r["trimmed_tail_s"] = round(r["orig_duration_s"] - r["trim_end_sample"] / SAMPLE_RATE, 3)
    with open(args.out / "dry_run.json", "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    for r in records:
        print(f"{r['id']:4d} {r['stratum']:3s} b{r['bucket_idx']} "
              f"{r['orig_duration_s']:6.2f}s -> {r['duration_s']:6.2f}s "
              f"(-{r['trimmed_head_s']:.2f}/-{r['trimmed_tail_s']:.2f})  "
              f"SNR {r['snr_db']:7.3f} -> {r['snr_achieved_db']:7.3f}  | {r['transcript']}")


if __name__ == "__main__":
    main()
