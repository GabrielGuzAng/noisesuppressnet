"""
scripts/seal_test_set_mls_es.py

Sella el test set v3 en español sobre canal de audiolibro (Multilingual
LibriSpeech), como control del confound idioma/canal que arrastran las
comparaciones entre test_v1_en (LibriSpeech, audiolibro) y test_v2_es
(Common Voice, crowdsourced).

Diseño, fijado en el preregistro antes de que existiera el dato:

1. VOZ: MLS español, splits `dev` + `test` combinados (40 hablantes disjuntos,
   verificado). No se usa `train`: aportaría 36 hablantes más a costa de 14,5 GB
   y no cambia el orden de magnitud del desbalance contra los ~248 de
   Common Voice. Selección balanceada por hablante (round-robin).

2. RUIDO: **las condiciones de ruido se reusan verbatim de test_v1_metadata.json**
   — mismo noise_file, mismo noise_offset y mismo snr_db, par por par. Es la
   diferencia central respecto de seal_test_set_en.py y seal_test_set_es.py, que
   sortean el ruido. Al fijarlo, la única variable que cambia entre test_v1_en y
   este set es la fuente de voz, lo que convierte la comparación de no apareada a
   apareada por condición de ruido. Sigue el diseño de Wang et al. 2022
   (Interspeech, TAT), que mantiene fijo todo salvo la variable de interés.

3. Los buckets de SNR quedan determinados por el reuso: son los mismos de
   test_v1_en, con la misma cantidad de pares por bucket.

La metadata incluye `speaker_id` y `transcript` por par, que los otros dos
sellados no tienen: el primero es necesario para la inferencia agrupada por
hablante (los pares no son independientes, ~6 por hablante) y el segundo para la
evaluación downstream con ASR sin tener que hacer un join posterior.

Uso:
    python -m scripts.seal_test_set_mls_es

Requiere haber descargado antes los parquet de MLS español (ver docs).
"""
import hashlib
import io
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

# ── Configuración (NO cambiar después del primer sellado) ──
SEED = 42
SAMPLE_RATE = 16000
TARGET_DURATION = 4.0
N_PAIRS_TOTAL = 250

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLS_DIR = PROJECT_ROOT / "data" / "raw" / "clean_speech" / "es" / "mls"
MLS_PARQUETS = ("dev-00000-of-00001.parquet", "test-00000-of-00001.parquet")

# Fuente de las condiciones de ruido: se reusan, no se sortean.
REFERENCE_METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"

TEST_OUT_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v3_mls_es"
HASH_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_hash.txt"
METADATA_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_metadata.json"


def load_reference_conditions() -> list[dict]:
    """Load the noise conditions of test_v1_en, to be reused verbatim.

    Returns:
        The `pairs` list of test_v1_metadata.json, sorted by id.

    Raises:
        FileNotFoundError: if the reference metadata is missing.
        ValueError: if it does not hold exactly N_PAIRS_TOTAL pairs.
    """
    if not REFERENCE_METADATA.exists():
        raise FileNotFoundError(
            f"No existe {REFERENCE_METADATA}. Es la fuente de las condiciones de "
            f"ruido que este sellado reusa; sin ella el diseño apareado no se puede armar."
        )
    with open(REFERENCE_METADATA) as f:
        pairs = json.load(f)["pairs"]
    if len(pairs) != N_PAIRS_TOTAL:
        raise ValueError(
            f"{REFERENCE_METADATA} tiene {len(pairs)} pares, se esperaban {N_PAIRS_TOTAL}"
        )
    return sorted(pairs, key=lambda p: p["id"])


def load_mls_index() -> list[dict]:
    """Index MLS Spanish dev+test, keeping only clips long enough to crop.

    Returns:
        One dict per usable utterance with split, row index, speaker_id,
        transcript, mls_id and duration. Sorted for reproducibility.
    """
    target_s = TARGET_DURATION
    index: list[dict] = []
    for name in MLS_PARQUETS:
        path = MLS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Descargá los parquet de MLS español antes de sellar."
            )
        split = name.split("-")[0]
        table = pq.read_table(
            path, columns=["speaker_id", "transcript", "audio_duration", "id"]
        )
        speakers = table.column("speaker_id").to_pylist()
        transcripts = table.column("transcript").to_pylist()
        durations = table.column("audio_duration").to_pylist()
        ids = table.column("id").to_pylist()
        for row, (spk, txt, dur, uid) in enumerate(
            zip(speakers, transcripts, durations, ids)
        ):
            if float(dur) >= target_s:
                index.append({
                    "split": split,
                    "row": row,
                    "speaker_id": str(spk),
                    "transcript": txt,
                    "mls_id": str(uid),
                    "duration_s": float(dur),
                })
    # Orden estable: el parquet no garantiza orden entre corridas de pyarrow.
    index.sort(key=lambda r: (r["split"], r["speaker_id"], r["mls_id"]))
    return index


def select_speaker_balanced(index: list[dict], n_pairs: int,
                            rng: random.Random) -> list[dict]:
    """Pick `n_pairs` utterances spread as evenly as possible over speakers.

    Round-robin over speakers, drawing without replacement inside each speaker,
    so no utterance repeats and the per-speaker counts differ by at most one.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for rec in index:
        by_speaker[rec["speaker_id"]].append(rec)

    speakers = sorted(by_speaker)
    for spk in speakers:
        rng.shuffle(by_speaker[spk])

    selected: list[dict] = []
    round_idx = 0
    while len(selected) < n_pairs:
        progressed = False
        for spk in speakers:
            if len(selected) >= n_pairs:
                break
            pool = by_speaker[spk]
            if round_idx < len(pool):
                selected.append(pool[round_idx])
                progressed = True
        if not progressed:
            raise ValueError(
                f"Utterances insuficientes: {len(selected)} de {n_pairs} pedidos"
            )
        round_idx += 1
    return selected


_AUDIO_CACHE: dict[str, object] = {}


def decode_mls_audio(split: str, row: int) -> torch.Tensor:
    """Decode one MLS utterance to a mono float32 tensor at SAMPLE_RATE.

    The audio column of each split is read once and cached: re-reading the
    157 MB parquet per call would make sealing quadratic.
    """
    if split not in _AUDIO_CACHE:
        path = MLS_DIR / f"{split}-00000-of-00001.parquet"
        _AUDIO_CACHE[split] = pq.read_table(path, columns=["audio"]).column("audio")
    blob = _AUDIO_CACHE[split][row].as_py()
    raw = blob["bytes"] if isinstance(blob, dict) else blob
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"MLS {split} fila {row} vino a {sr} Hz; se esperaba {SAMPLE_RATE}. "
            f"Si el corpus cambió, hay que resamplear explícitamente antes de sellar."
        )
    return torch.from_numpy(np.ascontiguousarray(wav))


def load_noise(path: Path) -> torch.Tensor:
    """Load a noise file as mono float32 at SAMPLE_RATE."""
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def crop_at(x: torch.Tensor, target_len: int, offset: int) -> torch.Tensor:
    """Take `target_len` samples starting at `offset`, looping if too short."""
    if x.numel() < target_len:
        repeats = target_len // x.numel() + 1
        x = x.repeat(repeats)
    start = offset if offset + target_len <= x.numel() else 0
    return x[start:start + target_len]


def crop_random(x: torch.Tensor, target_len: int,
                rng: random.Random) -> tuple[torch.Tensor, int]:
    """Take a random `target_len` window; returns the crop and its offset."""
    if x.numel() < target_len:
        repeats = target_len // x.numel() + 1
        return x.repeat(repeats)[:target_len], 0
    start = rng.randint(0, x.numel() - target_len)
    return x[start:start + target_len], start


def mix_at_snr(speech: torch.Tensor, noise: torch.Tensor,
               snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix speech and noise at the given SNR; returns (mixture, clean)."""
    def rms(x):
        return torch.sqrt(torch.mean(x ** 2) + 1e-9)
    target_noise_rms = rms(speech) / (10 ** (snr_db / 20))
    return speech + noise * (target_noise_rms / rms(noise)), speech


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


def seal_test_set() -> None:
    print("=" * 70)
    print("SELLADO DE TEST SET v3 (ES, canal audiolibro — MLS)")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Voz: MLS español, splits {' + '.join(p.split('-')[0] for p in MLS_PARQUETS)}")
    print(f"Ruido: condiciones reusadas verbatim de {REFERENCE_METADATA.name}")
    print(f"Total pares: {N_PAIRS_TOTAL}")
    print(f"Salida: {TEST_OUT_DIR}")
    print()

    if TEST_OUT_DIR.exists() and any(TEST_OUT_DIR.iterdir()):
        raise RuntimeError(
            f"ABORT: {TEST_OUT_DIR} ya existe con contenido.\n"
            f"El test set sellado NO debe regenerarse.\n"
            f"Si querés forzar la regeneración, borralo manualmente PRIMERO y "
            f"confirmá que no rompés la reproducibilidad de resultados previos."
        )

    reference = load_reference_conditions()
    index = load_mls_index()

    n_speakers = len({r["speaker_id"] for r in index})
    print(f"Utterances MLS usables (>= {TARGET_DURATION}s): {len(index)}")
    print(f"Hablantes disponibles: {n_speakers}")
    if n_speakers < 20:
        raise ValueError(f"Solo {n_speakers} hablantes; se esperaban al menos 20")

    speech_rng = random.Random(SEED)
    offset_rng = random.Random(SEED + 2)

    selected = select_speaker_balanced(index, N_PAIRS_TOTAL, speech_rng)
    per_speaker = defaultdict(int)
    for rec in selected:
        per_speaker[rec["speaker_id"]] += 1
    counts = sorted(per_speaker.values())
    print(f"Balance por hablante: {len(per_speaker)} hablantes, "
          f"min {counts[0]} / max {counts[-1]} pares cada uno")
    print()

    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_len = int(TARGET_DURATION * SAMPLE_RATE)

    metadata = {
        "seed": SEED,
        "language": "es",
        "channel": "audiobook (LibriVox) — control de canal contra test_v1_en",
        "source": "Multilingual LibriSpeech español, splits dev + test",
        "noise_conditions_reused_from": str(REFERENCE_METADATA.relative_to(PROJECT_ROOT)),
        "n_pairs_total": N_PAIRS_TOTAL,
        "n_speakers": len(per_speaker),
        "snr_buckets": None,  # se completa desde la referencia
        "sample_rate": SAMPLE_RATE,
        "target_duration_s": TARGET_DURATION,
        "generated_at": datetime.now().isoformat(),
        "pairs": [],
    }

    buckets = sorted({p["bucket_idx"] for p in reference})
    for pair_idx, (ref, speech_rec) in enumerate(zip(reference, selected)):
        if ref["id"] != pair_idx:
            raise ValueError(f"Referencia desalineada en el par {pair_idx}: id={ref['id']}")

        noise_path = PROJECT_ROOT / ref["noise_file"]
        if not noise_path.exists():
            raise FileNotFoundError(f"Falta el ruido del par {pair_idx}: {noise_path}")

        sp_full = decode_mls_audio(speech_rec["split"], speech_rec["row"])
        sp, sp_offset = crop_random(sp_full, target_len, offset_rng)

        ns = crop_at(load_noise(noise_path), target_len, ref["noise_offset"])

        # RMS normalization (etapa 1 del paper Tan & Wang), igual que los otros sellados
        c = torch.sqrt(
            torch.tensor(target_len, dtype=torch.float32) / (sp.pow(2).sum() + 1e-8)
        )
        sp = sp * c

        mixture, clean = mix_at_snr(sp, ns, ref["snr_db"])

        peak = max(mixture.abs().max(), clean.abs().max(), torch.tensor(1e-8))
        scale = 0.9 / peak
        mixture, clean = mixture * scale, clean * scale

        pair_dir = TEST_OUT_DIR / f"pair_{pair_idx:04d}"
        pair_dir.mkdir(exist_ok=True)
        torchaudio.save(str(pair_dir / "noisy.wav"), mixture.unsqueeze(0), SAMPLE_RATE)
        torchaudio.save(str(pair_dir / "clean.wav"), clean.unsqueeze(0), SAMPLE_RATE)

        metadata["pairs"].append({
            "id": pair_idx,
            "bucket_idx": ref["bucket_idx"],
            "snr_db": ref["snr_db"],
            "speech_source": f"mls_es/{speech_rec['split']}/{speech_rec['mls_id']}",
            "speaker_id": speech_rec["speaker_id"],
            "transcript": speech_rec["transcript"],
            "noise_file": ref["noise_file"],
            "noise_category": ref["noise_category"],
            "speech_offset": sp_offset,
            "noise_offset": ref["noise_offset"],
        })

        if (pair_idx + 1) % 50 == 0:
            print(f"  {pair_idx + 1}/{N_PAIRS_TOTAL}")

    metadata["snr_buckets"] = [
        [min(p["snr_db"] for p in reference if p["bucket_idx"] == b),
         max(p["snr_db"] for p in reference if p["bucket_idx"] == b)]
        for b in buckets
    ]

    pair_dirs = sorted(TEST_OUT_DIR.glob("pair_*"))
    dataset_hash, n_files = compute_dataset_hash(pair_dirs)

    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(
        "# Test Set v3 ES (canal audiolibro, MLS) - Hash de integridad\n"
        f"# Generado: {metadata['generated_at']}\n"
        f"# Seed: {SEED}\n"
        f"# Total pares: {N_PAIRS_TOTAL}\n"
        f"# Total archivos WAV: {n_files}\n"
        f"# Hablantes: {metadata['n_speakers']}\n"
        f"# Condiciones de ruido reusadas de: {metadata['noise_conditions_reused_from']}\n"
        "# Hash SHA-256 acumulativo:\n"
        f"{dataset_hash}\n"
    )
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"Pares sellados: {len(pair_dirs)}")
    print(f"Archivos wav:   {n_files}")
    print(f"Hash SHA-256:   {dataset_hash}")
    print(f"Hash escrito:   {HASH_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Metadata:       {METADATA_FILE.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    seal_test_set()
