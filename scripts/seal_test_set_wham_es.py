"""
scripts/seal_test_set_wham_es.py

Sella el test set `wham_es`: la misma voz de test_v3_mls_es con ruido WHAM!, que es
el único ruido disponible que NUNCA entró a ningún entrenamiento nuestro.

Por qué existe. El pool de ruido (MUSAN + ESC-50, 4016 archivos) es idéntico en
datasets/make_mixtures.py y en los tres sellados anteriores: no tiene holdout.
Con 52.000 sorteos por dataset cada archivo de ruido se vio k=12,95 veces en
entrenamiento (k=25,90 para las variantes que además entrenaron en español), o
sea cobertura esperada del 99,9998%. Ningún sellado anterior mide generalización
a ruido no visto, y eso vale para TODAS las variantes, V7 incluida.

Diseño — una sola variable respecto de test_v3_mls_es:

1. VOZ: **idéntica, verbatim**. Se reusa de test_v3_mls_es_metadata.json el
   speech_source, el speech_offset, el snr_db y el bucket_idx, par por par. Se
   vuelve a decodificar la misma utterance de MLS y se recorta en el mismo
   offset. speaker_id y transcript se arrastran.

2. RUIDO: WHAM! (Wichern et al. 2019), splits `cv` + `tt`. Es ruido ambiente de
   bar/restaurante grabado por los autores, no derivado de Freesound ni de
   Jamendo, así que es ajeno tanto a MUSAN como a ESC-50.

Es la imagen espejo de seal_test_set_mls_es.py: ese fijó el ruido y cambió la
voz para separar idioma de canal; este fija la voz y cambia el ruido para
separar el efecto del modelo del de la exposición al ruido. Apareado por
pair_id contra test_v3_mls_es, la diferencia es el efecto de novedad del ruido
con todo lo demás constante.

Tres decisiones que no son obvias y que verifiqué antes de escribir esto:

- **Canal 0, no el promedio de los dos.** WHAM! es binaural (la metadata trae
  `L to R Width (cm)`: 17 cm). Promediar canales — que es lo que hace
  load_noise() en los otros sellados, inocuo porque MUSAN y ESC-50 son mono —
  introduce comb filtering dependiente de la separación de micrófonos, un
  artefacto que no existe en la escena acústica real.

- **Splits `cv` + `tt`, no `tr`.** Los tres splits de WHAM! son completamente
  disjuntos por grabación fuente y por locación (verificado: 0 solapamiento en
  los tres pares). cv+tt da 61 grabaciones fuente y 19 locaciones contra las 24
  y 8 de `tt` solo; `tr` sumaría 188 más pero es el split sobre el que un
  sistema externo podría haber entrenado, y la comparación contra DeepFilterNet2
  no debería depender de asumir nada sobre su pipeline.

- **La estratificación es por grabación fuente, no por segmento.** Los 8000
  segmentos de cv+tt salen de 61 grabaciones, así que los segmentos NO son
  independientes: la unidad de remuestreo para cualquier bootstrap es la
  grabación. El reparto da conteos exactos (6 grabaciones × 5 pares + 55 × 4) y
  ninguna grabación aparece dos veces dentro del mismo bucket de SNR, de modo
  que hay independencia intra-bucket. La metadata guarda wham_file_id y
  wham_location_id justamente para que ese bootstrap sea posible después; sin
  esos campos la sede queda inanalizable.

Diferencia conocida e inocua respecto de test_v3_mls_es: clean.wav NO sale
bit-idéntico entre los dos sellados. El contenido y el recorte son los mismos,
pero el escalado final a 0,9 de pico se calcula sobre el máximo de (mixture,
clean) y la mixture cambia. Es una ganancia escalar común a noisy y clean, así
que no mueve SNR, ni PESQ (que alinea nivel internamente), ni SI-SDR ni STOI
(invariantes de escala por construcción).

ADVERTENCIA: data/raw/noise/wham es un symlink nuevo al lado de musan/ y esc50/.
RAW_NOISE_DIRS en datasets/make_mixtures.py los lista explícitamente y NO
incluye wham. Si alguien lo agrega, WHAM! entra a entrenamiento y esta sede
muere: no habría más ruido holdout en el proyecto.

Uso:
    python -m scripts.seal_test_set_wham_es
"""
import csv
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
WHAM_SPLITS = ("cv", "tt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLS_DIR = PROJECT_ROOT / "data" / "raw" / "clean_speech" / "es" / "mls"
MLS_PARQUETS = ("dev-00000-of-00001.parquet", "test-00000-of-00001.parquet")
WHAM_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "wham"

# Fuente de la voz y de los SNR: se reusan verbatim, no se sortean.
REFERENCE_METADATA = PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_metadata.json"

TEST_OUT_DIR = PROJECT_ROOT / "data" / "test_sealed" / "wham_es"
HASH_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_wham_es_hash.txt"
METADATA_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_wham_es_metadata.json"


def load_reference_conditions() -> list[dict]:
    """Load test_v3_mls_es's pairs: speech and SNR are reused verbatim.

    Returns:
        The `pairs` list of test_v3_mls_es_metadata.json, sorted by id.

    Raises:
        FileNotFoundError: if the reference metadata is missing.
        ValueError: if it does not hold exactly N_PAIRS_TOTAL pairs.
    """
    if not REFERENCE_METADATA.exists():
        raise FileNotFoundError(
            f"No existe {REFERENCE_METADATA}. Es la fuente de la voz y de los SNR "
            f"que este sellado reusa; sin ella el diseño apareado no se puede armar."
        )
    with open(REFERENCE_METADATA) as f:
        pairs = json.load(f)["pairs"]
    if len(pairs) != N_PAIRS_TOTAL:
        raise ValueError(
            f"{REFERENCE_METADATA} tiene {len(pairs)} pares, se esperaban {N_PAIRS_TOTAL}"
        )
    return sorted(pairs, key=lambda p: p["id"])


def build_mls_lookup() -> dict[str, dict]:
    """Map each MLS utterance id to the (split, row) that decodes it.

    Returns:
        {mls_id: {"split": str, "row": int}} over dev + test.

    Raises:
        FileNotFoundError: if a parquet is missing.
    """
    lookup: dict[str, dict] = {}
    for name in MLS_PARQUETS:
        path = MLS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Descargá los parquet de MLS español antes de sellar."
            )
        split = name.split("-")[0]
        ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        for row, uid in enumerate(ids):
            lookup[str(uid)] = {"split": split, "row": row}
    return lookup


def load_wham_pool() -> dict[str, list[dict]]:
    """Index WHAM! cv+tt segments, grouped by source recording.

    Segments shorter than TARGET_DURATION are dropped: cropping them would loop
    the noise, an artefact absent from every other sealed set.

    Returns:
        {file_id: [{"split", "utterance_id", "location_id", "duration_s"}, ...]},
        each list sorted for reproducibility.

    Raises:
        FileNotFoundError: if the WHAM! tree or a metadata csv is missing.
    """
    if not WHAM_DIR.exists():
        raise FileNotFoundError(
            f"No existe {WHAM_DIR}. Es un symlink al corpus WHAM!; sin él no hay "
            f"ruido holdout y este sellado no tiene sentido."
        )
    pool: dict[str, list[dict]] = defaultdict(list)
    dropped = 0
    for split in WHAM_SPLITS:
        meta = WHAM_DIR / "metadata" / f"noise_meta_{split}.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Falta {meta}, que define las grabaciones fuente.")
        with open(meta) as f:
            for row in csv.DictReader(f):
                path = WHAM_DIR / split / row["utterance_id"]
                if not path.exists():
                    raise FileNotFoundError(f"Falta el segmento {path}")
                duration = sf.info(str(path)).duration
                if duration < TARGET_DURATION:
                    dropped += 1
                    continue
                pool[row["File ID"]].append({
                    "split": split,
                    "utterance_id": row["utterance_id"],
                    "location_id": row["Location ID"],
                    "duration_s": duration,
                })
    for file_id in pool:
        pool[file_id].sort(key=lambda r: (r["split"], r["utterance_id"]))
    print(f"WHAM! {'+'.join(WHAM_SPLITS)}: {sum(len(v) for v in pool.values())} segmentos "
          f"usables en {len(pool)} grabaciones fuente "
          f"({len({s['location_id'] for v in pool.values() for s in v})} locaciones); "
          f"{dropped} descartados por durar menos de {TARGET_DURATION}s")
    return dict(pool)


def assign_recordings(reference: list[dict], file_ids: list[str],
                      rng: random.Random) -> list[str]:
    """Assign one source recording per pair, balanced and bucket-disjoint.

    Counts are exact (floor/ceil of N_PAIRS_TOTAL over the recordings) and no
    recording is used twice inside one SNR bucket, which is what gives the
    clustered bootstrap within-bucket independence.

    Args:
        reference: the reference pairs, each carrying `id` and `bucket_idx`.
        file_ids: the available WHAM! source recording ids.
        rng: seeded RNG; consumed for the shuffles only.

    Returns:
        A list of file_id, one per pair, indexed by pair id.

    Raises:
        ValueError: if the degree-constrained assignment cannot be completed.
    """
    order = sorted(file_ids)
    rng.shuffle(order)
    base, rem = divmod(len(reference), len(order))
    if base == 0:
        raise ValueError(
            f"{len(order)} grabaciones para {len(reference)} pares: alguna quedaría sin usar"
        )
    counts = {rec: base + (1 if i < rem else 0) for i, rec in enumerate(order)}

    ids_by_bucket: dict[int, list[int]] = defaultdict(list)
    for pair in reference:
        ids_by_bucket[pair["bucket_idx"]].append(pair["id"])
    capacity = {b: len(v) for b, v in ids_by_bucket.items()}
    chosen: dict[int, list[str]] = {b: [] for b in ids_by_bucket}

    # Las grabaciones con más pares restringen más: se reparten primero.
    for rec in sorted(order, key=lambda r: (-counts[r], r)):
        free = sorted(chosen, key=lambda b: (-(capacity[b] - len(chosen[b])), b))
        if counts[rec] > len(free):
            raise ValueError(
                f"La grabación {rec} pide {counts[rec]} pares y hay {len(free)} buckets"
            )
        picked = free[:counts[rec]]
        for bucket in picked:
            if len(chosen[bucket]) >= capacity[bucket]:
                raise ValueError(f"Bucket {bucket} sobreasignado al repartir {rec}")
            chosen[bucket].append(rec)

    assignment: list[str | None] = [None] * len(reference)
    for bucket, recs in chosen.items():
        if len(recs) != capacity[bucket]:
            raise ValueError(
                f"Bucket {bucket}: {len(recs)} grabaciones para {capacity[bucket]} pares"
            )
        shuffled = list(recs)
        rng.shuffle(shuffled)  # que la grabación no quede atada al snr exacto
        for pair_id, rec in zip(sorted(ids_by_bucket[bucket]), shuffled):
            assignment[pair_id] = rec
    if any(a is None for a in assignment):
        raise ValueError("Quedaron pares sin grabación asignada")
    return assignment  # type: ignore[return-value]


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


def load_wham_channel0(split: str, utterance_id: str) -> torch.Tensor:
    """Load one WHAM! segment as channel 0 only, at SAMPLE_RATE.

    Channel 0 rather than the mean of both: WHAM! is binaural, and averaging two
    microphones separated by 17 cm comb-filters the noise. See the module
    docstring.

    Raises:
        ValueError: if the segment is not 16 kHz, or not the expected 2 channels.
    """
    wav, sr = torchaudio.load(str(WHAM_DIR / split / utterance_id))
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"WHAM! {split}/{utterance_id} vino a {sr} Hz; se esperaba {SAMPLE_RATE}"
        )
    if wav.shape[0] != 2:
        raise ValueError(
            f"WHAM! {split}/{utterance_id} tiene {wav.shape[0]} canales; se esperaban 2"
        )
    return wav[0].contiguous()


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
    print("SELLADO DE TEST SET wham_es (ES audiolibro, ruido HOLDOUT — WHAM!)")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Voz y SNR: reusados verbatim de {REFERENCE_METADATA.name}")
    print(f"Ruido: WHAM! splits {' + '.join(WHAM_SPLITS)}, canal 0")
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
    mls_lookup = build_mls_lookup()
    pool = load_wham_pool()

    noise_rng = random.Random(SEED + 3)
    offset_rng = random.Random(SEED + 4)

    assignment = assign_recordings(reference, sorted(pool), noise_rng)
    per_rec = defaultdict(int)
    for rec in assignment:
        per_rec[rec] += 1
    counts = sorted(per_rec.values())
    print(f"Reparto: {len(per_rec)} grabaciones fuente, "
          f"min {counts[0]} / max {counts[-1]} pares cada una")

    # Un segmento distinto por par: no se reusa ningún wav de WHAM!.
    segment_for_pair: list[dict] = []
    taken: dict[str, int] = defaultdict(int)
    shuffled_pool = {rec: list(segs) for rec, segs in pool.items()}
    for rec in shuffled_pool:
        noise_rng.shuffle(shuffled_pool[rec])
    for pair_id, rec in enumerate(assignment):
        idx = taken[rec]
        if idx >= len(shuffled_pool[rec]):
            raise ValueError(
                f"La grabación {rec} tiene {len(shuffled_pool[rec])} segmentos usables "
                f"y se le pidieron {idx + 1}"
            )
        taken[rec] += 1
        segment_for_pair.append(shuffled_pool[rec][idx])
    print(f"Segmentos distintos usados: {len({s['utterance_id'] for s in segment_for_pair})}")
    print()

    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_len = int(TARGET_DURATION * SAMPLE_RATE)

    metadata = {
        "seed": SEED,
        "language": "es",
        "channel": "audiobook (LibriVox) — idéntico a test_v3_mls_es",
        "source": "Multilingual LibriSpeech español, splits dev + test",
        "noise_source": f"WHAM! (Wichern et al. 2019), splits {' + '.join(WHAM_SPLITS)}, canal 0",
        "noise_is_holdout": True,
        "speech_conditions_reused_from": str(REFERENCE_METADATA.relative_to(PROJECT_ROOT)),
        "n_pairs_total": N_PAIRS_TOTAL,
        "n_speakers": len({p["speaker_id"] for p in reference}),
        "n_noise_recordings": len(per_rec),
        "n_noise_locations": len({s["location_id"] for s in segment_for_pair}),
        "bootstrap_unit": "wham_file_id — los segmentos de una grabación no son independientes",
        "snr_buckets": None,  # se completa desde la referencia
        "sample_rate": SAMPLE_RATE,
        "target_duration_s": TARGET_DURATION,
        "generated_at": datetime.now().isoformat(),
        "pairs": [],
    }

    for pair_idx, (ref, segment) in enumerate(zip(reference, segment_for_pair)):
        if ref["id"] != pair_idx:
            raise ValueError(f"Referencia desalineada en el par {pair_idx}: id={ref['id']}")

        mls_id = ref["speech_source"].rsplit("/", 1)[-1]
        if mls_id not in mls_lookup:
            raise ValueError(
                f"El par {pair_idx} referencia la utterance MLS {mls_id}, que no está "
                f"en los parquet actuales. El corpus cambió y el diseño apareado se rompe."
            )
        loc = mls_lookup[mls_id]

        sp_full = decode_mls_audio(loc["split"], loc["row"])
        sp = crop_at(sp_full, target_len, ref["speech_offset"])

        ns_full = load_wham_channel0(segment["split"], segment["utterance_id"])
        ns, ns_offset = crop_random(ns_full, target_len, offset_rng)

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
            "speech_source": ref["speech_source"],
            "speaker_id": ref["speaker_id"],
            "transcript": ref["transcript"],
            "speech_offset": ref["speech_offset"],
            "noise_file": str(
                (WHAM_DIR / segment["split"] / segment["utterance_id"]).relative_to(PROJECT_ROOT)
            ),
            "noise_category": "wham",
            "wham_split": segment["split"],
            "wham_file_id": assignment[pair_idx],
            "wham_location_id": segment["location_id"],
            "noise_offset": ns_offset,
        })

        if (pair_idx + 1) % 50 == 0:
            print(f"  {pair_idx + 1}/{N_PAIRS_TOTAL}")

    buckets = sorted({p["bucket_idx"] for p in reference})
    metadata["snr_buckets"] = [
        [min(p["snr_db"] for p in reference if p["bucket_idx"] == b),
         max(p["snr_db"] for p in reference if p["bucket_idx"] == b)]
        for b in buckets
    ]

    pair_dirs = sorted(TEST_OUT_DIR.glob("pair_*"))
    dataset_hash, n_files = compute_dataset_hash(pair_dirs)

    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(
        "# Test Set wham_es (ES, canal audiolibro, ruido holdout WHAM!) - Hash de integridad\n"
        f"# Generado: {metadata['generated_at']}\n"
        f"# Seed: {SEED}\n"
        f"# Total pares: {N_PAIRS_TOTAL}\n"
        f"# Total archivos WAV: {n_files}\n"
        f"# Hablantes: {metadata['n_speakers']}\n"
        f"# Grabaciones de ruido: {metadata['n_noise_recordings']} "
        f"en {metadata['n_noise_locations']} locaciones\n"
        f"# Voz y SNR reusados de: {metadata['speech_conditions_reused_from']}\n"
        f"# Ruido: {metadata['noise_source']}\n"
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
