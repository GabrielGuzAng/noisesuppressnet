import torch
import torchaudio
import random
import hashlib
import json
from pathlib import Path
from datetime import datetime


# ── Configuración (NO cambiar después del primer sellado) ──
SEED = 42
SAMPLE_RATE = 16000
TARGET_DURATION = 4.0
N_PAIRS_PER_BUCKET = 50
SNR_BUCKETS = [(-5, 0), (0, 5), (5, 10), (10, 15),(15,20)]  # dB

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_SPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "clean_speech" / "en" / "librispeech" / "LibriSpeech"
RAW_NOISE_DIRS = {
    "musan": PROJECT_ROOT / "data" / "raw" / "noise" / "musan",
    "esc50": PROJECT_ROOT / "data" / "raw" / "noise" / "esc50",
}

# Salida del test sellado — RUTA DIFERENTE a data/processed
TEST_OUT_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
HASH_FILE = PROJECT_ROOT / "data" / "test_v1_hash.txt"
METADATA_FILE = PROJECT_ROOT / "data" / "test_v1_metadata.json"

SPEECH_EXTS = ("*.flac",)
NOISE_EXTS = ("*.wav",)


def collect_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)  # sorted para reproducibilidad


def load_resample_mono(path, target_sr=SAMPLE_RATE):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def rms(x):
    return torch.sqrt(torch.mean(x**2) + 1e-9)


def mix_at_snr(speech, noise, snr_db):
    target_noise_rms = rms(speech) / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / rms(noise))
    return speech + scaled_noise, speech


def pad_or_crop_deterministic(x, target_len, offset_rng):
    """Determinista con RNG externo — clave para reproducibilidad."""
    if x.numel() >= target_len:
        max_start = x.numel() - target_len
        start = offset_rng.randint(0, max_start)
        return x[start:start + target_len], start
    repeats = target_len // x.numel() + 1
    return x.repeat(repeats)[:target_len], 0


def get_noise_category(noise_path):
    """Devuelve el nombre del subdirectorio raíz de ruido."""
    for key, root in RAW_NOISE_DIRS.items():
        try:
            noise_path.relative_to(root)
            return key
        except ValueError:
            continue
    return "unknown"


def compute_file_hash(path):
    """SHA-256 de un archivo."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dataset_hash(pair_dirs):
    """Hash acumulativo de todos los archivos, ordenado alfabéticamente."""
    combined = hashlib.sha256()
    all_files = []
    for pd in sorted(pair_dirs):
        for f in sorted(pd.iterdir()):
            if f.suffix == ".wav":
                all_files.append(f)
    for f in all_files:
        combined.update(f.name.encode())
        combined.update(compute_file_hash(f).encode())
    return combined.hexdigest(), len(all_files)


def seal_test_set():
    print("=" * 70)
    print("SELLADO DE TEST SET v1 (EN)")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Buckets SNR: {SNR_BUCKETS}")
    print(f"Pares por bucket: {N_PAIRS_PER_BUCKET}")
    print(f"Total pares esperados: {N_PAIRS_PER_BUCKET * len(SNR_BUCKETS)}")
    print(f"Salida: {TEST_OUT_DIR}")
    print()

    # Verificar que no existe (evitar sobrescribir sellado previo)
    if TEST_OUT_DIR.exists() and any(TEST_OUT_DIR.iterdir()):
        raise RuntimeError(
            f"❌ ABORT: {TEST_OUT_DIR} ya existe con contenido.\n"
            f"El test set sellado NO debe regenerarse.\n"
            f"Si querés forzar la regeneración, borralo manualmente PRIMERO\n"
            f"y confirmá que no rompés reproducibilidad de resultados previos."
        )

    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recolectar archivos
    speech_files = collect_files(RAW_SPEECH_DIR, SPEECH_EXTS)
    noise_files = []
    for name, d in RAW_NOISE_DIRS.items():
        noise_files.extend(collect_files(d, NOISE_EXTS))

    print(f"Archivos de voz disponibles: {len(speech_files)}")
    print(f"Archivos de ruido disponibles: {len(noise_files)}")
    assert len(speech_files) > 200, "Muy pocos speech files"
    assert len(noise_files) > 50, "Muy pocos noise files"
    print()

    # RNG separados para diferentes decisiones — permite trazabilidad
    file_rng = random.Random(SEED)
    snr_rng = random.Random(SEED + 1)
    offset_rng = random.Random(SEED + 2)

    target_len = int(TARGET_DURATION * SAMPLE_RATE)
    metadata = {
        "seed": SEED,
        "n_pairs_total": N_PAIRS_PER_BUCKET * len(SNR_BUCKETS),
        "snr_buckets": [list(b) for b in SNR_BUCKETS],
        "sample_rate": SAMPLE_RATE,
        "target_duration_s": TARGET_DURATION,
        "generated_at": datetime.now().isoformat(),
        "pairs": [],
    }

    pair_idx = 0
    for bucket_idx, (snr_min, snr_max) in enumerate(SNR_BUCKETS):
        print(f"Bucket {bucket_idx + 1}/{len(SNR_BUCKETS)}: SNR [{snr_min}, {snr_max}] dB")

        for i in range(N_PAIRS_PER_BUCKET):
            speech_path = file_rng.choice(speech_files)
            noise_path = file_rng.choice(noise_files)
            snr = snr_rng.uniform(snr_min, snr_max)

            sp = load_resample_mono(speech_path)
            ns = load_resample_mono(noise_path)

            sp, sp_offset = pad_or_crop_deterministic(sp, target_len, offset_rng)
            ns, ns_offset = pad_or_crop_deterministic(ns, target_len, offset_rng)

            # RMS normalization
            c = torch.sqrt(torch.tensor(target_len, dtype=torch.float32) / (sp.pow(2).sum() + 1e-8))
            sp = sp * c

            mixture, clean = mix_at_snr(sp, ns, snr)

            # Peak norm anti-clipping
            peak = max(mixture.abs().max(), clean.abs().max(), torch.tensor(1e-8))
            scale = 0.9 / peak
            mixture = mixture * scale
            clean = clean * scale

            # Guardar
            pair_dir = TEST_OUT_DIR / f"pair_{pair_idx:04d}"
            pair_dir.mkdir(exist_ok=True)
            torchaudio.save(str(pair_dir / "noisy.wav"), mixture.unsqueeze(0), SAMPLE_RATE)
            torchaudio.save(str(pair_dir / "clean.wav"), clean.unsqueeze(0), SAMPLE_RATE)

            # Metadata
            metadata["pairs"].append({
                "id": pair_idx,
                "bucket_idx": bucket_idx,
                "snr_db": round(float(snr), 3),
                "speech_file": str(speech_path.relative_to(PROJECT_ROOT)),
                "noise_file": str(noise_path.relative_to(PROJECT_ROOT)),
                "noise_category": get_noise_category(noise_path),
                "speech_offset": sp_offset,
                "noise_offset": ns_offset,
            })

            pair_idx += 1

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N_PAIRS_PER_BUCKET}")

    # Guardar metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata: {METADATA_FILE}")

    # Calcular hash acumulativo
    print("\nCalculando hash SHA-256 del test set completo...")
    pair_dirs = [p for p in TEST_OUT_DIR.iterdir() if p.is_dir()]
    dataset_hash, n_files = compute_dataset_hash(pair_dirs)

    # Escribir hash file
    with open(HASH_FILE, "w") as f:
        f.write(f"# Test Set v1 EN - Hash de integridad\n")
        f.write(f"# Generado: {datetime.now().isoformat()}\n")
        f.write(f"# Seed: {SEED}\n")
        f.write(f"# Total pares: {pair_idx}\n")
        f.write(f"# Total archivos WAV: {n_files}\n")
        f.write(f"# Hash SHA-256 acumulativo:\n")
        f.write(f"{dataset_hash}\n")

    print(f"Hash file: {HASH_FILE}")
    print(f"SHA-256: {dataset_hash}")

    # Resumen
    print("\n" + "=" * 70)
    print(f"✅ TEST SET v1 SELLADO")
    print("=" * 70)
    print(f"  Directorio: {TEST_OUT_DIR}")
    print(f"  Pares generados: {pair_idx}")
    print(f"  Metadata: {METADATA_FILE}")
    print(f"  Hash: {HASH_FILE}")
    print()
    print("SIGUIENTE PASO — Committear el hash y la metadata (NO los WAV):")
    print()
    print(f"  git add {HASH_FILE.relative_to(PROJECT_ROOT)} \\")
    print(f"          {METADATA_FILE.relative_to(PROJECT_ROOT)} \\")
    print(f"          scripts/seal_test_set_en.py")
    print(f"  git commit -m 'Seal test set v1 (EN, {pair_idx} pairs)'")
    print(f"  git tag -a test_set_v1 -m 'Sealed test set v1'")
    print(f"  git push && git push --tags")
    print()
    print("NOTA: Los archivos WAV del test están en data/test_sealed/ que")
    print("      está cubierto por .gitignore. NO se suben al repo — solo")
    print("      el hash prueba su integridad.")


if __name__ == "__main__":
    seal_test_set()
