# datasets/make_mixtures.py 
import torch
import torchaudio
import random
from pathlib import Path
import random
import pandas as pd

SAMPLE_RATE = 16000
SNR_RANGE = [0, 5, 10, 15, 20]
TARGET_DURATION = 4.0
N_TRAIN=50000
N_VAL=2000
SEED = 42




PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rutas reales:
RAW_SPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "clean_speech" / "en" / "librispeech" / "LibriSpeech"
RAW_NOISE_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "noise" / "musan",
    PROJECT_ROOT / "data" / "raw" / "noise" / "esc50",
]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED_ES = PROJECT_ROOT / "data" / "processed_es"

# Extensiones a buscar:
SPEECH_EXTS = ("*.flac", "*.mp3")  
NOISE_EXTS  = ("*.wav",)          # MUSAN y ESC-50 son .wav

def collect_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)  # orden estable: rglob() no garantiza el mismo orden entre corridas



def load_resample_mono(path, target_sr=SAMPLE_RATE):
    wav, sr = torchaudio.load(str(path))  # ← acepta .flac Y .mp3
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

def pad_or_crop(x, target_len):
    if x.numel() >= target_len:
        start = random.randint(0, x.numel() - target_len)
        return x[start:start + target_len]
    repeats = target_len // x.numel() + 1
    return x.repeat(repeats)[:target_len]

def make_pairs(speech_files, noise_files, out_dir, n_pairs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_len = int(TARGET_DURATION * SAMPLE_RATE)

    for i in range(n_pairs):
        sp = load_resample_mono(random.choice(speech_files))
        ns = load_resample_mono(random.choice(noise_files))
        #snr = random.choice(SNR_RANGE)
        snr = random.uniform(-5, 15)

        sp = pad_or_crop(sp, target_len)
        ns = pad_or_crop(ns, target_len)

        # RMS normalization (etapa 1 del paper Tan & Wang)
        c  = torch.sqrt(torch.tensor(target_len, dtype=torch.float32) / (sp.pow(2).sum() + 1e-8))
        sp = sp * c

        mixture, clean = mix_at_snr(sp, ns, snr)

        # ── FIX: peak normalization para evitar clipping en WAV 16-bit ──
        peak = max(mixture.abs().max(), clean.abs().max(), torch.tensor(1e-8))
        scale = 0.9 / peak
        mixture = mixture * scale
        clean   = clean * scale
        # ─────────────────────────────────────────────────────────────

        pair_dir = out_dir / f"pair_{i:04d}"
        pair_dir.mkdir(exist_ok=True)
        torchaudio.save(str(pair_dir / "noisy.wav"), mixture.unsqueeze(0), SAMPLE_RATE)
        torchaudio.save(str(pair_dir / "clean.wav"), clean.unsqueeze(0),   SAMPLE_RATE)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_pairs}")


def collect_common_voice_files(manifest_path: Path,
                                 balance_gender: bool = False,
                                 random_seed: int = 42) -> list[Path]:
    """
    Recolecta archivos MP3 de Common Voice ES desde un manifest ya filtrado
    y verificado sin leakage por scripts/analyze_cv26_es.py (train_manifest.tsv,
    dev_manifest.tsv o test_manifest.tsv). No lee validated.tsv directo: ese
    archivo mezcla los tres splits oficiales y rompe la separación por
    hablante/frase que ya se verificó entre ellos.

    Args:
        manifest_path: ej. data/interim/cv26_es/train_manifest.tsv
        balance_gender: si True, balancea 50/50 masculino/femenino
        random_seed: para reproducibilidad del muestreo balanceado

    Returns:
        lista de paths absolutos a los archivos MP3
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Corré antes: python -m scripts.analyze_cv26_es"
        )

    manifest = pd.read_csv(manifest_path, sep="\t")
    print(f"Common Voice ES ({manifest_path.name}): {len(manifest)} clips")

    # Balanceo de género opcional
    if balance_gender:
        males = manifest[manifest["gender"] == "male_masculine"]
        females = manifest[manifest["gender"] == "female_feminine"]
        min_count = min(len(males), len(females))
        males = males.sample(min_count, random_state=random_seed)
        females = females.sample(min_count, random_state=random_seed)
        manifest = pd.concat([males, females])
        print(f"  Balanceado por género: {len(manifest)} clips ({min_count} M + {min_count} F)")

    speech_files = [Path(p) for p in manifest["filepath"]]

    # Verificar que los archivos existan (algunos pueden faltar)
    existing = [p for p in speech_files if p.exists()]
    if len(existing) < len(speech_files):
        missing = len(speech_files) - len(existing)
        print(f"  ⚠ {missing} archivos referenciados en manifest pero no encontrados en disco")

    return existing



CV_MANIFEST_DIR = PROJECT_ROOT / "data" / "interim" / "cv26_es"

if __name__ == "__main__":
    random.seed(SEED)  # reproducibilidad: TODAS las operaciones aleatorias de este script dependen de esta semilla

    language = "es"  # o "en"

    noise_files = []
    for d in RAW_NOISE_DIRS:
        noise_files.extend(collect_files(d, NOISE_EXTS))
    print(f"Noise files: {len(noise_files)}")
    assert len(noise_files) > 100, "Muy pocos archivos de ruido"

    if language == "en":
        speech_files = collect_files(RAW_SPEECH_DIR, SPEECH_EXTS)
        print(f"Speech files: {len(speech_files)}")
        assert len(speech_files) > 100, f"Muy pocos archivos de voz en {RAW_SPEECH_DIR}"

        print("→ Generando train pairs...")
        make_pairs(speech_files, noise_files, PROCESSED / "train", N_TRAIN)
        print("→ Generando val pairs...")
        make_pairs(speech_files, noise_files, PROCESSED / "val", N_VAL)

    elif language == "es":
        # train y val vienen de splits disjuntos por hablante/frase (verificado
        # en scripts/analyze_cv26_es.py) para que processed_es/val sea una
        # validación real y no contamine el futuro test_sealed/v2_es.
        train_speech = collect_common_voice_files(
            manifest_path=CV_MANIFEST_DIR / "train_manifest.tsv",
            balance_gender=True,  # recomendado por sesgo demográfico
            random_seed=42,
        )
        val_speech = collect_common_voice_files(
            manifest_path=CV_MANIFEST_DIR / "dev_manifest.tsv",
            balance_gender=True,
            random_seed=42,
        )
        print(f"Speech files train: {len(train_speech)}, val: {len(val_speech)}")
        assert len(train_speech) > 100, "Muy pocos archivos de voz en train_manifest.tsv"
        assert len(val_speech) > 100, "Muy pocos archivos de voz en dev_manifest.tsv"

        print("→ Generando train pairs...")
        make_pairs(train_speech, noise_files, PROCESSED_ES / "train", N_TRAIN)
        print("→ Generando val pairs...")
        make_pairs(val_speech, noise_files, PROCESSED_ES / "val", N_VAL)

    else:
        raise ValueError(f"Idioma no soportado: {language}")

    print("Listo.") 