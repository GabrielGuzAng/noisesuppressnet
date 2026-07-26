# datasets/make_mixtures.py 
import torch
import torchaudio
import random
from pathlib import Path
import random

SAMPLE_RATE = 16000
SNR_RANGE = [0, 5, 10, 15, 20]
TARGET_DURATION = 4.0
N_TRAIN=50000
N_VAL=2000




PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rutas reales:
RAW_SPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "clean_speech" / "en" / "librispeech" / "LibriSpeech"
RAW_NOISE_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "noise" / "musan",
    PROJECT_ROOT / "data" / "raw" / "noise" / "esc50",
]
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Extensiones a buscar:
SPEECH_EXTS = ("*.flac",)         # LibriSpeech es .flac
NOISE_EXTS  = ("*.wav",)          # MUSAN y ESC-50 son .wav

def collect_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return files

def load_resample_mono(path, target_sr=SAMPLE_RATE):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono
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


if __name__ == "__main__":
    speech_files = collect_files(RAW_SPEECH_DIR, SPEECH_EXTS)
    noise_files  = []
    for d in RAW_NOISE_DIRS:
        noise_files.extend(collect_files(d, NOISE_EXTS))

    print(f"Speech files: {len(speech_files)}")
    print(f"Noise files:  {len(noise_files)}")
    assert len(speech_files) > 100, f"Muy pocos archivos de voz en {RAW_SPEECH_DIR}"
    assert len(noise_files)  > 100, "Muy pocos archivos de ruido"

    print("→ Generando train pairs...")
    make_pairs(speech_files, noise_files, PROCESSED / "train", N_TRAIN)
    print("→ Generando val pairs...")
    make_pairs(speech_files, noise_files, PROCESSED / "val", N_VAL)
    print("Listo.")
