# baselines/butterworth.py
import torch
import torchaudio
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR    = PROJECT_ROOT / "data" / "processed" / "val"
OUTPUT_DIR = PROJECT_ROOT / "data" / "estimates" / "butterworth"

SAMPLE_RATE = 16000
CUTOFF_HZ   = 4000  # frecuencia de corte: voz típicamente bajo 4kHz
ORDER       = 5

def butterworth_lowpass(audio_np, sr=SAMPLE_RATE, cutoff=CUTOFF_HZ, order=ORDER):
    """Aplica un pasabajo Butterworth orden 5 con fase cero (zero-phase)."""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype="low", output="sos")
    return sosfiltfilt(sos, audio_np)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_pairs = sorted(VAL_DIR.iterdir())

    print(f"Processing {len(val_pairs)} pairs with Butterworth LP {CUTOFF_HZ} Hz, order {ORDER}...")

    for pair_dir in val_pairs:
        noisy, _ = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair_dir / "clean.wav"))
        noisy_np = noisy.squeeze(0).numpy()
        clean_np = clean.squeeze(0).numpy()

        filtered_np = butterworth_lowpass(noisy_np)
        filtered = torch.from_numpy(filtered_np.copy()).float()

        out_subdir = OUTPUT_DIR / pair_dir.name
        out_subdir.mkdir(exist_ok=True)
        torchaudio.save(str(out_subdir / "noisy.wav"),    noisy,                  SAMPLE_RATE)
        torchaudio.save(str(out_subdir / "clean.wav"),    clean,                  SAMPLE_RATE)
        torchaudio.save(str(out_subdir / "filtered.wav"), filtered.unsqueeze(0),  SAMPLE_RATE)

    print(f"Done. Outputs at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
