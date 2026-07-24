# analysis/plot_spectrogram.py
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PAIR_NAME     = "pair_0000"   # cambialo para mirar distintos casos
ENH_DIR       = PROJECT_ROOT / "data" / "estimates" / "v0"        / PAIR_NAME
BUT_DIR       = PROJECT_ROOT / "data" / "estimates" / "butterworth" / PAIR_NAME
OUT_FIG       = PROJECT_ROOT / "figures" / f"spec_{PAIR_NAME}.png"

SAMPLE_RATE = 16000

def load_mono(path):
    wav, _ = torchaudio.load(str(path))
    return wav.squeeze(0)

def spectrogram_db(audio, n_fft=512, hop_length=128):
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)(audio)
    return torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(spec)

def main():
    audios = {
        "Noisy":            load_mono(ENH_DIR / "noisy.wav"),
        "Butterworth LP":   load_mono(BUT_DIR / "filtered.wav"),
        "CRN v0 (ours)":    load_mono(ENH_DIR / "enhanced.wav"),
        "Clean (target)":   load_mono(ENH_DIR / "clean.wav"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (label, audio) in zip(axes.ravel(), audios.items()):
        spec_db = spectrogram_db(audio).numpy()
        im = ax.imshow(spec_db, origin="lower", aspect="auto",
                       extent=[0, audio.numel() / SAMPLE_RATE, 0, SAMPLE_RATE / 2 / 1000],
                       cmap="magma")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        plt.colorbar(im, ax=ax, label="dB")

    fig.suptitle(f"Spectrogram comparison — {PAIR_NAME}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")

if __name__ == "__main__":
    main()
