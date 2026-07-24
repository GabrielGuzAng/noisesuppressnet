# analysis/plot_psd_comparison.py
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENH_DIR = PROJECT_ROOT / "data" / "estimates" / "v0"
BUT_DIR = PROJECT_ROOT / "data" / "estimates" / "butterworth"

SAMPLE_RATE = 16000
N_FFT = 512

def compute_psd_db(audio, n_fft=N_FFT):
    """PSD promediada en el tiempo, en dB."""
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=n_fft//4, power=2)(audio)
    psd = spec.mean(dim=-1)              # promedio sobre tiempo → [F]
    psd_db = 10 * torch.log10(psd + 1e-10)
    return psd_db.numpy()

def collect_psd(directory, audio_name):
    """Acumula PSD de todos los pares del directorio."""
    psds = []
    for pair_dir in sorted(directory.iterdir()):
        wav, _ = torchaudio.load(str(pair_dir / audio_name))
        psds.append(compute_psd_db(wav.squeeze(0)))
    return np.stack(psds)  # [N_clips, F]

# Recolectar de los 50 pares en cada condición
psd_noisy   = collect_psd(ENH_DIR, "noisy.wav")
psd_clean   = collect_psd(ENH_DIR, "clean.wav")
psd_enh     = collect_psd(ENH_DIR, "enhanced.wav")
psd_butter  = collect_psd(BUT_DIR, "filtered.wav")

# Frecuencias correspondientes
freqs = np.linspace(0, SAMPLE_RATE / 2, psd_noisy.shape[1])

# Plot con media y banda de ±1 std
fig, ax = plt.subplots(figsize=(11, 6))
for label, psd_arr, color in [
    ("Noisy",          psd_noisy,  "#888888"),
    ("Butterworth LP", psd_butter, "#E37222"),
    ("CRN v0 (ours)",  psd_enh,    "#2E7D32"),
    ("Clean (target)", psd_clean,  "#1565C0"),
]:
    mean = psd_arr.mean(axis=0)
    std  = psd_arr.std(axis=0)
    ax.plot(freqs / 1000, mean, label=label, color=color, lw=2)
    ax.fill_between(freqs / 1000, mean - std, mean + std, color=color, alpha=0.15)

ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Power Spectral Density (dB)")
ax.set_title("Average PSD across 50 validation clips (mean ± std)")
ax.set_xlim(0, 8)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "psd_comparison.png", dpi=130, bbox_inches="tight")
print("Saved psd_comparison.png")
