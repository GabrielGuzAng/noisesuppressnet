"""baselines/butterworth.py

Non-DL reference baseline: a Butterworth low-pass filter.

Exposes the filter both as a batch script over a directory of pairs and as a
single-waveform function, so that `evaluation.evaluate_variant` can score it on
the sealed test sets with the same machinery used for the trained variants.

The filter has two modes. Zero-phase (`sosfiltfilt`, the default) filters
forward and backward: it is NOT causal and therefore does not meet the
operating constraint the CRN is held to. It is kept as the default because the
V0-era numbers were produced with it, and because beating a baseline that is
allowed to see the future is the stronger claim. `causal=True` (`sosfilt`) is
the apples-to-apples comparison.
"""
import torch
import torchaudio
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR    = PROJECT_ROOT / "data" / "processed" / "val"
OUTPUT_DIR = PROJECT_ROOT / "data" / "estimates" / "butterworth"

SAMPLE_RATE = 16000
CUTOFF_HZ   = 4000  # frecuencia de corte: voz típicamente bajo 4kHz
ORDER       = 5

def butterworth_lowpass(audio_np, sr=SAMPLE_RATE, cutoff=CUTOFF_HZ, order=ORDER,
                        causal=False):
    """Apply a Butterworth low-pass filter to a 1-D numpy waveform.

    Args:
        audio_np: 1-D waveform.
        sr: Sample rate in Hz.
        cutoff: Cutoff frequency in Hz.
        order: Filter order.
        causal: If True, use a single forward pass (`sosfilt`), which is causal
            and introduces phase distortion. If False (default), use zero-phase
            filtering (`sosfiltfilt`), which is not causal.

    Returns:
        The filtered waveform, same shape as the input.
    """
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype="low", output="sos")
    if causal:
        return sosfilt(sos, audio_np)
    return sosfiltfilt(sos, audio_np)


def apply_to_waveform(wav, sr=SAMPLE_RATE, cutoff=CUTOFF_HZ, order=ORDER,
                      causal=False):
    """Filter a torch waveform, returning a torch waveform.

    Thin adapter so the baseline can be dropped into the same evaluation loop
    as the trained variants. `sosfiltfilt` returns a reversed-strides view that
    torch cannot wrap, hence the copy.

    Args:
        wav: 1-D float tensor.
        sr: Sample rate in Hz.
        cutoff: Cutoff frequency in Hz.
        order: Filter order.
        causal: See `butterworth_lowpass`.

    Returns:
        A 1-D float tensor on the CPU, same length as the input.
    """
    filtered = butterworth_lowpass(wav.detach().cpu().numpy(), sr=sr,
                                   cutoff=cutoff, order=order, causal=causal)
    return torch.from_numpy(np.ascontiguousarray(filtered)).float()

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
