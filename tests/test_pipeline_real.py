from pathlib import Path
import torch
from torch.utils.data import DataLoader

from datasets import NSDataset
from models.crn import CRN
from stft import STFTHelper   # ajustá el import si lo pusiste en otro lado

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"

# DataLoader real
ds = NSDataset(TRAIN_DIR, segment_samples=64000)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
noisy, clean = next(iter(loader))
print(f"Audio batch: noisy={tuple(noisy.shape)}, clean={tuple(clean.shape)}")

# STFT
stft = STFTHelper(n_fft=320, hop_length=160)
mag_noisy, phase_noisy = stft.to_spec(noisy)
mag_clean, _           = stft.to_spec(clean)
print(f"Spectrogram: mag_noisy={tuple(mag_noisy.shape)}, mag_clean={tuple(mag_clean.shape)}")

# Modelo
model = CRN()
mag_est = model(mag_noisy)
print(f"Estimated:   mag_est={tuple(mag_est.shape)}")

# Reconstrucción al dominio temporal
audio_est = stft.from_spec(mag_est, phase_noisy, length=64000)
print(f"Reconstructed audio: {tuple(audio_est.shape)}")

# Loss MSE entre magnitudes
loss = torch.nn.functional.mse_loss(mag_est, mag_clean)
print(f"Initial loss (random model): {loss.item():.4f}")

# Backward
loss.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm: {grad_norm:.4f}")
