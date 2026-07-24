import torch
from stft import STFTHelper

stft = STFTHelper()
x = torch.randn(2, 16000)  # 1 segundo
mag, phase = stft.to_spec(x)
x_hat = stft.from_spec(mag, phase, length=16000)

err = (x - x_hat).abs().max().item()
print(f"Error de reconstrucción: {err:.2e}")
# Esperado: ~1e-6 o menos. Si es mayor, hay un bug en n_fft/hop.