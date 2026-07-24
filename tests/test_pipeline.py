# test_pipeline.py
import torch
from crn import CRN
from stft import STFTHelper  # tu wrapper

# Simular 1 segundo de audio (lo que vendrá del DataLoader)
B = 2
audio_noisy = torch.randn(B, 16000)
audio_clean = torch.randn(B, 16000)

# 1) STFT del noisy
stft = STFTHelper(n_fft=320, hop_length=160)
mag_noisy, phase_noisy = stft.to_spec(audio_noisy)
print(f"mag_noisy shape: {mag_noisy.shape}")
# esperado: [2, T_frames, 161] — T_frames ≈ 1 + 16000//160 = 101

# 2) STFT del clean (target para la loss)
mag_clean, _ = stft.to_spec(audio_clean)
print(f"mag_clean shape: {mag_clean.shape}")

# 3) CRN sobre magnitud noisy
m = CRN()
mag_est = m(mag_noisy)
print(f"mag_est shape:   {mag_est.shape}")
# esperado: igual que mag_noisy

# 4) Reconstrucción usando fase del noisy
audio_est = stft.from_spec(mag_est, phase_noisy, length=16000)
print(f"audio_est shape: {audio_est.shape}")
# esperado: [2, 16000]

# 5) Loss MSE simple (sin máscara por ahora)
loss = torch.nn.functional.mse_loss(mag_est, mag_clean)
print(f"loss: {loss.item():.4f}")

# 6) Backward (¿gradientes fluyen?)
loss.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
print(f"grad_norm: {grad_norm:.4f}")
# esperado: > 0 y < 1000 (no NaN, no infinito)
