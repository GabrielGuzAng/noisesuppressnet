# training/losses.py
"""
Funciones de loss para las distintas variantes.

- mse_magnitude: V1 (baseline reproducible del paper Tan & Wang 2018)
- si_sdr_loss:   componente temporal para V2
- mse_plus_sisdr: V2 (combinada, α * MSE + (1-α) * SI-SDR)

Referencias:
- Tan & Wang 2018: MSE sobre magnitud STFT
- Le Roux et al. 2019: "SDR — Half-baked or Well Done?" ICASSP 2019
- Braun & Tashev 2021: combinación de losses en speech enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_magnitude(mag_est, mag_clean):
    """
    MSE sobre magnitud STFT.
    
    Args:
        mag_est: [B, 1, T_frames, F_bins] magnitud estimada
        mag_clean: [B, 1, T_frames, F_bins] magnitud clean target
    Returns:
        loss escalar
    """
    return F.mse_loss(mag_est, mag_clean)


def si_sdr_loss(audio_est, audio_clean, eps=1e-8):
    """
    Scale-Invariant SDR como loss (negativo de SI-SDR en dB).
    
    Minimizar esta loss = maximizar SI-SDR = mejor separación temporal.
    
    Args:
        audio_est: [B, T_samples] audio estimado
        audio_clean: [B, T_samples] audio clean target
    Returns:
        loss escalar (positiva si SI-SDR es negativo, negativa si SI-SDR es positivo)
    """
    # Asegurar 2D: [B, T]
    if audio_est.dim() == 3:  # [B, 1, T]
        audio_est = audio_est.squeeze(1)
    if audio_clean.dim() == 3:
        audio_clean = audio_clean.squeeze(1)
    
    # Zero-mean por batch (SI-SDR es scale y offset invariante)
    audio_est = audio_est - audio_est.mean(dim=-1, keepdim=True)
    audio_clean = audio_clean - audio_clean.mean(dim=-1, keepdim=True)
    
    # Proyección de audio_est sobre audio_clean
    dot = (audio_est * audio_clean).sum(dim=-1, keepdim=True)
    target_norm = (audio_clean ** 2).sum(dim=-1, keepdim=True) + eps
    proj = (dot / target_norm) * audio_clean
    
    # Error (componente ortogonal a target)
    noise = audio_est - proj
    
    # SI-SDR en dB
    signal_power = (proj ** 2).sum(dim=-1) + eps
    noise_power = (noise ** 2).sum(dim=-1) + eps
    si_sdr = 10 * torch.log10(signal_power / noise_power)
    
    # Loss = negativo (queremos maximizar SI-SDR)
    return -si_sdr.mean()


def mse_plus_sisdr(mag_est, mag_clean, audio_est, audio_clean, alpha=0.7,
                   sisdr_scale=0.03):
    """
    Loss combinada: α * MSE_magnitud + (1-α) * SI-SDR normalizado.
    
    Escalas típicas observadas:
    - MSE_magnitud: rango ~0.05 a 0.5 (según convergencia)
    - -SI-SDR:      rango ~-15 a -3 dB (magnitud absoluta 3 a 15)
    
    Sin escalado, SI-SDR dominaría por magnitud absoluta.
    Aplicamos sisdr_scale (~1/30) para llevar SI-SDR al rango de MSE.
    
    Args:
        mag_est: magnitud estimada [B, 1, T_frames, F_bins]
        mag_clean: magnitud clean [B, 1, T_frames, F_bins]
        audio_est: audio reconstruido [B, T_samples]
        audio_clean: audio clean [B, T_samples]
        alpha: peso del MSE (0.7 = 70% MSE, 30% SI-SDR)
        sisdr_scale: factor de normalización de escala del SI-SDR
    Returns:
        (loss_total, dict con componentes para logging)
    """
    mse_val = mse_magnitude(mag_est, mag_clean)
    sisdr_val = si_sdr_loss(audio_est, audio_clean)
    
    loss = alpha * mse_val + (1 - alpha) * sisdr_val * sisdr_scale
    
    return loss, {
        "mse_component": mse_val.item(),
        "sisdr_component": sisdr_val.item(),  # negativo SI-SDR en dB
        "sisdr_scaled": (sisdr_val * sisdr_scale).item(),
    }


# Factory para elegir loss según config
LOSS_REGISTRY = {
    "mse_magnitude": "mse_magnitude",
    "mse_plus_sisdr": "mse_plus_sisdr",
}


def get_loss_name(config):
    """Devuelve el nombre de la loss según config, con default seguro."""
    return config.get("loss", "mse_magnitude")


if __name__ == "__main__":
    # Sanity test
    print("Testing losses.py...")
    
    B, T = 4, 64000
    n_frames = T // 160 + 1  # hop_length=160
    F_bins = 161
    
    # Datos dummy
    mag_est = torch.rand(B, 1, n_frames, F_bins, requires_grad=True)
    mag_clean = torch.rand(B, 1, n_frames, F_bins)
    audio_est = torch.randn(B, T, requires_grad=True)
    audio_clean = torch.randn(B, T)
    
    # Test MSE
    loss_mse = mse_magnitude(mag_est, mag_clean)
    loss_mse.backward()
    print(f"  MSE loss: {loss_mse.item():.4f}, grad OK: {mag_est.grad.norm().item():.2e}")
    
    # Test SI-SDR
    audio_est_grad = torch.randn(B, T, requires_grad=True)
    loss_sisdr = si_sdr_loss(audio_est_grad, audio_clean)
    loss_sisdr.backward()
    print(f"  SI-SDR loss: {loss_sisdr.item():.4f}, grad OK: {audio_est_grad.grad.norm().item():.2e}")
    
    # Test combined
    mag_est_grad = torch.rand(B, 1, n_frames, F_bins, requires_grad=True)
    audio_est_grad = torch.randn(B, T, requires_grad=True)
    loss_combined, components = mse_plus_sisdr(
        mag_est_grad, mag_clean, audio_est_grad, audio_clean, alpha=0.7
    )
    loss_combined.backward()
    print(f"  Combined loss: {loss_combined.item():.4f}")
    print(f"    MSE component:   {components['mse_component']:.4f}")
    print(f"    SI-SDR component: {components['sisdr_component']:.4f} dB")
    print(f"    SI-SDR scaled:    {components['sisdr_scaled']:.4f}")
    print("✓ All losses functional")
