# model/stft.py
import torch

class STFTHelper:
    def __init__(self, n_fft=320, hop_length=160, sr=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self._window = None
    
    def window(self, device):
        if self._window is None or self._window.device != device:
            self._window = torch.hamming_window(self.n_fft).to(device)
        return self._window
    
    def to_spec(self, x):
        """x: [B, T_samples] → (mag [B, T_frames, F], phase [B, T_frames, F])"""
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window(x.device), return_complex=True,
                       center=True)
        # X: [B, F, T_frames] → transponer a [B, T_frames, F]
        X = X.transpose(1, 2)
        return X.abs(), X.angle()
    
    def from_spec(self, mag, phase, length=None):
        """mag, phase: [B, T_frames, F] → x: [B, T_samples]"""
        X = mag * torch.exp(1j * phase)
        X = X.transpose(1, 2)  # [B, F, T_frames]
        return torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=self.window(X.device), length=length)
