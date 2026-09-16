# stft.py
"""STFT helper with an optional strictly-causal framing mode.

Two framing modes coexist on purpose:

`causal=False` (default) reproduces the original behaviour bit for bit:
`torch.stft(center=True)`, which reflection-pads n_fft//2 on both sides. Frame
`t` is centred on sample `t*hop`, so producing it needs 160 samples (10.0 ms) of
future. V1, V2, V3, V3b and V3e were all trained under this framing and must
keep being evaluated under it, so the default must never change.

`causal=True` pads `n_fft` zeros at the START only and uses `center=False`, so
frame `t` covers the window ENDING at sample `t*hop` and never reads ahead. The
frame count is unchanged (padding n_fft keeps 1 + T//hop frames), which means
the model and everything downstream see the same shapes.

Motivation: Tan & Wang 2018, section 1, states that "a delay of longer than 10
milliseconds is objectionable". The default framing sits exactly at that bound
before counting any compute, which the causal mode removes.
"""
import torch


class STFTHelper:
    def __init__(self, n_fft=320, hop_length=160, sr=16000, causal=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.causal = causal
        self._window = None

    def window(self, device):
        if self._window is None or self._window.device != device:
            self._window = torch.hamming_window(self.n_fft).to(device)
        return self._window

    def to_spec(self, x):
        """x: [B, T_samples] → (mag [B, T_frames, F], phase [B, T_frames, F])"""
        if self.causal:
            # Solo contexto pasado: el pad va al inicio y nunca al final.
            x = torch.nn.functional.pad(x, (self.n_fft, 0))
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window(x.device), return_complex=True,
                       center=not self.causal)
        # X: [B, F, T_frames] → transponer a [B, T_frames, F]
        X = X.transpose(1, 2)
        return X.abs(), X.angle()

    def from_spec(self, mag, phase, length=None):
        """mag, phase: [B, T_frames, F] → x: [B, T_samples]"""
        X = mag * torch.exp(1j * phase)
        X = X.transpose(1, 2)  # [B, F, T_frames]
        if self.causal:
            # Reconstruir con el pad incluido y recortarlo después, para que la
            # normalización de overlap-add vea las mismas ventanas que el forward.
            padded_len = None if length is None else length + self.n_fft
            y = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                            window=self.window(X.device), center=False,
                            length=padded_len)
            return y[..., self.n_fft:]
        return torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=self.window(X.device), length=length)
