  # datasets/dataset.py
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

SAMPLE_RATE = 16000

class NSDataset(Dataset):
    def __init__(self, pairs_dir, segment_samples=None):
        self.pairs = sorted(Path(pairs_dir).iterdir())
        self.segment_samples = segment_samples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        noisy, _ = torchaudio.load(str(pair / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair / "clean.wav"))
        noisy = noisy.squeeze(0)
        clean = clean.squeeze(0)

        if self.segment_samples:
            n = noisy.numel()
            if n > self.segment_samples:
                # Recorte aleatorio (audios más largos que el segmento)
                start = torch.randint(0, n - self.segment_samples, (1,)).item()
                noisy = noisy[start:start + self.segment_samples]
                clean = clean[start:start + self.segment_samples]
            elif n < self.segment_samples:
                # Pad con ceros al final (audios más cortos)
                pad = self.segment_samples - n
                noisy = torch.nn.functional.pad(noisy, (0, pad))
                clean = torch.nn.functional.pad(clean, (0, pad))
            # Si n == segment_samples, no hace falta hacer nada

        return noisy, clean
