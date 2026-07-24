import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
import json

from datasets import NSDataset
from models.crn import CRN
from stft import STFTHelper


class Trainer:
    def __init__(self,
                 train_dir, val_dir,
                 checkpoint_dir,
                 segment_samples=64000,
                 batch_size=4,
                 lr=2e-4,
                 lr_decay_factor=0.98,
                 lr_decay_period=2,
                 n_epochs=10,
                 #n_epochs=3 ,
                 device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        self.train_loader = DataLoader(
            NSDataset(train_dir, segment_samples=segment_samples),
            batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            NSDataset(val_dir, segment_samples=segment_samples),
            batch_size=batch_size, shuffle=False, num_workers=2
        )

        self.model = CRN().to(self.device)
        self.stft = STFTHelper(n_fft=320, hop_length=160)

        # Hamming window al device
        self.stft._window = torch.hamming_window(320).to(self.device)

        self.opt = Adam(self.model.parameters(), lr=lr, amsgrad=False)
        self.sched = StepLR(self.opt, step_size=lr_decay_period, gamma=lr_decay_factor)
        self.n_epochs = n_epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
        self.best_val = float("inf")

    def _compute_loss(self, noisy, clean):
        mag_noisy, _ = self.stft.to_spec(noisy)
        mag_clean, _ = self.stft.to_spec(clean)
        mag_est = self.model(mag_noisy)
        return nn.functional.mse_loss(mag_est, mag_clean)

    def train_epoch(self):
        self.model.train()
        total = 0.0
        n_batches = 0
        for noisy, clean in self.train_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            loss = self._compute_loss(noisy, clean)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
            n_batches += 1
        return total / n_batches

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        n_batches = 0
        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            loss = self._compute_loss(noisy, clean)
            total += loss.item()
            n_batches += 1
        return total / n_batches

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr_loss = self.train_epoch()
            va_loss = self.validate()
            self.sched.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["epoch_time_s"].append(elapsed)

            print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}")

            # Guardar best
            if va_loss < self.best_val:
                self.best_val = va_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_state": self.opt.state_dict(),
                    "val_loss": va_loss,
                }, self.checkpoint_dir / "best.pt")

            # Guardar history a json en cada época
            with open(self.checkpoint_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    trainer = Trainer(
        train_dir=PROJECT_ROOT / "data" / "processed" / "train",
        val_dir=PROJECT_ROOT / "data" / "processed" / "val",
        checkpoint_dir=PROJECT_ROOT / "checkpoints" / "v0",
        n_epochs=10,
        batch_size=4,
    )
    trainer.fit()
