# trainer.py
import torch
import torch.nn as nn
from training.losses import mse_magnitude, mse_plus_sisdr, get_loss_name
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
import json
import argparse
import sys

from datasets import NSDataset
from models.crn import CRN
from stft import STFTHelper


class Trainer:
    def __init__(self, config):
        """
        config: dict con todos los parámetros necesarios.
        """
        self.config = config

        # Fijar semilla para reproducibilidad
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        # Parámetros extraídos
        train_dir = config["train_dir"]
        val_dir = config["val_dir"]
        segment_samples = config.get("segment_samples", 64000)  # valor por defecto
        batch_size = config["batch_size"]
        train_shuffle = config.get("train_shuffle", True)
        val_shuffle = config.get("val_shuffle", False)
        lr = config["lr"]
        lr_decay_factor = config["scheduler_gamma"]
        lr_decay_period = config["scheduler_step"]
        self.n_epochs = config["n_epochs"]
        self.checkpoint_dir = Path(config["checkpoint_dir"])

        self.train_loader = DataLoader(
            NSDataset(train_dir, segment_samples=segment_samples),
            batch_size=batch_size, shuffle=train_shuffle, num_workers=2
        )
        self.val_loader = DataLoader(
            NSDataset(val_dir, segment_samples=segment_samples),
            batch_size=batch_size, shuffle=val_shuffle, num_workers=2
        )

        self.model = CRN().to(self.device)
        self.stft = STFTHelper(n_fft=320, hop_length=160)
        self.stft._window = torch.hamming_window(320).to(self.device)

        self.opt = Adam(self.model.parameters(), lr=lr, amsgrad=False)
        self.sched = StepLR(self.opt, step_size=lr_decay_period, gamma=lr_decay_factor)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss selection
        self.loss_name = get_loss_name(config)
        self.loss_alpha = config.get("loss_alpha", 0.7)
        self.sisdr_scale = config.get("sisdr_scale", 0.03)
        print(f"Loss: {self.loss_name}")
        if self.loss_name == "mse_plus_sisdr":
            print(f"  alpha (MSE weight): {self.loss_alpha}")
            print(f"  sisdr_scale:        {self.sisdr_scale}")

        self.history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
        self.best_val = float("inf")

        # Guardar la configuración usada en el directorio de checkpoints
        with open(self.checkpoint_dir / "config.json", "w") as f:
            # Convertir rutas a string para JSON
            config_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
            json.dump(config_serializable, f, indent=2)

    def _compute_loss(self, noisy, clean):
        """Calcula la pérdida MSE entre la magnitud estimada y la limpia."""
        mag_noisy, _ = self.stft.to_spec(noisy)
        mag_clean, _ = self.stft.to_spec(clean)
        mag_est = self.model(mag_noisy)
        return nn.functional.mse_loss(mag_est, mag_clean)

    def _compute_loss(self, noisy, clean, return_components=False):
        """ Calcula la pérdida según config. """
    
        """Args:noisy: [B, T] audio ruidoso
        clean: [B, T] audio limpio target
        return_components: si True, devuelve dict con desglose (solo mse_plus_sisdr)
        Returns:
        loss escalar (y opcionalmente components dict) """
    # Asegurar shape [B, T]
        if noisy.dim() == 3:
            noisy = noisy.squeeze(1)
        if clean.dim() == 3:
            clean = clean.squeeze(1)
    
        mag_noisy, phase_noisy = self.stft.to_spec(noisy)
        mag_clean, _ = self.stft.to_spec(clean)
        mag_est = self.model(mag_noisy)
    
        if self.loss_name == "mse_magnitude":
            loss = mse_magnitude(mag_est, mag_clean)
            if return_components:
                return loss, {"mse": loss.item()}
            return loss
    
        elif self.loss_name == "mse_plus_sisdr":
        # Reconstruir audio para SI-SDR
            audio_est = self.stft.from_spec(mag_est, phase_noisy, length=noisy.shape[-1])
            loss, components = mse_plus_sisdr(
                mag_est, mag_clean, audio_est, clean,
                alpha=self.loss_alpha,
                sisdr_scale=self.sisdr_scale,
            )
            if return_components:
                return loss, components
            return loss
    
        else:   
            raise ValueError(f"Loss desconocida: {self.loss_name}")


    def train_epoch(self):
        """Entrena una época completa."""
        self.model.train()
        total = 0.0
        total_mse = 0.0
        total_sisdr = 0.0
        n_batches = 0
    
        for noisy, clean in self.train_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            loss, components = self._compute_loss(noisy, clean, return_components=True)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
            total_mse += components.get("mse_component", components.get("mse", 0.0))
            total_sisdr += components.get("sisdr_component", 0.0)
            n_batches += 1
    
        return {
            "loss": total / n_batches,
            "mse": total_mse / n_batches,
            "sisdr": total_sisdr / n_batches,
        }

    @torch.no_grad()
    def validate(self):
        """Evalúa el modelo en el conjunto de validación."""
        self.model.eval()
        total = 0.0
        total_mse = 0.0
        total_sisdr = 0.0
        n_batches = 0
    
        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            loss, components = self._compute_loss(noisy, clean, return_components=True)
            total += loss.item()
            total_mse += components.get("mse_component", components.get("mse", 0.0))
            total_sisdr += components.get("sisdr_component", 0.0)
            n_batches += 1
    
        return {
            "loss": total / n_batches,
            "mse": total_mse / n_batches,
            "sisdr": total_sisdr / n_batches,
        }


    def fit(self):
        """Bucle principal de entrenamiento."""
        # Extender history con componentes
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_mse": [], "val_mse": [],
            "train_sisdr": [], "val_sisdr": [],
            "epoch_time_s": [],
        }
    
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr = self.train_epoch()
            va = self.validate()
            self.sched.step()
            elapsed = time.time() - t0
        
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(va["loss"])
            self.history["train_mse"].append(tr["mse"])
            self.history["val_mse"].append(va["mse"])
            self.history["train_sisdr"].append(tr["sisdr"])
            self.history["val_sisdr"].append(va["sisdr"])
            self.history["epoch_time_s"].append(elapsed)
        
            # Print adaptado según loss
            if self.loss_name == "mse_plus_sisdr":
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                    f"train={tr['loss']:.4f} (mse={tr['mse']:.4f}, sisdr={tr['sisdr']:+.2f}dB)  "
                    f"val={va['loss']:.4f} (mse={va['mse']:.4f}, sisdr={va['sisdr']:+.2f}dB)  "
                    f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                    f"train={tr['loss']:.4f}  val={va['loss']:.4f}  "
                    f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}")
        
            # Guardar el mejor modelo (por val_loss combinada)
            if va["loss"] < self.best_val:
                self.best_val = va["loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_state": self.opt.state_dict(),
                    "val_loss": va["loss"],
                    "val_mse": va["mse"],
                    "val_sisdr": va["sisdr"],
                    "loss_name": self.loss_name,
                }, self.checkpoint_dir / "best.pt")
        
            # Guardar historial en cada época
            with open(self.checkpoint_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)



if __name__ == "__main__":
    # Cambiamos a importación absoluta para evitar problemas con -m
    from .config import CONFIG_V1, CONFIG_V2

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="V1",
                        help="Nombre de la configuración a usar (V1, V2, ...)")
    args = parser.parse_args()

    configs = {
        "V1": CONFIG_V1,
        "V2": CONFIG_V2
    }
    if args.config not in configs:
        print(f"Configuración '{args.config}' no encontrada. Las disponibles: {list(configs.keys())}")
        sys.exit(1)

    config = configs[args.config]
    trainer = Trainer(config)
    trainer.fit()