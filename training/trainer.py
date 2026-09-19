# trainer.py
import numpy as np
import torch
import torch.nn as nn
from torchaudio.pipelines import SQUIM_OBJECTIVE
from training.losses import mse_magnitude, mse_plus_sisdr, mse_plus_squim, get_loss_name
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

        # Fijar semilla para reproducibilidad (restricción dura del proyecto:
        # seed 42 en TODAS las operaciones aleatorias)
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Default True: reproducibilidad bit-exact. Costo real medido:
            # ~2x tiempo/época en este modelo (94% LSTM, cuDNN determinista
            # es notablemente más lento en RNN). Para corridas exploratorias
            # descartables (ej. sweep de lr de V3b) se puede desactivar
            # explícitamente vía config — ver docs/decisions.md 22/08/2026.
            if config.get("cudnn_deterministic", True):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True  # shapes fijas (segment_samples constante) -> autotune ayuda

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
        lr_decay_factor = config.get("scheduler_gamma")
        lr_decay_period = config.get("scheduler_step")
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

        self.use_gate = config.get("gate", False)
        self.model = CRN(gate=self.use_gate).to(self.device)

        init_checkpoint = config.get("init_checkpoint")
        if init_checkpoint:
            init_checkpoint = Path(init_checkpoint)
            print(f"Cargando pesos iniciales desde: {init_checkpoint}")
            ckpt = torch.load(init_checkpoint, map_location=self.device, weights_only=False)
            if self.use_gate:
                # El checkpoint de partida no tiene la cabeza de compuerta. Se
                # permite que falte SOLO ella: cualquier otra clave ausente
                # significa que el checkpoint no corresponde a esta arquitectura
                # y tiene que romper acá, no 6 horas despues.
                missing, unexpected = self.model.load_state_dict(
                    ckpt["model_state"], strict=False)
                expected_missing = {"conv_gate.weight", "conv_gate.bias"}
                if set(missing) != expected_missing or unexpected:
                    raise RuntimeError(
                        f"Checkpoint incompatible. Faltan {sorted(missing)} "
                        f"(se esperaba {sorted(expected_missing)}), sobran {sorted(unexpected)}")
                print(f"  compuerta inicializada de cero (g0 = "
                      f"{torch.sigmoid(self.model.conv_gate.bias).item():.4f})")
            else:
                self.model.load_state_dict(ckpt["model_state"])

        self.stft = STFTHelper(n_fft=320, hop_length=160)
        self.stft._window = torch.hamming_window(320).to(self.device)

        # La cabeza de compuerta es nueva sobre un backbone convergido, asi que
        # lleva lr propio. Declarado en el preregistro como eleccion de diseno,
        # no como hiperparametro barrido.
        gate_lr = config.get("gate_lr")
        if self.use_gate and gate_lr is not None:
            gate_params = [p_ for n_, p_ in self.model.named_parameters()
                           if n_.startswith("conv_gate")]
            backbone = [p_ for n_, p_ in self.model.named_parameters()
                        if not n_.startswith("conv_gate")]
            self.opt = Adam([{"params": backbone, "lr": lr},
                             {"params": gate_params, "lr": gate_lr}], amsgrad=False)
            print(f"  lr backbone {lr:.1e} / lr compuerta {gate_lr:.1e}")
        else:
            self.opt = Adam(self.model.parameters(), lr=lr, amsgrad=False)
        # scheduler_step/scheduler_gamma ausentes en la config = lr fijo, sin decay
        # (usado por el sweep de V3b, que quiere medir el efecto de un lr constante)
        if lr_decay_factor is not None and lr_decay_period is not None:
            self.sched = StepLR(self.opt, step_size=lr_decay_period, gamma=lr_decay_factor)
        else:
            self.sched = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss selection
        self.loss_name = get_loss_name(config)
        self.loss_alpha = config.get("loss_alpha", 0.7)
        self.sisdr_scale = config.get("sisdr_scale", 0.03)
        self.squim_alpha = config.get("squim_alpha", 0.9)
        self.squim_scale = config.get("squim_scale")
        print(f"Loss: {self.loss_name}")
        if self.loss_name == "mse_plus_sisdr":
            print(f"  alpha (MSE weight): {self.loss_alpha}")
            print(f"  sisdr_scale:        {self.sisdr_scale}")
        if self.loss_name == "mse_plus_squim":
            print(f"  alpha (MSE weight): {self.squim_alpha}")
            print(f"  squim_scale:        {self.squim_scale}")
            # Squim frozen: eval() + requires_grad_(False) en todos los params,
            # cudnn desactivado solo alrededor del forward (ver _compute_loss)
            # -- solución (1) validada en tests/test_squim_differenciable.py.
            self.squim_model = SQUIM_OBJECTIVE.get_model().to(self.device)
            self.squim_model.eval()
            for p in self.squim_model.parameters():
                p.requires_grad_(False)

        self.save_every_n_epochs = config.get("save_every_n_epochs")

        self._reset_gate_stats()
        self.history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
        self.best_val = float("inf")

        # Reanudación de una corrida cortada. Va al FINAL de __init__ a propósito:
        # replica el hecho de que en la corrida original todo __init__ corrió antes
        # de la primera época, así que el estado del RNG global que se replica acá
        # es exactamente el que había al empezar la época 1.
        self.start_epoch = 1
        resume_from = config.get("resume_from")
        if resume_from:
            self._resume(Path(resume_from))

        # Guardar la configuración usada en el directorio de checkpoints. Al
        # reanudar NO se pisa la config original: es la evidencia de con qué se
        # lanzó la corrida, y el preregistro apunta a ella.
        config_name = ("config.json" if self.start_epoch == 1
                       else f"config_resume_ep{self.start_epoch:02d}.json")
        with open(self.checkpoint_dir / config_name, "w") as f:
            # Convertir rutas a string para JSON
            config_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
            json.dump(config_serializable, f, indent=2)

    def _resume(self, ckpt_path):
        """Resume an interrupted run from a periodic checkpoint, bit-exactly.

        Restores the model weights, the optimizer moments, the LR schedule and
        the global CPU RNG stream, so that the remaining epochs see the same
        data order they would have seen had the run never been interrupted.

        The RNG state is not stored in the checkpoint, so it is replayed rather
        than restored. ``RandomSampler.__iter__`` draws its per-epoch seed from
        the global CPU generator, and so does each DataLoader iterator for its
        worker base seed; nothing else in this training loop touches it (no
        dropout, and the random crop in ``NSDataset`` runs inside the workers,
        off that base seed). Rebuilding one train and one val iterator per
        completed epoch therefore advances the generator by exactly what those
        epochs consumed. It is done with the real loaders (~1 s for 14 epochs)
        so that no assumption about the cost of an epoch is hardcoded.

        The CUDA generator needs no replay: this model has no stochastic op on
        the device, so ``torch.cuda.manual_seed_all`` in ``__init__`` leaves it
        where the original run had it.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        done = ckpt["epoch"]
        if ckpt.get("gate", False) != self.use_gate:
            raise RuntimeError(
                f"El checkpoint tiene gate={ckpt.get('gate')} y la config pide "
                f"gate={self.use_gate}. No son la misma corrida.")
        if done >= self.n_epochs:
            raise RuntimeError(
                f"El checkpoint ya está en la época {done} de {self.n_epochs}: "
                f"no queda nada por reanudar.")

        self.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path) as f:
            history = json.load(f)
        if len(history["val_loss"]) < done:
            raise RuntimeError(
                f"{history_path} tiene {len(history['val_loss'])} épocas pero el "
                f"checkpoint dice {done}. Historial incompleto.")
        # history.json puede tener MÁS épocas que el checkpoint si el corte cayó
        # entre un guardado periódico y el siguiente: se trunca al checkpoint,
        # que es lo único de lo que se puede seguir.
        self.history = {k: v[:done] for k, v in history.items()}
        self.best_val = min(self.history["val_loss"])

        # El lr ya viene restaurado dentro de opt_state: StepLR es MULTIPLICATIVO
        # sobre el lr corriente (no lo recalcula desde initial_lr), así que
        # replicar los step() lo decaería una segunda vez. Lo único que falta
        # reponer es la fase del escalón, o sea last_epoch.
        if self.sched is not None:
            self.sched.last_epoch = done
            self.sched._step_count = done + 1
            self.sched._last_lr = [g["lr"] for g in self.opt.param_groups]

        rng_before = torch.random.get_rng_state()
        for _ in range(done):
            it = iter(self.train_loader); del it
            it = iter(self.val_loader); del it
        rng_after = torch.random.get_rng_state()

        self.start_epoch = done + 1
        print(f"Reanudando desde {ckpt_path} (época {done} completa).")
        print(f"  próxima época:  {self.start_epoch}/{self.n_epochs}")
        print(f"  val_loss mejor: {self.best_val:.6f}")
        print(f"  lr restaurado:  {self.opt.param_groups[0]['lr']:.2e}")
        print(f"  RNG replayado:  {'avanzado' if not torch.equal(rng_before, rng_after) else 'SIN CAMBIO (revisar)'}")

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
        if self.use_gate:
            mag_est, g = self.model(mag_noisy, return_gate=True)
            # Guarda de degeneracion (P4 del preregistro): si g colapsa a 0 la
            # red se apago, si colapsa a 1 la compuerta nunca aprendio. Se mide
            # epoca por epoca en vez de descubrirlo al evaluar.
            self._gate_sum += g.mean().item()
            self._gate_min = min(self._gate_min, g.min().item())
            self._gate_max = max(self._gate_max, g.max().item())
            self._gate_n += 1
        else:
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

        elif self.loss_name == "mse_plus_squim":
            # Reconstruir audio para Squim (mismo mecanismo que mse_plus_sisdr)
            audio_est = self.stft.from_spec(mag_est, phase_noisy, length=noisy.shape[-1])
            # cuDNN desactivado SOLO alrededor de este forward -- no debe
            # interferir con cudnn_deterministic/cudnn.benchmark que ya
            # configura el LSTM del propio CRN (ver docs/PLAN_V4.md Fase 2).
            with torch.backends.cudnn.flags(enabled=False):
                _, pesq_hat, _ = self.squim_model(audio_est)
            loss, components = mse_plus_squim(
                mag_est, mag_clean, pesq_hat,
                alpha=self.squim_alpha,
                squim_scale=self.squim_scale,
            )
            if return_components:
                return loss, components
            return loss

        else:
            raise ValueError(f"Loss desconocida: {self.loss_name}")


    def _reset_gate_stats(self):
        self._gate_sum, self._gate_n = 0.0, 0
        self._gate_min, self._gate_max = 1.0, 0.0

    def _gate_stats(self):
        if not self.use_gate or not self._gate_n:
            return None
        return {"mean": self._gate_sum / self._gate_n,
                "min": self._gate_min, "max": self._gate_max}

    def train_epoch(self):
        """Entrena una época completa."""
        self._reset_gate_stats()
        self.model.train()
        total = 0.0
        total_mse = 0.0
        total_sisdr = 0.0
        total_squim = 0.0
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
            total_squim += components.get("squim_component", 0.0)
            n_batches += 1

        return {
            "loss": total / n_batches,
            "mse": total_mse / n_batches,
            "sisdr": total_sisdr / n_batches,
            "squim": total_squim / n_batches,
        }

    @torch.no_grad()
    def validate(self):
        """Evalúa el modelo en el conjunto de validación."""
        self._reset_gate_stats()
        self.model.eval()
        total = 0.0
        total_mse = 0.0
        total_sisdr = 0.0
        total_squim = 0.0
        n_batches = 0

        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            loss, components = self._compute_loss(noisy, clean, return_components=True)
            total += loss.item()
            total_mse += components.get("mse_component", components.get("mse", 0.0))
            total_sisdr += components.get("sisdr_component", 0.0)
            total_squim += components.get("squim_component", 0.0)
            n_batches += 1

        return {
            "loss": total / n_batches,
            "mse": total_mse / n_batches,
            "sisdr": total_sisdr / n_batches,
            "squim": total_squim / n_batches,
            "gate": self._gate_stats(),
        }


    def fit(self):
        """Bucle principal de entrenamiento."""
        # Extender history con componentes. Al reanudar, _resume ya dejó el
        # historial de las épocas hechas: pisarlo las perdería.
        if self.start_epoch == 1:
            self.history = {
                "train_loss": [], "val_loss": [],
                "train_mse": [], "val_mse": [],
                "train_sisdr": [], "val_sisdr": [],
                "train_squim": [], "val_squim": [],
                "epoch_time_s": [],
            }

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            t0 = time.time()
            tr = self.train_epoch()
            va = self.validate()
            if self.sched is not None:
                self.sched.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(va["loss"])
            self.history["train_mse"].append(tr["mse"])
            self.history["val_mse"].append(va["mse"])
            self.history["train_sisdr"].append(tr["sisdr"])
            self.history["val_sisdr"].append(va["sisdr"])
            self.history["train_squim"].append(tr["squim"])
            self.history["val_squim"].append(va["squim"])
            self.history["epoch_time_s"].append(elapsed)
            if va.get("gate"):
                self.history.setdefault("val_gate", []).append(va["gate"])

            # Print adaptado según loss
            if self.loss_name == "mse_plus_sisdr":
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                    f"train={tr['loss']:.4f} (mse={tr['mse']:.4f}, sisdr={tr['sisdr']:+.2f}dB)  "
                    f"val={va['loss']:.4f} (mse={va['mse']:.4f}, sisdr={va['sisdr']:+.2f}dB)  "
                    f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}"
                    + (f"  g=[{va['gate']['min']:.3f} {va['gate']['mean']:.4f} "
                       f"{va['gate']['max']:.3f}]" if va.get("gate") else ""))
            elif self.loss_name == "mse_plus_squim":
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                    f"train={tr['loss']:.4f} (mse={tr['mse']:.4f}, squim={tr['squim']:+.4f})  "
                    f"val={va['loss']:.4f} (mse={va['mse']:.4f}, squim={va['squim']:+.4f})  "
                    f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                    f"train={tr['loss']:.4f}  val={va['loss']:.4f}  "
                    f"t={elapsed:.1f}s  lr={self.opt.param_groups[0]['lr']:.2e}")

            checkpoint_payload = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "opt_state": self.opt.state_dict(),
                "val_loss": va["loss"],
                "val_mse": va["mse"],
                "val_sisdr": va["sisdr"],
                "val_squim": va["squim"],
                "loss_name": self.loss_name,
                "gate": self.use_gate,
                "val_gate": va.get("gate"),
            }

            # Guardar el mejor modelo (por val_loss combinada)
            if va["loss"] < self.best_val:
                self.best_val = va["loss"]
                torch.save(checkpoint_payload, self.checkpoint_dir / "best.pt")

            # Checkpoints periódicos (necesario para monitor_correlation.py
            # y el chequeo cruzado de gaming de V4 -- ver docs/PLAN_V4.md
            # Fase 2/7. No se activa salvo que la config lo pida.
            if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
                torch.save(checkpoint_payload, self.checkpoint_dir / f"epoch_{epoch:02d}.pt")

            # Guardar historial en cada época
            with open(self.checkpoint_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)



if __name__ == "__main__":
    # Cambiamos a importación absoluta para evitar problemas con -m
    from .config import (CONFIG_V1, CONFIG_V2, CONFIG_V3, CONFIG_V3B, CONFIG_V3E,
                         CONFIG_V5, CONFIG_V5_S43, CONFIG_V5_S44, CONFIG_V5_SMOKE,
                         CONFIG_V5_SMOKE_LR5E5, CONFIG_V5_SMOKE_LR2E4,
                         CONFIG_V6_GATE, CONFIG_V6_PLACEBO, CONFIG_V6_SMOKE,
                         CONFIG_V7_GATE, CONFIG_V7_CONTROL,
                         CONFIG_V7_GATE_S43, CONFIG_V7_CONTROL_S43,
                         CONFIG_V7_GATE_S44, CONFIG_V7_CONTROL_S44)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="V1",
                        help="Nombre de la configuración a usar (V1, V2, ...)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint periódico desde el cual reanudar una "
                             "corrida cortada (ej. checkpoints/v7_control/epoch_14.pt). "
                             "No se toca la config: se reanuda la misma.")
    args = parser.parse_args()

    configs = {
        "V1": CONFIG_V1,
        "V2": CONFIG_V2,
        "V3": CONFIG_V3,
        "V3B": CONFIG_V3B,
        "V3E": CONFIG_V3E,
        "V5": CONFIG_V5,
        "V5S43": CONFIG_V5_S43,
        "V5S44": CONFIG_V5_S44,
        "V5SMOKE": CONFIG_V5_SMOKE,
        "V5SMOKE_LR5E5": CONFIG_V5_SMOKE_LR5E5,
        "V5SMOKE_LR2E4": CONFIG_V5_SMOKE_LR2E4,
        "V6GATE": CONFIG_V6_GATE,
        "V6PLACEBO": CONFIG_V6_PLACEBO,
        "V6SMOKE": CONFIG_V6_SMOKE,
        "V7GATE": CONFIG_V7_GATE,
        "V7CONTROL": CONFIG_V7_CONTROL,
        "V7GATE_S43": CONFIG_V7_GATE_S43,
        "V7CONTROL_S43": CONFIG_V7_CONTROL_S43,
        "V7GATE_S44": CONFIG_V7_GATE_S44,
        "V7CONTROL_S44": CONFIG_V7_CONTROL_S44,
    }
    if args.config not in configs:
        print(f"Configuración '{args.config}' no encontrada. Las disponibles: {list(configs.keys())}")
        sys.exit(1)

    config = configs[args.config]
    if args.resume_from:
        config = {**config, "resume_from": args.resume_from}
    trainer = Trainer(config)
    trainer.fit()