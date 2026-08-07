# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_V1 = {
    "train_dir": PROJECT_ROOT / "data" / "processed" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed" / "val",
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v1",
    "n_epochs": 30,
    "batch_size": 4,
    "lr": 2e-4,
    "scheduler_step": 2,
    "scheduler_gamma": 0.98,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,
    # Otros parámetros que quieras añadir en el futuro
}

# Configuración V2, V3, etc.
CONFIG_V2 = {
    #**_BASE,  # hereda TODO de V1 (misma seed, lr, batch, etc.)
    "train_dir": PROJECT_ROOT / "data" / "processed" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed" / "val",
    "variant": "V2",
    "loss": "mse_plus_sisdr",
    "loss_alpha": 0.7,        # 70% MSE, 30% SI-SDR
    "n_epochs": 20,
    "batch_size": 4,   
    "lr": 2e-4,
    "scheduler_step": 2,
    "scheduler_gamma": 0.98,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,        # ← menor que V1 porque ya sabemos convergencia ~19
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v2",
    "description": "V1 + loss combinada MSE (magnitud) + SI-SDR (tiempo), alpha=0.7",
}
