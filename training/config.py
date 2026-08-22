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

# CONFIG_V3: fine-tuning de V1 sobre Common Voice ES.
# Misma loss, mismo lr, mismas épocas que V1 — la única variable que
# cambia respecto a V1 es el dataset (EN → ES) y el punto de partida
# (pesos de V1 en vez de random). Ver docs/decisions.md 19/08/2026.
CONFIG_V3 = {
    "train_dir": PROJECT_ROOT / "data" / "processed_es" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed_es" / "val",
    "variant": "V3",
    "loss": "mse_magnitude",       # misma loss que V1 (fidelidad al ablation V1→V3)
    "init_checkpoint": PROJECT_ROOT / "checkpoints" / "v1" / "best.pt",
    "n_epochs": 30,
    "batch_size": 4,
    "lr": 2e-4,
    "scheduler_step": 2,
    "scheduler_gamma": 0.98,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v3",
    "description": "Fine-tuning de V1 sobre Common Voice ES, misma loss (MSE) e hiperparámetros que V1.",
}
