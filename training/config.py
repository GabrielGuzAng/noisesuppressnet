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

# CONFIG_V3_SWEEP_BASE: base para el barrido de learning rate de V3b
# (docs/PLAN_V3B.md, en noisesuppressnet-ai). Self-contained a propósito
# (no depende de un _BASE compartido — ese patrón nunca se usó en este
# archivo, el comentario en CONFIG_V2 quedó como referencia muerta).
#
# Sin "scheduler_step"/"scheduler_gamma": Trainer trata su ausencia como
# lr fijo, sin decay (ver training/trainer.py) — necesario para que el
# sweep mida el efecto de un lr constante por corrida, no uno que decae.
# "lr" y "n_epochs" se pisan por corrida en scripts/lr_sweep_v3b.py.
CONFIG_V3_SWEEP_BASE = {
    "train_dir": PROJECT_ROOT / "data" / "processed_es" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed_es" / "val",
    "loss": "mse_magnitude",       # misma loss que V1 y V3 (aísla lr como variable)
    "init_checkpoint": PROJECT_ROOT / "checkpoints" / "v1" / "best.pt",
    "batch_size": 4,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,
    # Desactivado a propósito solo acá: corridas exploratorias descartables
    # (comparación relativa entre 5 lr, no el resultado final reportado).
    # Mide ~2x más rápido por época en este modelo (94% LSTM). El V3b final
    # (CONFIG_V3B, Etapa 4) NO define esta clave -> queda determinista por
    # default. Ver docs/decisions.md 22/08/2026.
    "cudnn_deterministic": False,
    "description": "Base config para sweep de lr de V3b",
}

# CONFIG_V3B: fine-tuning conservador de V1 sobre Common Voice ES.
# lr elegido empíricamente por scripts/lr_sweep_v3b.py + lr_sweep_v3b_evaluate.py
# (5 candidatos, ganador lr=5e-5, score +0.121 -- ver docs/decisions.md
# 22-23/08/2026 y results/v3_sweep/sweep_comparison.json). A diferencia
# del sweep, sin "cudnn_deterministic" -> True por default (esta sí es
# la corrida que se reporta/tagea).
CONFIG_V3B = {
    "train_dir": PROJECT_ROOT / "data" / "processed_es" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed_es" / "val",
    "variant": "V3b",
    "loss": "mse_magnitude",       # misma loss que V1/V3
    "init_checkpoint": PROJECT_ROOT / "checkpoints" / "v1" / "best.pt",
    "n_epochs": 10,
    "batch_size": 4,
    "lr": 5e-5,                    # ganador del sweep
    "scheduler_step": 4,
    "scheduler_gamma": 0.5,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v3b",
    "description": "Fine-tuning conservador de V1 sobre Common Voice ES, lr=5e-5 elegido por sweep empírico (10 épocas, StepLR step=4 gamma=0.5).",
}

# CONFIG_V3E: explota el hallazgo del sweep -- lr=1e-4 tuvo la mayor
# ganancia en español (+0.162 PESQ-NB a 5 épocas) sin haber convergido
# todavía. Se extiende a 25 épocas con decay tardío (época 12, no época 4
# como V3b) para darle más recorrido al lr alto antes de bajarlo.
# (V3c-b, el control para desconfundir épocas vs lr, quedó pospuesto por
# tiempo -- ver docs/decisions.md. V3e no depende de ese resultado, es una
# hipótesis distinta: explotar el punto del sweep con mayor ΔPESQ_ES.)
CONFIG_V3E = {
    **CONFIG_V3_SWEEP_BASE,
    "variant": "V3e",
    "lr": 1e-4,
    "n_epochs": 25,
    "scheduler_step": 12,             # decay tardío -- V3b decayó en época 4
                                       # y se estabilizó demasiado pronto
    "scheduler_gamma": 0.5,
    "cudnn_deterministic": True,      # esta SÍ es corrida reportable
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v3e",
    "description": "lr=1e-4: la corrida del sweep con mayor ΔPESQ_ES "
                   "(+0.162 a 5 épocas), sin haber convergido todavía. "
                   "Extendida a 25 épocas con decay tardío (época 12, no "
                   "época 4 como V3b) para darle más recorrido antes de "
                   "que el lr empiece a bajar. Hipótesis: puede igualar o "
                   "superar el score de V3 (+0.151) con menos forgetting "
                   "en inglés que V3 (-0.079).",
}
