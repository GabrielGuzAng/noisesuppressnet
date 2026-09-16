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


# ─────────────────────────────────────────────────────────────────────────────
# V5 — propuesta completa. NO es una celda del ablation.
#
# La cadena V1→V2→V3e está cerrada y es la que da la atribución causal (una
# variable por vez). V5 combina lo que funcionó y su rol es ser el mejor sistema,
# no aislar un efecto. Hay que enunciarlo así en el informe o un jurado va a
# objetar, con razón, que cambian dos cosas a la vez.
#
# El anteproyecto define V5 como "español + PESQNet". PESQNet/Squim se descartó
# con evidencia propia (V4 y V4b: el proxy fijo se gamea y dos tercios del daño
# sobreviven al protocolo estabilizado). Era el riesgo R01 de la matriz, el más
# alto, y su contingencia declarada era caer a la loss combinada de V2. Eso es
# exactamente lo que hace V5.
#
# Dos cambios respecto de V3e:
#
# 1. Arranca de V2, no de V1 — y por lo tanto conserva la loss combinada
#    MSE + SI-SDR. Van juntos: fine-tunear con MSE puro desde un checkpoint que
#    ganó +0.20 PESQ gracias a la loss combinada desharía la mitad de esa
#    ganancia. V3/V3b/V3e partieron de V1 con MSE puro por disciplina de
#    ablation; la línea española nunca heredó la mejora de V2.
#
# 2. Entrena sobre data/processed_es_wide, con SNR ~ uniform(-5, 20) en vez de
#    uniform(-5, 15). El bucket [15,20] de los test sellados quedaba entero
#    fuera de distribución, y era justo donde vive el hallazgo principal.
#    Meterlo en distribución elimina la objeción de extrapolación
#    (Braun & Tashev 2020). Puede ATENUAR el efecto observado: es un test de
#    robustez, no un amplificador, y así hay que reportarlo.
#
# Lo que NO cambia, y se evaluó: `center=False`. Se midió que el lookahead sigue
# n_fft − hop con cualquier framing — es latencia del overlap-add, no del
# padding de la STFT. Ver tests/test_causality.py.
#
# Receta de optimización: la de V3e (lr 1e-4, 25 épocas, decay tardío en la 12),
# que dio el mejor balance de las tres iteraciones de fine-tuning.
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_V5_BASE = {
    "train_dir": PROJECT_ROOT / "data" / "processed_es_wide" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed_es_wide" / "val",
    "loss": "mse_plus_sisdr",      # la de V2, coherente con partir de su checkpoint
    "loss_alpha": 0.7,             # 70% MSE, 30% SI-SDR — idéntico a V2
    "sisdr_scale": 0.03,
    "init_checkpoint": PROJECT_ROOT / "checkpoints" / "v2" / "best.pt",
    "n_epochs": 25,
    "batch_size": 4,
    "lr": 1e-4,
    "scheduler_step": 12,
    "scheduler_gamma": 0.5,
    "train_shuffle": True,
    "val_shuffle": False,
    "cudnn_deterministic": True,   # corrida reportable
}

CONFIG_V5 = {
    **_CONFIG_V5_BASE,
    "variant": "V5",
    "seed": 42,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5",
    "description": "Propuesta completa: V2 (MSE+SI-SDR) fine-tuneado a español "
                   "sobre mixturas de rango SNR ancho, receta de V3e.",
}

# Semillas 43 y 44: miden la varianza del FINE-TUNING (orden de los datos), no
# la de inicialización — las tres parten del mismo checkpoint de V2 y el CRN no
# tiene dropout. Es la varianza relevante para la afirmación sobre adaptación al
# español, y hay que enunciarla con esa precisión: no habilita decir "robusto a
# la semilla" a secas.
# save_every_n_epochs=3: V5 solo guardó best.pt y quedó sin poder chequear si la
# época de mínima val_loss es también la mejor por PESQ. En V4b se vio que NO
# tienen por qué coincidir (el placebo en la época 2 superaba a V1 en las cuatro
# métricas pese a peor val_loss). Cada 3 épocas da 8 checkpoints por corrida
# (1.6 GB) y cubre la meseta donde cae el mínimo, sin acopiar 5 GB por corrida.
CONFIG_V5_S43 = {
    **_CONFIG_V5_BASE,
    "variant": "V5s43",
    "seed": 43,
    "save_every_n_epochs": 3,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5_s43",
    "description": "V5 con seed 43 — réplica para medir varianza de fine-tuning.",
}

CONFIG_V5_S44 = {
    **_CONFIG_V5_BASE,
    "variant": "V5s44",
    "seed": 44,
    "save_every_n_epochs": 3,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5_s44",
    "description": "V5 con seed 44 — réplica para medir varianza de fine-tuning.",
}

# Smoke de 3 épocas antes de comprometer las 25. Descartable, así que sin
# determinism (misma convención que CONFIG_V3_SWEEP_BASE). save_every_n_epochs=1
# para poder mirar época por época, lección de V4.
CONFIG_V5_SMOKE = {
    **_CONFIG_V5_BASE,
    "variant": "V5smoke",
    "seed": 42,
    "n_epochs": 3,
    "cudnn_deterministic": False,
    "save_every_n_epochs": 1,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5_smoke",
    "description": "Smoke de 3 épocas: confirma que arrancar de V2 sobre el "
                   "dataset de rango ancho no degrada antes de comprometer 15 h.",
}

# Screening de lr para V5, 3 épocas cada uno. NO es un sweep de selección: es un
# descarte. La lección de V3b es que en este modelo un sweep corto elige por
# convergencia rápida y se equivoca -- su ganador (5e-5) terminó peor que el
# candidato que a 5 épocas todavía no había convergido (1e-4, que fue V3e).
#
# Criterio declarado ANTES de correr: se descarta el lr que DEGRADE respecto del
# punto de partida (V2) o diverja. Si los tres pasan, se usa 1e-4 por la
# evidencia de las corridas convergidas (V3 2e-4 / V3e 1e-4 / V3b 5e-5: el lr
# gobierna el trade-off adaptación-olvido y 1e-4 es el punto de equilibrio),
# NO por cuál queda más alto a 3 épocas.
CONFIG_V5_SMOKE_LR5E5 = {
    **CONFIG_V5_SMOKE,
    "variant": "V5smoke_lr5e5",
    "lr": 5e-5,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5_smoke_lr5e5",
    "description": "Screening de lr para V5: 5e-5 (el conservador, lr de V3b).",
}

CONFIG_V5_SMOKE_LR2E4 = {
    **CONFIG_V5_SMOKE,
    "variant": "V5smoke_lr2e4",
    "lr": 2e-4,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v5_smoke_lr2e4",
    "description": "Screening de lr para V5: 2e-4 (el agresivo, lr de V1/V2/V3).",
}
