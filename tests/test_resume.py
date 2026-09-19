"""
tests/test_resume.py

Verifica que reanudar una corrida cortada produce EXACTAMENTE los mismos
checkpoints que si nunca se hubiera cortado.

Por qué hace falta. El 16/09/2026 un corte de energía mató V7control en la
época 15 de 20, y las épocas perdidas eran justamente las que cargan el
endpoint primario preregistrado (el contraste promediado sobre 15-20). La
alternativa a reanudar era reentrenar 10,5 h desde cero, y la razón para
reentrenar era que una reanudación "casi igual" mete una perturbación de punto
flotante del mismo tamaño que el efecto buscado — el mismo argumento que ya
había obligado a reentrenar el control de V7 en vez de reusar V2 (ver el
comentario de `_CONFIG_V7_BASE`). O sea: si la reanudación no es bit-exacta, no
sirve. Esto lo verifica en vez de asumirlo.

Las tres piezas que hay que reponer:

1. Pesos y momentos de Adam — vienen en el checkpoint periódico.
2. La fase del StepLR — el lr YA viene restaurado dentro de `opt_state`, y
   StepLR es multiplicativo sobre el lr corriente (no lo recalcula desde
   `initial_lr`), así que replicar los `step()` lo decaería una segunda vez.
   Sólo hay que reponer `last_epoch`. Este test agarró ese bug.
3. El RNG global de CPU — no se guarda, se replica. `RandomSampler.__iter__`
   saca su semilla de época de ahí, y cada iterador de DataLoader saca de ahí
   su `base_seed` de workers (del que salen, a su vez, los recortes aleatorios
   de `NSDataset`). Reconstruir un iterador de train y uno de val por época ya
   hecha avanza el generador exactamente lo que esas épocas consumieron.

Cada entrenamiento corre en un proceso APARTE. Reanudar después de un corte de
energía es necesariamente un proceso nuevo, y probarlo dentro del mismo
intérprete dejaría sin verificar justamente lo que podría no reproducirse entre
procesos: la elección de kernels de cuDNN y la alineación de memoria en la GPU.

`epoch_time_s` es lo único que legítimamente difiere: es reloj de pared.

Uso:
    python -m pytest tests/test_resume.py -v
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from training.config import PROJECT_ROOT

PAIRS_TRAIN = 40
PAIRS_VAL = 16
N_EPOCHS = 4
CUT_AT = 2

_RUNNER_SOURCE = "\n".join([
    "import json, sys",
    "from pathlib import Path",
    "from training.trainer import Trainer",
    "config = json.loads(sys.argv[1])",
    "for key in ('train_dir', 'val_dir', 'checkpoint_dir'):",
    "    config[key] = Path(config[key])",
    "Trainer(config).fit()",
])


def _launch(config, runner):
    """Corre un entrenamiento en su propio intérprete y devuelve el resultado."""
    payload = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    # El runner vive en tmp, así que sys.path[0] no es el repo: hay que decirle
    # dónde están los paquetes del proyecto.
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(
        [sys.executable, str(runner), json.dumps(payload)],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    path = tmp_path_factory.mktemp("runner") / "run_trainer.py"
    path.write_text(_RUNNER_SOURCE)
    return path


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    """Dataset diminuto de pares reales, por symlink: épocas de ~2 s."""
    source = PROJECT_ROOT / "data" / "processed" / "val"
    if not source.is_dir():
        pytest.skip(f"No está {source}; el test necesita pares reales.")
    pairs = sorted(source.iterdir())[: PAIRS_TRAIN + PAIRS_VAL]
    if len(pairs) < PAIRS_TRAIN + PAIRS_VAL:
        pytest.skip(f"{source} tiene {len(pairs)} pares, hacen falta "
                    f"{PAIRS_TRAIN + PAIRS_VAL}.")

    root = tmp_path_factory.mktemp("mini")
    for name, chunk in (("train", pairs[:PAIRS_TRAIN]), ("val", pairs[PAIRS_TRAIN:])):
        (root / name).mkdir()
        for pair in chunk:
            (root / name / pair.name).symlink_to(pair.resolve(), target_is_directory=True)
    return root


def _config(mini_dataset, checkpoint_dir, n_epochs, resume_from=None):
    config = {
        "train_dir": mini_dataset / "train",
        "val_dir": mini_dataset / "val",
        "loss": "mse_plus_sisdr",
        "loss_alpha": 0.7,
        "sisdr_scale": 0.03,
        "n_epochs": n_epochs,
        "batch_size": 4,
        "lr": 2e-4,
        "scheduler_step": 2,
        "scheduler_gamma": 0.98,
        "seed": 42,
        "train_shuffle": True,
        "val_shuffle": False,
        "cudnn_deterministic": True,
        "save_every_n_epochs": 1,
        "gate": False,
        "checkpoint_dir": checkpoint_dir,
        "variant": checkpoint_dir.name,
    }
    if resume_from is not None:
        config["resume_from"] = str(resume_from)
    return config


@pytest.fixture(scope="module")
def runs(mini_dataset, runner, tmp_path_factory):
    """Corre la continua y la cortada+reanudada, cada una en su propio proceso."""
    root = tmp_path_factory.mktemp("ckpt")
    continua, cortada = root / "continua", root / "cortada"

    for config in (_config(mini_dataset, continua, N_EPOCHS),
                   _config(mini_dataset, cortada, CUT_AT),
                   _config(mini_dataset, cortada, N_EPOCHS,
                           resume_from=cortada / f"epoch_{CUT_AT:02d}.pt")):
        result = _launch(config, runner)
        assert result.returncode == 0, (
            f"el entrenamiento falló:\n{result.stdout}\n{result.stderr}")
    return continua, cortada


@pytest.mark.parametrize(
    "name", [f"epoch_{e:02d}.pt" for e in range(CUT_AT + 1, N_EPOCHS + 1)] + ["best.pt"])
def test_checkpoints_bit_identicos(runs, name):
    """Cada checkpoint posterior al corte es bit-idéntico al de la continua."""
    continua, cortada = runs
    a = torch.load(continua / name, map_location="cpu", weights_only=False)
    b = torch.load(cortada / name, map_location="cpu", weights_only=False)

    assert a["epoch"] == b["epoch"]
    assert a["val_loss"] == b["val_loss"], (
        f"{name}: val_loss {a['val_loss']!r} vs {b['val_loss']!r}")
    assert a["model_state"].keys() == b["model_state"].keys()
    for key, tensor in a["model_state"].items():
        assert torch.equal(tensor, b["model_state"][key]), f"{name}: difiere {key}"


@pytest.mark.parametrize("metric", ["train_loss", "val_loss", "train_mse", "val_mse",
                                    "train_sisdr", "val_sisdr"])
def test_historial_identico(runs, metric):
    """El historial reanudado conserva las épocas previas y continúa igual."""
    continua, cortada = runs
    a = json.loads((continua / "history.json").read_text())
    b = json.loads((cortada / "history.json").read_text())
    assert len(b[metric]) == N_EPOCHS, "la reanudación pisó el historial previo"
    assert a[metric] == b[metric]


def test_config_original_no_se_pisa(runs):
    """La config de lanzamiento es evidencia: la reanudación escribe aparte."""
    _, cortada = runs
    original = json.loads((cortada / "config.json").read_text())
    assert original["n_epochs"] == CUT_AT
    assert "resume_from" not in original
    reanudada = json.loads((cortada / f"config_resume_ep{CUT_AT + 1:02d}.json").read_text())
    assert reanudada["resume_from"].endswith(f"epoch_{CUT_AT:02d}.pt")


def test_rechaza_arquitectura_distinta(mini_dataset, runs, runner, tmp_path):
    """Reanudar un checkpoint sin compuerta con gate=True tiene que romper ya."""
    _, cortada = runs
    config = _config(mini_dataset, tmp_path / "gate", N_EPOCHS,
                     resume_from=cortada / f"epoch_{CUT_AT:02d}.pt")
    config["gate"] = True
    result = _launch(config, runner)
    assert result.returncode != 0, "tendría que haber roto y no rompió"
    assert "No son la misma corrida" in result.stderr, result.stderr


def test_rechaza_corrida_ya_terminada(mini_dataset, runs, runner, tmp_path):
    """Reanudar en la última época no tiene nada que reanudar."""
    continua, _ = runs
    config = _config(mini_dataset, tmp_path / "terminada", CUT_AT,
                     resume_from=continua / f"epoch_{CUT_AT:02d}.pt")
    result = _launch(config, runner)
    assert result.returncode != 0, "tendría que haber roto y no rompió"
    assert "no queda nada por reanudar" in result.stderr, result.stderr
