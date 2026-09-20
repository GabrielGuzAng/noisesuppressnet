"""baselines/external.py

Reader for the audio cache of the external baselines (RNNoise, DeepFilterNet2).

Neither model runs inside the project environment. They live in a separate venv
(`.venv-baselines`, Python 3.11) because DeepFilterNet 0.5.6 requires
`numpy<2` and ships no wheel for Python 3.12, while this project runs on
Python 3.12 with numpy 2.4.4. Installing it here would downgrade numpy under
torch, numba and librosa.

So enhancement is a previous step: `baselines/run_external.py`, run with that
other interpreter, writes one float32 WAV at 16 kHz per pair under
`data/baselines_external/<baseline>/<test_set>/`. This module only reads them
back, which lets `evaluation.evaluate_variant` score external audio through the
exact same machinery as the trained variants -- same SNR buckets, same noise
categories, same JSON schema -- without importing a single external dependency.

The cached audio is the model output at the project's sample rate, with no
post-processing: no gain, no clipping, no alignment. What the loop scores is
what came out of the model.
"""
from pathlib import Path

import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / "baselines_external"
SAMPLE_RATE = 16000


def cache_dir(name: str, test_set: str) -> Path:
    """Directory holding the cached waveforms of `name` over `test_set`.

    Args:
        name: Baseline name, as registered in `evaluation.evaluate_variant`.
        test_set: Directory name of the sealed test set (e.g. "v1_en").

    Returns:
        The directory path. It is not created and may not exist.
    """
    return CACHE_ROOT / name / test_set


def cached_baseline(name: str):
    """Build the callable that `evaluate_variant` uses for an external baseline.

    Args:
        name: Baseline name; also the cache subdirectory.

    Returns:
        A callable `(wav, *, pair_id, test_set, causal=False) -> Tensor` that
        returns the cached enhanced waveform for that pair. `wav` is only used
        to check that the cached file has the expected length: the enhancement
        itself happened offline.
    """
    def apply(wav, *, pair_id, test_set, causal=False):
        if causal:
            raise ValueError(
                f"--baseline_causal no aplica a '{name}': es una opción del "
                "filtro Butterworth. RNNoise y DeepFilterNet2 traen su propia "
                "latencia algorítmica fija, documentada en "
                "docs/baselines_externos.md."
            )

        path = cache_dir(name, test_set) / f"pair_{pair_id:04d}.wav"
        if not path.exists():
            raise FileNotFoundError(
                f"Falta el audio de '{name}' para pair_{pair_id:04d} de "
                f"'{test_set}': {path}\n"
                "Generalo primero con el venv aparte:\n"
                f"  .venv-baselines/bin/python -m baselines.run_external "
                f"--baseline {name} --test_dir data/test_sealed/{test_set}"
            )

        enhanced, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate {sr}, se esperaba {SAMPLE_RATE}")
        if enhanced.shape[0] != 1:
            raise ValueError(f"{path}: {enhanced.shape[0]} canales, se esperaba mono")
        if enhanced.shape[-1] != wav.shape[-1]:
            raise ValueError(
                f"{path}: {enhanced.shape[-1]} muestras contra {wav.shape[-1]} "
                "del noisy sellado. El cache no corresponde a este test set."
            )
        return enhanced.squeeze(0)

    apply.__name__ = f"cached_{name}"
    apply.__doc__ = f"Cached output of the external baseline '{name}'."
    return apply
