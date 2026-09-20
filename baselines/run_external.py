"""baselines/run_external.py

Runs the external baselines (RNNoise, DeepFilterNet2) over a sealed test set and
caches one enhanced WAV per pair, so that `evaluation.evaluate_variant` can
score them with the same machinery as the trained variants.

THIS SCRIPT DOES NOT RUN IN THE PROJECT ENVIRONMENT. It needs the separate venv
described in `docs/baselines_externos.md` (Python 3.11, numpy<2, torch CPU-only)
because DeepFilterNet 0.5.6 pins `numpy<2` and has no wheel for Python 3.12::

    .venv-baselines/bin/python -m baselines.run_external \
        --baseline deepfilternet2 --test_dir data/test_sealed/v1_en

Both models operate at 48 kHz and the project works at 16 kHz, so every pair
goes 16k -> 48k -> model -> 16k. That transit is not free, and `--baseline
resample_roundtrip` is the control that prices it: the identical resampling
chain with no model in between. Whether that control belongs in the reported
comparison is a protocol decision, not this script's.

The torch install in that venv is CPU-only (`+cpu` wheels), so nothing here can
touch the GPU even by accident.
"""
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / "baselines_external"

SAMPLE_RATE = 16000       # el del proyecto
EXTERNAL_SR = 48000       # el de RNNoise y DeepFilterNet2
SEED = 42

# Parámetros explícitos del resampler: entran al manifiesto como material
# reproducible en vez de quedar escondidos en los defaults de torchaudio.
RESAMPLE_KWARGS = {
    "resampling_method": "sinc_interp_hann",
    "lowpass_filter_width": 6,
    "rolloff": 0.99,
}

logger = logging.getLogger("run_external")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────── baselines ───────────────────────────

def build_rnnoise():
    """Load RNNoise (Valin 2018) through its ctypes bindings.

    The bundled `librnnoise.so` carries the trained weights compiled in, so the
    hash of that shared object is what identifies the model.

    Returns:
        A tuple (enhance_fn, model_info). `enhance_fn` maps a 1-D float32
        tensor at 48 kHz to another one of the same length.
    """
    from pyrnnoise import RNNoise
    from pyrnnoise.rnnoise import FRAME_SIZE, LIBRNNOISE

    def enhance_fn(wav48: torch.Tensor) -> torch.Tensor:
        # Estado nuevo por par: sin arrastre entre archivos del sellado.
        denoiser = RNNoise(sample_rate=EXTERNAL_SR)
        frames = [frame for _, frame in
                  denoiser.denoise_chunk(wav48.numpy(), partial=True)]
        out = np.concatenate(frames, axis=-1).reshape(-1)
        # La API de RNNoise trabaja en el rango de int16 y pyrnnoise devuelve
        # int16: la cuantización a 16 bits es parte del modelo, no del cache.
        return torch.from_numpy(out.astype(np.float32) / 32768.0)

    info = {
        "name": "RNNoise",
        "reference": "Valin 2018, A Hybrid DSP/Deep Learning Approach to "
                     "Real-Time Full-Band Speech Enhancement (MMSP)",
        "library": str(LIBRNNOISE),
        "library_sha256": sha256_file(Path(LIBRNNOISE)),
        "frame_size_samples": FRAME_SIZE,
        "frame_size_ms": FRAME_SIZE * 1000 / EXTERNAL_SR,
    }
    return enhance_fn, info


def build_deepfilternet2():
    """Load DeepFilterNet2 (Schröter et al. 2022) from the official weights.

    `init_df` downloads them once into ~/.cache/DeepFilterNet/DeepFilterNet2.
    Its log line says "Using DeepFilterNet3 model at .../DeepFilterNet2": that
    is DeepFilterNet printing its `default_model` argument, not the model it
    loaded. The line that matters is "Initializing model `deepfilternet2`".

    Returns:
        A tuple (enhance_fn, model_info), as in `build_rnnoise`.
    """
    from df.enhance import enhance, init_df

    model, df_state, model_name = init_df(model_base_dir="DeepFilterNet2",
                                          log_file=None)
    model.eval()

    if df_state.sr() != EXTERNAL_SR:
        raise RuntimeError(f"DeepFilterNet2 espera {df_state.sr()} Hz, "
                           f"no {EXTERNAL_SR}")

    def enhance_fn(wav48: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # pad=True compensa el lookahead del modelo y devuelve el audio
            # alineado con la entrada.
            return enhance(model, df_state, wav48.unsqueeze(0), pad=True).squeeze(0)

    base = Path.home() / ".cache" / "DeepFilterNet" / "DeepFilterNet2"
    ckpts = sorted(base.glob("checkpoints/*.ckpt*"))
    info = {
        "name": model_name,
        "reference": "Schröter et al. 2022, DeepFilterNet2 (IWAENC)",
        "params": sum(p.numel() for p in model.parameters()),
        "sr": df_state.sr(),
        "fft_size": df_state.fft_size(),
        "hop_size": df_state.hop_size(),
        "checkpoints": [{"file": str(c.relative_to(base)),
                         "sha256": sha256_file(c)} for c in ckpts],
        "config_sha256": sha256_file(base / "config.ini"),
    }
    return enhance_fn, info


def build_resample_roundtrip():
    """Control: the 16k->48k->16k transit with no model in between.

    Returns:
        A tuple (enhance_fn, model_info) where `enhance_fn` is the identity.
    """
    return (lambda wav48: wav48), {"name": "resample_roundtrip", "params": 0}


BUILDERS = {
    "rnnoise": build_rnnoise,
    "deepfilternet2": build_deepfilternet2,
    "resample_roundtrip": build_resample_roundtrip,
}


# ─────────────────────────── corrida ───────────────────────────

def run(baseline: str, test_dir: Path, metadata_path: Path, output_root: Path,
        threads: int) -> Path:
    """Enhance every pair of a sealed test set and cache the result.

    Args:
        baseline: One of `BUILDERS`.
        test_dir: Sealed test set directory (contains pair_XXXX/noisy.wav).
        metadata_path: Sealed metadata; its pair ids select what gets processed.
        output_root: Root of the cache; the audio lands in
            <output_root>/<baseline>/<test_dir.name>/.
        threads: torch CPU threads. Fixed on purpose: it lands in the manifest
            and CPU results move in the last bits with the thread count.

    Returns:
        Path of the written manifest.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    metadata = json.loads(metadata_path.read_text())
    pair_ids = sorted(p["id"] for p in metadata["pairs"])

    out_dir = output_root / baseline / test_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Baseline: %s", baseline)
    logger.info("Test set: %s (%d pares)", test_dir, len(pair_ids))
    logger.info("Salida:   %s", out_dir)
    logger.info("Hilos de torch: %d", threads)

    up = torchaudio.transforms.Resample(SAMPLE_RATE, EXTERNAL_SR, **RESAMPLE_KWARGS)
    down = torchaudio.transforms.Resample(EXTERNAL_SR, SAMPLE_RATE, **RESAMPLE_KWARGS)

    enhance_fn, model_info = BUILDERS[baseline]()

    pairs_manifest = []
    for pair_id in pair_ids:
        in_path = test_dir / f"pair_{pair_id:04d}" / "noisy.wav"
        noisy, sr = torchaudio.load(str(in_path))
        if sr != SAMPLE_RATE:
            raise ValueError(f"{in_path}: sample rate {sr}")
        noisy = noisy.squeeze(0)

        t0 = time.perf_counter()
        wav48 = up(noisy)
        enhanced48 = enhance_fn(wav48)
        if enhanced48.numel() != wav48.numel():
            logger.warning("pair_%04d: %s devolvió %d muestras a 48 kHz contra "
                           "%d de entrada; se recorta/rellena al largo original",
                           pair_id, baseline, enhanced48.numel(), wav48.numel())
            enhanced48 = torch.nn.functional.pad(
                enhanced48[:wav48.numel()],
                (0, max(0, wav48.numel() - enhanced48.numel())))
        enhanced = down(enhanced48)
        elapsed = time.perf_counter() - t0

        if enhanced.numel() != noisy.numel():
            raise RuntimeError(f"pair_{pair_id:04d}: {enhanced.numel()} muestras "
                               f"contra {noisy.numel()} de entrada")

        out_path = out_dir / f"pair_{pair_id:04d}.wav"
        # float32: el cache no agrega una cuantización que las variantes
        # entrenadas no sufren.
        sf.write(str(out_path), enhanced.numpy(), SAMPLE_RATE, subtype="FLOAT")

        pairs_manifest.append({
            "pair_id": pair_id,
            "n_samples": int(noisy.numel()),
            "duration_s": noisy.numel() / SAMPLE_RATE,
            "elapsed_s": elapsed,
            "rtf": elapsed / (noisy.numel() / SAMPLE_RATE),
            "input_sha256": sha256_file(in_path),
            "output_sha256": sha256_file(out_path),
        })
        if len(pairs_manifest) % 25 == 0:
            logger.info("  %d/%d procesados", len(pairs_manifest), len(pair_ids))

    rtfs = np.array([p["rtf"] for p in pairs_manifest])
    manifest = {
        "baseline": baseline,
        "test_set": test_dir.name,
        "test_dir": str(test_dir),
        "metadata": str(metadata_path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_pairs": len(pairs_manifest),
        "sample_rate": SAMPLE_RATE,
        "external_sample_rate": EXTERNAL_SR,
        "resampler": {"implementation": "torchaudio.transforms.Resample",
                      **RESAMPLE_KWARGS},
        "output_format": "WAV float32 (soundfile subtype FLOAT), sin clipping "
                         "ni normalización",
        "torch_threads": threads,
        "seed": SEED,
        "environment": {
            "python": f"{__import__('sys').version_info.major}."
                      f"{__import__('sys').version_info.minor}."
                      f"{__import__('sys').version_info.micro}",
            "packages": {p: version(p) for p in
                         ("torch", "torchaudio", "numpy", "soundfile",
                          "deepfilternet", "pyrnnoise", "deepfilterlib", "av")},
        },
        "model": model_info,
        # RTF de la cadena completa (resample + modelo + resample) en CPU, con
        # los hilos declarados arriba. No es comparable sin más a los 0,34 p95
        # de benchmarks/measure_rtf.py: esa medición es de otra cadena.
        "rtf_chain_mean": float(rtfs.mean()),
        "rtf_chain_p95": float(np.percentile(rtfs, 95)),
        "pairs": pairs_manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("RTF de la cadena en CPU: media %.3f, p95 %.3f",
                manifest["rtf_chain_mean"], manifest["rtf_chain_p95"])
    logger.info("Manifiesto: %s", manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--baseline", required=True, choices=sorted(BUILDERS),
                        help="Baseline externo a correr")
    parser.add_argument("--test_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "test_sealed" / "v1_en"),
                        help="Directorio del test set sellado")
    parser.add_argument("--metadata", type=str,
                        default=str(PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"),
                        help="Metadata del sellado: sus ids fijan qué pares se procesan")
    parser.add_argument("--output_root", type=str, default=str(CACHE_ROOT),
                        help="Raíz del cache de audio")
    parser.add_argument("--threads", type=int, default=2,
                        help="Hilos de CPU para torch (default 2, para no "
                             "comerse la máquina si hay un entrenamiento corriendo)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    run(baseline=args.baseline,
        test_dir=Path(args.test_dir).resolve(),
        metadata_path=Path(args.metadata).resolve(),
        output_root=Path(args.output_root).resolve(),
        threads=args.threads)


if __name__ == "__main__":
    main()
