"""asr/transcribe.py

Transcribes a manifest of 16 kHz WAVs with the downstream ASR instrument and
writes the hypotheses to JSON. WER is computed elsewhere (`asr.wer`).

THIS MODULE DOES NOT RUN IN THE PROJECT ENVIRONMENT. It needs `.venv-asr`
(symlink to /mnt/Datos), pinned by `asr/requirements.txt`::

    .venv-asr/bin/python -m asr.transcribe            # smoke on data/asr_smoke

Instrument (preregistro downstream, §3; normative, do not tune):

- Primary: Whisper large-v3 through faster-whisper (CTranslate2), HF revision
  pinned below, fp16, language="es", beam_size=1, temperature=[0.0] with no
  fallback, condition_on_previous_text=False, vad_filter=False, and the retry
  thresholds (compression ratio, log-prob, no-speech) disabled by passing None.
  With those three at None, faster-whisper 1.2.1 neither falls back nor skips
  segments as silence (`generate_with_fallback` and the no-speech check in
  `generate_segments` are both guarded by `is not None`).
- Secondary: jonatasgrosman/wav2vec2-large-xlsr-53-spanish, CTC greedy (argmax),
  no language model. It only answers whether the direction of an effect belongs
  to the enhancement or to Whisper.

Every run transcribes the manifest TWICE per model, with a fresh model load per
pass. Pass 0 (the first in run order) governs; pass 1 exists only to measure how
many hypotheses change on GPU. It is never averaged, chosen or used to break ties.

Before the first transcription the instrument is written to
`results/asr_instrumento.json`; on later runs the current instrument is compared
against that file and the run aborts on any difference.
"""
import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
# torch se importa ANTES que faster_whisper a propósito: carga las libs de
# nvidia-cublas-cu12 / nvidia-cudnn-cu12 del venv en el proceso, que es de donde
# CTranslate2 las toma. Sin esto: "Library libcublas.so.12 is not found".
import torch
from faster_whisper import WhisperModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "asr_models"          # symlink a /mnt/Datos
INSTRUMENT_FILE = PROJECT_ROOT / "results" / "asr_instrumento.json"
SMOKE_MANIFEST = PROJECT_ROOT / "data" / "asr_smoke" / "manifest.json"
HYP_ROOT = PROJECT_ROOT / "data" / "asr_hyp"

SAMPLE_RATE = 16000
SEED = 42
N_PASSES = 2
GOVERNING_PASS = 0

WHISPER_REPO = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_DECODE = {
    "language": "es",
    "task": "transcribe",
    "beam_size": 1,
    "temperature": [0.0],
    "condition_on_previous_text": False,
    "vad_filter": False,
    "compression_ratio_threshold": None,
    "log_prob_threshold": None,
    "no_speech_threshold": None,
}

CTC_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
CTC_REVISION = "96d7e9b4e4a78af515a3c6d3cee7c0826045d276"
CTC_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin",
             "special_tokens_map.json", "vocab.json"]
# Decisión propia (la spec no fija precisión del secundario): fp32, batch 1 para
# que no haya padding que dependa de qué otras utterances caen en el batch.
CTC_DECODE = {"decoding": "greedy argmax, collapse CTC (processor.batch_decode)",
              "language_model": None, "dtype": "float32", "batch_size": 1}

NORMALIZER = {
    "class": "transformers.models.whisper.english_normalizer.BasicTextNormalizer",
    "kwargs": {"remove_diacritics": True, "split_letters": False},
    "applied_to": "referencia e hipótesis, idéntico",
}

logger = logging.getLogger("asr.transcribe")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(repo: str, revision: str) -> Path:
    """Local snapshot of a pinned HF revision (downloaded beforehand, never here)."""
    path = MODELS_DIR / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(
            f"Falta {repo}@{revision} en {MODELS_DIR}. Bajalo con "
            "huggingface_hub.snapshot_download(repo, revision=..., cache_dir=...)")
    return path


def model_files(path: Path, names: list[str] | None = None) -> list[dict]:
    """Hash every weight/config file of a snapshot, so the record pins bytes, not names."""
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name != "README.md")
    if names is not None:
        files = [p for p in files if p.name in names]
    return [{"file": p.name, "sha256": sha256_file(p.resolve())} for p in files]


def build_instrument_record() -> dict:
    """Everything that defines the instrument, as it is installed right now."""
    whisper_path = snapshot_dir(WHISPER_REPO, WHISPER_REVISION)
    ctc_path = snapshot_dir(CTC_REPO, CTC_REVISION)
    packages = ("faster-whisper", "ctranslate2", "jiwer", "transformers", "tokenizers",
                "torch", "numpy", "soundfile", "huggingface_hub", "av",
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12")
    return {
        "primary": {
            "model": "Whisper large-v3 (conversión CTranslate2 de openai/whisper-large-v3, "
                     "pesos guardados en FP16 por el conversor)",
            "hf_repo": WHISPER_REPO,
            "hf_revision": WHISPER_REVISION,
            "files": model_files(whisper_path),
            "runtime": "faster-whisper WhisperModel.transcribe (secuencial, no batched)",
            "device": "cuda",
            "compute_type": WHISPER_COMPUTE_TYPE,
            "decode_args": WHISPER_DECODE,
            "decode_args_note": "Los argumentos efectivos completos, defaults incluidos, "
                                "quedan en effective_transcription_options.",
        },
        "secondary": {
            "model": "wav2vec2-large-xlsr-53-spanish (CTC)",
            "hf_repo": CTC_REPO,
            "hf_revision": CTC_REVISION,
            "files": model_files(ctc_path, CTC_FILES),
            "training_data": "Common Voice 6.1 es, splits train+validation (model card)",
            "device": "cuda",
            **CTC_DECODE,
        },
        "normalizer": NORMALIZER,
        "wer": {
            "library": "jiwer",
            "aggregate": "sum(S+D+I) / sum(n_words de referencia), sobre utterances; "
                         "NO es la media de los WER por utterance",
            "per_utterance_fields": ["S", "D", "I", "H", "n_words"],
        },
        "passes": {
            "n_passes": N_PASSES,
            "governing_pass": GOVERNING_PASS,
            "run_order": "pasada 0 completa con modelo recién cargado, después pasada 1 "
                         "con otra carga; primario antes que secundario",
            "policy": "Rige la pasada 0. La pasada 1 sólo mide la fracción de hipótesis "
                      "que cambian: no se promedia, no se elige, no desempata.",
        },
        "input_audio": {"sample_rate": SAMPLE_RATE, "channels": 1,
                        "format": "WAV float32, leído con soundfile, sin resampleo"},
        "environment": {
            "python": platform.python_version(),
            "packages": {p: version(p) for p in packages},
            "cuda_runtime_torch": torch.version.cuda,
            "cudnn_torch": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
    }


def ensure_instrument_record(effective_options: dict) -> dict:
    """Write the instrument record once; afterwards, refuse to run a different one.

    Args:
        effective_options: `TranscriptionOptions` that faster-whisper actually used.

    Returns:
        The record on disk.
    """
    current = build_instrument_record()
    current["primary"]["effective_transcription_options"] = effective_options
    if INSTRUMENT_FILE.exists():
        on_disk = json.loads(INSTRUMENT_FILE.read_text())
        frozen = {k: v for k, v in on_disk.items() if k not in ("written_at",)}
        if json.loads(json.dumps(current)) != frozen:
            raise RuntimeError(
                f"El instrumento instalado no coincide con {INSTRUMENT_FILE}. "
                "No se transcribe con un instrumento distinto del registrado.")
        return on_disk
    record = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **current}
    INSTRUMENT_FILE.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    logger.info("Instrumento registrado en %s", INSTRUMENT_FILE)
    return record


def load_audio(path: Path) -> np.ndarray:
    """Read a mono 16 kHz WAV as float32; anything else is an error, not a resample."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: {sr} Hz, se esperaba {SAMPLE_RATE}")
    if wav.ndim != 1:
        raise ValueError(f"{path}: {wav.shape}, se esperaba mono")
    return wav


def whisper_options_probe(model: WhisperModel) -> dict:
    """Effective TranscriptionOptions for the spec's decode args (1 s of silence)."""
    _, info = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), **WHISPER_DECODE)
    opts = info.transcription_options
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(opts).items()}


def run_whisper_pass(utts: list[dict], audios: list[np.ndarray]) -> tuple[list[dict], dict]:
    """One full pass of the primary, with a fresh model load."""
    model = WhisperModel(str(snapshot_dir(WHISPER_REPO, WHISPER_REVISION)),
                         device="cuda", compute_type=WHISPER_COMPUTE_TYPE)
    options = whisper_options_probe(model)
    out = []
    t0 = time.perf_counter()
    for u, wav in zip(utts, audios):
        segments, _ = model.transcribe(wav, **WHISPER_DECODE)
        segments = list(segments)          # el generador es perezoso: acá decodifica
        out.append({
            "utt_id": u["utt_id"],
            "hypothesis": " ".join(s.text.strip() for s in segments),
            "segments": [{"start": s.start, "end": s.end, "text": s.text,
                          "avg_logprob": s.avg_logprob,
                          "compression_ratio": s.compression_ratio,
                          "no_speech_prob": s.no_speech_prob,
                          "temperature": s.temperature} for s in segments],
        })
    elapsed = time.perf_counter() - t0
    del model
    return out, {"elapsed_s": elapsed, "effective_transcription_options": options}


def run_ctc_pass(utts: list[dict], audios: list[np.ndarray]) -> tuple[list[dict], dict]:
    """One full pass of the secondary, with a fresh model load."""
    path = snapshot_dir(CTC_REPO, CTC_REVISION)
    processor = Wav2Vec2Processor.from_pretrained(str(path))
    model = Wav2Vec2ForCTC.from_pretrained(str(path)).to("cuda").eval()
    out = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for u, wav in zip(utts, audios):
            inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            logits = model(inputs.input_values.to("cuda")).logits
            ids = torch.argmax(logits, dim=-1)
            out.append({"utt_id": u["utt_id"],
                        "hypothesis": processor.batch_decode(ids)[0]})
    elapsed = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return out, {"elapsed_s": elapsed}


def transcribe_manifest(manifest_path: Path, out_dir: Path) -> Path:
    """Transcribe every utterance of a manifest: 2 passes x (primary, secondary).

    Args:
        manifest_path: JSON with `utterances`, each carrying `utt_id`, `wav`
            (relative to the project root), `reference` and `corpus`.
        out_dir: Where `hypotheses.json` lands.

    Returns:
        Path of the hypotheses file.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest = json.loads(manifest_path.read_text())
    utts = manifest["utterances"]
    audios = [load_audio(PROJECT_ROOT / u["wav"]) for u in utts]
    audio_s = sum(len(a) for a in audios) / SAMPLE_RATE
    logger.info("%d utterances, %.1f min de audio", len(utts), audio_s / 60)

    # El registro del instrumento se escribe/verifica ANTES de la primera transcripción.
    probe = WhisperModel(str(snapshot_dir(WHISPER_REPO, WHISPER_REVISION)),
                         device="cuda", compute_type=WHISPER_COMPUTE_TYPE)
    ensure_instrument_record(whisper_options_probe(probe))
    del probe

    runs = {"primary": [], "secondary": []}
    for model_name, runner in (("primary", run_whisper_pass), ("secondary", run_ctc_pass)):
        for pass_idx in range(N_PASSES):
            logger.info("%s, pasada %d", model_name, pass_idx)
            hyps, meta = runner(utts, audios)
            meta.pop("effective_transcription_options", None)
            meta["pass_index"] = pass_idx
            meta["sec_per_audio_min"] = meta["elapsed_s"] / (audio_s / 60)
            meta["peak_vram_torch_mib"] = torch.cuda.max_memory_allocated() / 2**20
            logger.info("  %.1f s (%.2f s por minuto de audio)",
                        meta["elapsed_s"], meta["sec_per_audio_min"])
            runs[model_name].append({**meta, "hypotheses": hyps})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hypotheses.json"
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "instrument": str(INSTRUMENT_FILE.relative_to(PROJECT_ROOT)),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "governing_pass": GOVERNING_PASS,
        "audio_total_s": audio_s,
        "utterances": [{k: u[k] for k in ("utt_id", "corpus", "reference", "duration_s")}
                       for u in utts],
        "runs": runs,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Hipótesis: %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream ASR instrument")
    parser.add_argument("--manifest", type=str, default=str(SMOKE_MANIFEST),
                        help="Manifiesto de utterances (default: smoke de voz limpia)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: data/asr_hyp/<nombre del directorio del manifiesto>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else HYP_ROOT / manifest.parent.name
    transcribe_manifest(manifest, out_dir)


if __name__ == "__main__":
    main()
