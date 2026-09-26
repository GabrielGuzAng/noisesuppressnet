"""Downstream ASR instrument (WER) for NoiseSuppressNet.

Two environments, same split as `baselines/`:

- `asr.prepare_smoke` runs in the project venv (`.venv`): it decodes clean
  speech with the project's own loaders and caches 16 kHz WAVs plus references.
- `asr.transcribe` and `asr.wer` run in `.venv-asr` (faster-whisper,
  CTranslate2, transformers, jiwer), which never touches `.venv`.
"""
