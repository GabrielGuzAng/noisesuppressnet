# Changelog

## [v0.1.0] — 2026-06-30
### Added
- Modelo CRN funcional (models/crn.py, 17.58M params)
- STFT helper (stft.py, n_fft=320, hop=160, Hamming)
- DataLoader y mixtura on-the-fly (datasets/)
- Trainer con StepLR scheduler (training/trainer.py)
- Inferencia (inference/infer.py)
- Baseline Butterworth (baselines/butterworth.py)
- Análisis PSD comparativo (analysis/plot_psd.py)
- Suite de métricas: PESQ, STOI, SI-SDR (evaluation/metrics.py)
- Benchmark RTF (benchmarks/measure_rtf.py)
- Tests: causalidad bit-exact, reconstrucción STFT, pipeline integrado

### Validated
- Causalidad: diff frames pasados = 0.00e+00
- STFT reconstruction: error = 9.54e-07
- RTF p95 en CPU i5-4460: 0.34 (margen 3× sobre real-time)
- Val_loss monotónico decreciente en 10 épocas

### Known issues
- Modelo presenta output collapse — esperable con 200 pares × 10 épocas
- Referencias: Reddy 2021, Tan & Wang 2018, Xu 2022