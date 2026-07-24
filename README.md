# NoiseSuppressNet

Sistema de supresión de ruido en tiempo real para español basado en una red
convolucional-recurrente (CRN). Proyecto Final de Ingeniería Electrónica,
UTN FRBA.

## Estado

**Versión actual:** v0.1.0 (baseline funcional)
**Próximo hito:** V1 sobre dataset escalado (Julio 2026)

Ver [CHANGELOG.md](CHANGELOG.md) y [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Instalación

\`\`\`bash
git clone https://github.com/GabrielGuzAng/nosiesuppressnet.git
cd nosiesuppressnet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Uso

Entrenar V0 (200 pares, 10 épocas, ~90s en RTX 4060):
\`\`\`bash
python -m training.trainer
\`\`\`

Inferir sobre val set:
\`\`\`bash
python -m inference.infer
\`\`\`

Evaluar métricas:
\`\`\`bash
python -m evaluation.metrics
\`\`\`

## Estructura

- `models/`: arquitectura CRN
- `datasets/`: DataLoader y mixtura
- `training/`: trainer
- `evaluation/`: métricas PESQ/STOI/SI-SDR
- `benchmarks/`: RTF, latencia
- `analysis/`: análisis PSD y espectrogramas
- `baselines/`: filtro Butterworth
- `inference/`: pipeline de inferencia
- `tests/`: tests unitarios
- `docs/`: decisiones técnicas y experimentos

## Referencias

Tan, K. & Wang, D.L. (2018). *A Convolutional Recurrent Neural Network for
Real-Time Speech Enhancement*. Interspeech 2018.

## Autor

Gabriel Guzmán — UTN FRBA — 2026
\`\`\`
