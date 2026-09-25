#!/usr/bin/env bash
# KPI propio (tercer eje del aporte): correlación apareada entre el PESQ real y
# el estimado por Squim, sobre los tres sellados. Ver el docstring de
# evaluation/monitor_correlation.py y docs/PLAN_V4.md "Fase 4".
#
# NO LANZAR CON LA GPU OCUPADA: cada corrida levanta el CRN y Squim a la vez.
#
# Costo: 15 corridas de 250 pares. En GPU, del orden de 45-60 min en total; no
# hay entrenamiento, solo inferencia. En CPU multiplicar por ~20.
#
# Los tres grupos del régimen (3) no se declaran acá: la exposición a Squim la
# deduce el script del `loss` que figura en el config.json de cada checkpoint.
# Lo que sí se declara acá es el apareo checkpoint <-> resultados publicados,
# que es donde se cometen los errores. Dos trampas concretas:
#   - v4b_placebo/best.pt es la ÉPOCA 1 (mínimo de val MSE), pero el control de
#     protocolo idéntico que usa docs/decisions.md es la época 3. Se corren las
#     dos: la 3 como control primario, la 1 como secundario.
#   - los checkpoints de v4_sweep y v4b no siguen checkpoints/<variante>/best.pt
#     ni results/<variante>_<sellado>.json, así que van con rutas explícitas.
# El script de Python aborta si la época del checkpoint no coincide con la del
# JSON de PESQ real, así que un apareo mal escrito acá falla ruidoso.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MOD="evaluation.monitor_correlation"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "Hay otro proceso usando la GPU. Abortando." >&2
    exit 1
fi

# ── Limpias: nunca vieron Squim en su loss, y son la línea principal ──
for variant in v1 v2 v5; do
    for testset in v1_en v2_es v3_mls_es; do
        echo "=== ${variant} / ${testset} ==="
        $PY -m $MOD --variant "$variant" --test-set "$testset"
    done
done

# ── Gameadas: los dos extremos de la dosis-respuesta de V4, más V4b ──
# alpha_070 y alpha_095 son los extremos del sweep; v4b_main es el protocolo
# estabilizado de Xu. Los tres solo tienen resultados publicados sobre v1_en.
$PY -m $MOD --variant alpha_070 \
    --checkpoint checkpoints/v4_sweep/alpha_070/best.pt \
    --real-pesq-json results/v4_sweep/alpha_070_v1_en.json \
    --output results/v4_sweep/alpha_070_v1_en_correlation.json

$PY -m $MOD --variant alpha_095 \
    --checkpoint checkpoints/v4_sweep/alpha_095/best.pt \
    --real-pesq-json results/v4_sweep/alpha_095_v1_en.json \
    --output results/v4_sweep/alpha_095_v1_en_correlation.json

$PY -m $MOD --variant v4b_main \
    --checkpoint checkpoints/v4b/v4b_main/best.pt \
    --real-pesq-json results/v4b/v4b_main_epoch_03_v1_en.json \
    --output results/v4b/v4b_main_v1_en_correlation.json

# ── Controles: mismo protocolo que las gameadas, sin término perceptual ──
# Sin estos, el resultado se lee al revés. Es la lección de V4b.
$PY -m $MOD --variant control_mse \
    --checkpoint checkpoints/v4_sweep/control_mse/best.pt \
    --real-pesq-json results/v4_sweep/control_mse_v1_en.json \
    --output results/v4_sweep/control_mse_v1_en_correlation.json

$PY -m $MOD --variant v4b_placebo_epoch_03 \
    --checkpoint checkpoints/v4b/v4b_placebo/epoch_03.pt \
    --real-pesq-json results/v4b/v4b_placebo_epoch_03_v1_en.json \
    --output results/v4b/v4b_placebo_epoch_03_v1_en_correlation.json

$PY -m $MOD --variant v4b_placebo_epoch_01 \
    --checkpoint checkpoints/v4b/v4b_placebo/best.pt \
    --real-pesq-json results/v4b/v4b_placebo_epoch_01_v1_en.json \
    --output results/v4b/v4b_placebo_epoch_01_v1_en_correlation.json

# ── Régimen (3): la tabla limpio vs gameado. Exploratorio. ──
$PY -m $MOD --summarize \
    results/v1_v1_en_correlation.json \
    results/v1_v2_es_correlation.json \
    results/v1_v3_mls_es_correlation.json \
    results/v2_v1_en_correlation.json \
    results/v2_v2_es_correlation.json \
    results/v2_v3_mls_es_correlation.json \
    results/v5_v1_en_correlation.json \
    results/v5_v2_es_correlation.json \
    results/v5_v3_mls_es_correlation.json \
    results/v4_sweep/alpha_070_v1_en_correlation.json \
    results/v4_sweep/alpha_095_v1_en_correlation.json \
    results/v4_sweep/control_mse_v1_en_correlation.json \
    results/v4b/v4b_main_v1_en_correlation.json \
    results/v4b/v4b_placebo_epoch_03_v1_en_correlation.json \
    results/v4b/v4b_placebo_epoch_01_v1_en_correlation.json \
    --output results/squim_correlation_summary.json

echo
echo "Listo. El régimen (1) tiene que dar lo mismo en las corridas del mismo"
echo "sellado (mismo audio noisy, mismo Squim): si no, hay algo no determinista."
