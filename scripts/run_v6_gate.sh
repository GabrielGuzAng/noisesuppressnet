#!/usr/bin/env bash
# Corre el experimento de la compuerta (V6): tratamiento + placebo, y evalua
# cada rama sobre los tres sellados apenas termina de entrenar, para que si la
# segunda no llega haya resultados completos de la primera.
#
# Preregistro: ~/nosiesuppressnet-oracle/preregistro_compuerta.md
# Hash:        docs/preregistro_compuerta.sha256
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

evaluate () {   # $1 = nombre de variante (dir de checkpoints)
  for spec in "v1_en:test_v1_metadata.json" \
              "v2_es:test_v2_metadata.json" \
              "v3_mls_es:test_v3_mls_es_metadata.json"; do
    ts="${spec%%:*}"; md="${spec##*:}"
    echo "--- evaluando $1 sobre $ts  ($(date +%H:%M:%S))"
    $PY -m evaluation.evaluate_variant --variant "$1" \
        --test_dir "data/test_sealed/$ts" \
        --metadata "seal_test_metadata/$md" || echo "*** FALLO eval $1/$ts"
  done
}

echo "=========== V6 compuerta — inicio $(date) ==========="

echo ">>> TRATAMIENTO (con compuerta)  $(date +%H:%M:%S)"
$PY -m training.trainer --config V6GATE || { echo "*** FALLO entrenamiento tratamiento"; exit 1; }
evaluate v6_gate

echo ">>> PLACEBO (sin compuerta)  $(date +%H:%M:%S)"
$PY -m training.trainer --config V6PLACEBO || { echo "*** FALLO entrenamiento placebo"; exit 1; }
evaluate v6_placebo

echo "=========== V6 compuerta — fin $(date) ==========="
