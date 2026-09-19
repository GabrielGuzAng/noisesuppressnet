#!/usr/bin/env bash
# V7 sobre el TERCER sellado: test_v3_mls_es (audiolibro en español, MLS).
#
# Por qué falta. El barrido original de V7 (run_v7_scratch.sh) cubre los dos
# sellados que cargan E1 y E2. Pero E3 del preregistro pide ρ(g, SNR) negativo
# con |ρ| ≥ 0,15 en los TRES sellados, y sin este no se puede declarar E3.
# V6 sí reportó los tres (sección 5 de docs/v6_compuerta.md).
#
# Las dos ramas, épocas 15-20: la compuerta porque es la que tiene g, y el
# control para poder reportar también el contraste sobre este sellado con el
# mismo estimando promediado que E1/E2.
#
# 12 evaluaciones, ~2,5 min cada una: ~30 min. Idempotente.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ts=v3_mls_es
md=test_v3_mls_es_metadata.json

echo "=========== V7 sobre $ts — inicio $(date) ==========="
for ep in 15 16 17 18 19 20; do
  for br in v7_gate v7_control; do
    out="results/v7_epochs/${br}_ep${ep}_${ts}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "=== $br ep$ep sobre $ts  ($(date +%H:%M:%S))"
    $PY -m evaluation.evaluate_variant --variant "$br" \
        --checkpoint "checkpoints/${br}/epoch_${ep}.pt" \
        --test_dir "data/test_sealed/${ts}" --metadata "seal_test_metadata/${md}" \
        --output "$out" 2>&1 | grep -E "Compuerta|Resultados" | head -3
  done
done
echo "=========== $ts COMPLETO $(date) ==========="
