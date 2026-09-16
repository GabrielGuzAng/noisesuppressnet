#!/usr/bin/env bash
# V7: compuerta desde cero, screening. Tratamiento + control, 20 epocas cada uno.
# Preregistro: ~/nosiesuppressnet-oracle/preregistro_compuerta_desde_cero.md
# Hash:        docs/preregistro_v7_desde_cero.sha256
#
# E1 (primaria) es el contraste promediado sobre las epocas 15-20, asi que se
# evaluan esas seis de cada brazo sobre los dos sellados que cargan E1 y E2.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "=========== V7 desde cero — inicio $(date) ==========="

for cfg in V7GATE V7CONTROL; do
  echo ">>> $cfg  $(date +%H:%M:%S)"
  $PY -m training.trainer --config "$cfg" || { echo "*** FALLO $cfg"; exit 1; }
done

for spec in "v2_es:test_v2_metadata.json" "v1_en:test_v1_metadata.json"; do
  ts="${spec%%:*}"; md="${spec##*:}"
  for ep in 15 16 17 18 19 20; do
    for br in v7_gate v7_control; do
      out="results/v7_epochs/${br}_ep${ep}_${ts}.json"
      [ -f "$out" ] && continue
      echo "=== $br ep$ep sobre $ts  ($(date +%H:%M:%S))"
      $PY -m evaluation.evaluate_variant --variant "$br" \
          --checkpoint "checkpoints/${br}/epoch_${ep}.pt" \
          --test_dir "data/test_sealed/${ts}" --metadata "seal_test_metadata/${md}" \
          --output "$out" 2>&1 | grep -E "Compuerta|Resultados" | head -3
    done
  done
  echo ">>> $ts COMPLETO $(date +%H:%M:%S)"
done
echo "=========== V7 terminado $(date) ==========="
