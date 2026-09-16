#!/usr/bin/env bash
# Trayectoria por época de las dos ramas de V6 sobre los dos sellados que
# cargan los endpoints: test_v2_es (P1, primaria) y test_v1_en (P2, costo).
#
# NO se selecciona nada con esto. best.pt de las dos ramas cayó en la época 2
# por mínimo de val_loss, criterio que ya se sabe desalineado con PESQ (V4b y
# las réplicas de V5). Esto mide si el efecto de la compuerta está subestimado
# por evaluar una compuerta de solo 2 épocas. Se reporta la trayectoria entera.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# test_v2_es primero: es el endpoint primario.
for spec in "v2_es:test_v2_metadata.json" "v1_en:test_v1_metadata.json"; do
  ts="${spec%%:*}"; md="${spec##*:}"
  for ep in 01 02 03 04 05 06; do
    for br in v6_gate v6_placebo; do
      out="results/v6_epochs/${br}_ep${ep}_${ts}.json"
      [ -f "$out" ] && { echo "skip $out"; continue; }
      echo "=== $br epoca $ep sobre $ts  ($(date +%H:%M:%S))"
      $PY -m evaluation.evaluate_variant --variant "$br" \
          --checkpoint "checkpoints/${br}/epoch_${ep}.pt" \
          --test_dir "data/test_sealed/${ts}" \
          --metadata "seal_test_metadata/${md}" \
          --output "$out" 2>&1 | grep -E "Época|Compuerta|PESQ-NB|Δ:|Resultados" | head -6
    done
  done
  echo ">>> $ts COMPLETO $(date +%H:%M:%S)"
done
echo "=========== barrido de epocas V6 terminado $(date) ==========="
