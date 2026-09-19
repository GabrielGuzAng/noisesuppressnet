#!/usr/bin/env bash
# Reanudación de V7 tras el corte de energía del 16/09/2026 06:50.
#
# V7GATE terminó completo (20 épocas, 15/09 23:00). V7CONTROL alcanzó la época
# 14 y murió a mitad de la 15 — no por nada del proceso: el journal del sistema
# se corta en seco a las 06:50:10, sin secuencia de apagado, y el ext4 de sdb1
# necesitó replay del journal en el arranque siguiente. Corte de energía.
#
# Se reanuda desde checkpoints/v7_control/epoch_14.pt. La reanudación es
# BIT-EXACTA: repone pesos, momentos de Adam, la fase del StepLR y el stream del
# RNG global, así que las épocas 15-20 ven el mismo orden de datos que habrían
# visto sin el corte. Verificado en tests/test_resume.py, que corre cada
# entrenamiento en su propio proceso justamente para cubrir este caso.
#
# Eso importa acá y no es un detalle: el endpoint primario E1 es el contraste
# promediado sobre las épocas 15-20, o sea exactamente las que se perdieron. Una
# reanudación "equivalente pero no idéntica" metería una perturbación del mismo
# orden que el efecto buscado (sd 0,0099 por checkpoint medida en V6), que es el
# mismo argumento por el que el control de V7 se reentrenó en vez de reusar V2.
#
# El barrido de evaluación es idempotente: saltea los .json que ya existen, así
# que se hace cargo del v7_gate_ep20_v2_es.json que ya estaba.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "=========== V7 reanudado — inicio $(date) ==========="

$PY -m training.trainer --config V7CONTROL \
    --resume-from checkpoints/v7_control/epoch_14.pt \
    || { echo "*** FALLO V7CONTROL"; exit 1; }

for spec in "v2_es:test_v2_metadata.json" "v1_en:test_v1_metadata.json"; do
  ts="${spec%%:*}"; md="${spec##*:}"
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
  echo ">>> $ts COMPLETO $(date +%H:%M:%S)"
done
echo "=========== V7 terminado $(date) ==========="
