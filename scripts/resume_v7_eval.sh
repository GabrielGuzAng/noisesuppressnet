#!/usr/bin/env bash
# Reanudación del BARRIDO DE EVALUACIÓN de V7 tras el segundo corte de energía
# del 16/09/2026 ~13:08.
#
# Historia de los dos cortes:
#   16/09 06:50 — muere V7CONTROL a mitad de la época 15. Se reanuda bit-exacto
#                 desde epoch_14.pt (scripts/resume_v7_scratch.sh).
#   16/09 13:08 — el entrenamiento ya había terminado (20/20 a las 12:39) y el
#                 corte cae sobre el barrido de evaluación, con `v2_es` completo
#                 y `v1_en` recién empezado (v7_gate ep15 guardado, v7_control
#                 ep15 en vuelo). Journal cortado en seco otra vez, replay de
#                 ext4 en sdb1 al arrancar: energía, no el proceso.
#
# Este script NO entrena. Lo verifica antes de arrancar: relanzar
# resume_v7_scratch.sh tal cual volvería a reanudar desde epoch_14.pt y
# reharía 3,2 h de épocas 15-20 pisando los checkpoints que ya están.
#
# El barrido es idempotente: saltea los .json existentes. Quedan 11 de 24.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Guarda: el entrenamiento tiene que estar terminado, si no este script no aplica.
for br in v7_gate v7_control; do
  [ -f "checkpoints/${br}/epoch_20.pt" ] || {
    echo "*** falta checkpoints/${br}/epoch_20.pt: el entrenamiento de ${br} no terminó."
    echo "*** este script es sólo de evaluación. Revisar antes de seguir."
    exit 1; }
  n=$($PY -c "import json;print(len(json.load(open('checkpoints/${br}/history.json'))['val_loss']))")
  [ "$n" = "20" ] || { echo "*** ${br}/history.json tiene $n épocas, no 20."; exit 1; }
done

echo "=========== V7 barrido reanudado — inicio $(date) ==========="
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
echo "=========== V7 barrido terminado $(date) ==========="
