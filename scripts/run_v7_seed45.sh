#!/usr/bin/env bash
# Tercera réplica de confirmación de V7: semilla 45, los dos brazos.
#
# Va en un script APARTE y no dentro de run_v7_seeds.sh a propósito: ese script
# está corriendo, y bash lee el archivo de forma incremental desde el descriptor
# abierto. Editarlo en el lugar puede hacer que retome desde un offset
# equivocado y ejecute basura. Nunca se toca un script en ejecución.
#
# Se puede lanzar ya: la guarda de abajo espera a que termine el job de GPU en
# curso (las semillas 43 y 44, más su barrido) antes de empezar.
#
# Costo: 2 entrenamientos x ~10,6 h + ~1,5 h de barrido sobre los tres sellados.
# Con esto el total de la confirmación llega a 6 corridas, que es la lectura
# estricta de las "~63 h" que presupuestaba el preregistro de V7.
#
# REANUDACIÓN tras un corte: relanzar este mismo script. Detecta el checkpoint
# periódico más alto y reanuda bit-exacto (tests/test_resume.py).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# La guarda mira también el script padre y no sólo los procesos de python. Entre
# que una corrida de run_v7_seeds.sh termina y arranca la siguiente hay un hueco
# de segundos sin ningún proceso de GPU vivo; si un sondeo cayera justo ahí, las
# dos cosas arrancarían juntas y 8 GB no dan para dos. El script padre sí sigue
# vivo durante ese hueco.
while pgrep -f "run_v7_seeds.sh|evaluation.evaluate_variant|training.trainer" > /dev/null; do
  echo "[$(date +%H:%M:%S)] hay otro job de GPU corriendo, esperando..."
  sleep 300
done

echo "=========== V7 semilla 45 — inicio $(date) ==========="

for spec in "V7GATE_S45:v7_gate_s45" "V7CONTROL_S45:v7_control_s45"; do
  cfg="${spec%%:*}"; dir="checkpoints/${spec##*:}"

  if [ -f "$dir/history.json" ]; then
    n=$($PY -c "import json;print(len(json.load(open('$dir/history.json'))['val_loss']))" 2>/dev/null || echo 0)
    [ "$n" = "20" ] && { echo ">>> $cfg ya tiene 20 épocas, se saltea"; continue; }
  fi

  last=$(ls "$dir"/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
  if [ -n "${last:-}" ]; then
    echo ">>> $cfg reanudando desde $last  $(date +%H:%M:%S)"
    $PY -m training.trainer --config "$cfg" --resume-from "$last" \
        || { echo "*** FALLO $cfg"; exit 1; }
  else
    echo ">>> $cfg desde cero  $(date +%H:%M:%S)"
    $PY -m training.trainer --config "$cfg" || { echo "*** FALLO $cfg"; exit 1; }
  fi
done

for spec in "v2_es:test_v2_metadata.json" "v1_en:test_v1_metadata.json" \
            "v3_mls_es:test_v3_mls_es_metadata.json"; do
  ts="${spec%%:*}"; md="${spec##*:}"
  for ep in 15 16 17 18 19 20; do
    for br in v7_gate_s45 v7_control_s45; do
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
echo "=========== V7 semilla 45 terminado $(date) ==========="
