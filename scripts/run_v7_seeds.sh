#!/usr/bin/env bash
# Confirmación de V7 con réplicas de semilla: semillas 43 y 44, los dos brazos.
#
# V7 con semilla 42 fue SCREENING: E1 = +0,0795 contra un umbral de +0,050, con
# n=1 por brazo. El preregistro ata ese desenlace a confirmar con tres semillas.
# Esto corre las dos que faltan. La 42 ya está.
#
# ANTES DE LANZAR: el preregistro de la confirmación tiene que estar escrito y
# hasheado. La decisión que hay que tomar ahí y que este script no toma: si el
# estimando confirmatorio es el promedio de las DOS semillas nuevas (la 42 disparó
# la confirmación, así que incluirla infla el resultado) o el de las tres. Las dos
# opciones son defendibles; lo que no es defendible es elegir después de ver el dato.
#
# Costo: 4 entrenamientos x ~10,6 h = ~42 h, más ~3 h del barrido de evaluación
# (72 evaluaciones sobre los tres sellados). Total ~45 h.
#
# REANUDACIÓN: si un corte de energía mata una corrida, relanzar este mismo
# script. Detecta el checkpoint periódico más alto de cada directorio y reanuda
# bit-exacto desde ahí (tests/test_resume.py); las corridas ya terminadas las
# saltea por su history.json.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# No arrancar encima de otro job de GPU: 8 GB no dan para dos.
while pgrep -f "evaluation.evaluate_variant|training.trainer" > /dev/null; do
  echo "[$(date +%H:%M:%S)] hay otro job de GPU corriendo, esperando..."
  sleep 120
done

echo "=========== V7 semillas 43/44 — inicio $(date) ==========="

for spec in "V7GATE_S43:v7_gate_s43" "V7CONTROL_S43:v7_control_s43" \
            "V7GATE_S44:v7_gate_s44" "V7CONTROL_S44:v7_control_s44"; do
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

# Barrido de evaluación: las mismas épocas 15-20 del estimando, los tres sellados.
for spec in "v2_es:test_v2_metadata.json" "v1_en:test_v1_metadata.json" \
            "v3_mls_es:test_v3_mls_es_metadata.json"; do
  ts="${spec%%:*}"; md="${spec##*:}"
  for ep in 15 16 17 18 19 20; do
    for br in v7_gate_s43 v7_control_s43 v7_gate_s44 v7_control_s44; do
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
echo "=========== V7 semillas terminado $(date) ==========="
