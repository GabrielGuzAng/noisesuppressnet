#!/usr/bin/env bash
# Las tres sedes de contraste sobre test_wham_es (ruido holdout, WHAM!).
#
# Los brazos son los que declaran las secciones 4 y 7 del preregistro hasheado;
# no se agrega ninguno sin enmienda. Cada bloque de aca es un contraste o un
# descriptivo declarado:
#
#   C1  la compuerta      V7 gate vs V7 control, semillas 43/44/45, épocas 15-20
#                         Mismo estimando que la confirmación de V7: trayectoria
#                         promediada, porque el efecto (+0,044 en audiolibro) es
#                         ~4x el piso de ruido por checkpoint y un solo
#                         checkpoint no alcanza.           36 evals
#
#   C2  la adaptación     V5 (3 semillas) vs V2, un checkpoint cada uno.
#                         Acá sí alcanza: el efecto es +0,2175, 22x el piso. Y
#                         además V5/V2 sólo guardaron best.pt.      4 evals
#
#   C3  el mecanismo      rho(g, SNR) <= -0,15 en los 18 checkpoints de compuerta.
#                         Sale gratis: gate_stats viaja con cada evaluacion. 0 evals
#
#   D1  descriptivo       DeepFilterNet2, para describir quien se degrada mas al
#                         pasar de ruido visto a no visto. NO puntuado: la
#                         exposicion k esta confundida con el modelo (seccion 6). 1 eval
#
#   D3  descriptivo       semilla 42 de V7, la que genero la hipotesis. Fuera del
#                         estimando confirmatorio por diseno. OBLIGATORIO: la
#                         seccion 7 lo pide siempre y adjudicar() lo exige.  12 evals
#
#   D4  descriptivo       V1, para rho(ganancia de V1, SNR) del aporte 1. Decidido
#                         antes del hash, no despues de ver el dato.        1 eval
#
# 54 evaluaciones, ~2,5 min cada una: ~2 h 15. Idempotente: saltea lo que ya esta.
# NO cortar antes del final: sin el bloque D la adjudicacion tira DatosFaltantes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PYB=.venv-baselines/bin/python
ts=wham_es
md=seal_test_metadata/test_wham_es_metadata.json
td=data/test_sealed/wham_es

[ -d "$td" ] || { echo "FALTA $td. Corré primero: python -m scripts.seal_test_set_wham_es"; exit 1; }
mkdir -p results/v7_epochs logs

echo "=========== WHAM! ($ts) — inicio $(date) ==========="

run_epoch () {  # $1 rama  $2 época
  local out="results/v7_epochs/${1}_ep${2}_${ts}.json"
  [ -f "$out" ] && { echo "skip $out"; return; }
  echo "--- $1 ep$2  ($(date +%H:%M:%S))"
  $PY -m evaluation.evaluate_variant --variant "$1" \
      --checkpoint "checkpoints/${1}/epoch_${2}.pt" \
      --test_dir "$td" --metadata "$md" --output "$out" 2>&1 \
    | grep -E "PESQ-NB|Compuerta|RESULTADOS" | head -4
}

run_best () {  # $1 variante
  local out="results/${1}_${ts}.json"
  [ -f "$out" ] && { echo "skip $out"; return; }
  echo "--- $1 (best.pt)  ($(date +%H:%M:%S))"
  $PY -m evaluation.evaluate_variant --variant "$1" \
      --test_dir "$td" --metadata "$md" --output "$out" 2>&1 \
    | grep -E "PESQ-NB|Compuerta|RESULTADOS" | head -4
}

echo; echo "########## C1 — la compuerta (confirmatorio: 43/44/45) ##########"
for s in s43 s44 s45; do
  for ep in 15 16 17 18 19 20; do
    for br in v7_gate v7_control; do run_epoch "${br}_${s}" "$ep"; done
  done
done

echo; echo "########## C2 — la adaptación ##########"
for v in v5 v5_s43 v5_s44 v2; do run_best "$v"; done

echo; echo "########## D1 — descriptivo: DeepFilterNet2 ##########"
if [ -f "results/deepfilternet2_${ts}.json" ]; then
  echo "skip results/deepfilternet2_${ts}.json"
else
  echo "--- realce DFN2 (venv aparte)  ($(date +%H:%M:%S))"
  $PYB -m baselines.run_external --baseline deepfilternet2 \
      --test_dir "$td" --metadata "$md" 2>&1 | tail -3
  run_best deepfilternet2
fi

echo; echo "########## D3 — descriptivo: semilla 42 (OBLIGATORIO) ##########"
for ep in 15 16 17 18 19 20; do
  for br in v7_gate v7_control; do run_epoch "$br" "$ep"; done
done

echo; echo "########## D4 — descriptivo: V1 (aporte 1) ##########"
run_best v1

echo; echo "=========== $ts COMPLETO $(date) ==========="
echo "Adjudicar:  $PY -m analysis.wham_holdout_endpoints"
echo "            (NO v7_seeds_endpoints: ese reescribe el registro adjudicado de V7)"
