# Trazabilidad de números

Todo valor que entre al cuerpo del informe sale de esta tabla o de una fuente
citada explícitamente en el capítulo. Un número sin fila acá no se escribe.

Los JSON de `results/` tienen la estructura
`{variant, checkpoint_epoch, test_set, n_pairs_evaluated, global, by_bucket, by_noise_category, all_pairs}`.
`global` trae, por métrica, `noisy_mean`, `noisy_std`, `est_mean`, `est_std` y
`delta_mean`.

---

## Prohibidos — números retractados

**No vuelven al informe, ni con hedge, ni como "una primera lectura sugería".**
Si aparecen en un borrador, es un error de redacción, no un matiz.

| Número retractado | Dónde se publicó | Por qué cayó | Qué va en su lugar |
|---|---|---|---|
| "67 % de retención de la ganancia entre canales" | commit `f09e98f` | El estimando no era el mismo para las dos recetas | Con V2 sobre `test_v3_mls_es`: la mejora agnóstica al idioma transfiere al 132 %, los fine-tunings al español al 34–38 % (`decisions.md`, 09/09) |
| Costo en inglés de V6: −0,011 con p = 0,005 | borrador de `v6_compuerta.md` | No sobrevive a la trayectoria: rebota entre −0,013 y +0,031, media +0,004, 3 de 6 positivas | El contraste en inglés está centrado en cero. No hay trade-off entre idiomas en V6 (`v6_compuerta.md` §7) |
| V3e en inglés: −0,032 | `EXPERIMENTS.md` previo | Redondeo incorrecto | **−0,031** |
| "Score compuesto" que suma ganancia en ES y costo en EN | V3/V3b/V3e | Mezcla dos estimandos de naturaleza distinta | Los dos ejes se reportan por separado; el olvido se mide por fracción de archivos rotos, no por la media (`reanalisis_estadistico.md`, F2/F3) |

---

## Condiciones de referencia (audio ruidoso) — n = 250 por sellado

| Sellado | PESQ-NB | PESQ-WB | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| `test_v1_en` | 2,152 | 1,583 | 0,852 | 7,86 |
| `test_v2_es` | 2,161 | 1,641 | 0,850 | 8,41 |
| `test_v3_mls_es` | 2,195 | 1,636 | 0,851 | 8,03 |

Fuente: campo `global.*.noisy_mean` de cualquier JSON del sellado correspondiente.
Son idénticos entre variantes por construcción; sirve como verificación de
integridad.

---

## PESQ-NB por variante y sellado (`global.pesq_nb.est_mean`)

| Variante | época | `test_v1_en` | `test_v2_es` | `test_v3_mls_es` | JSON |
|---|---|---|---|---|---|
| V1 | 19 | 2,650 | 2,330 | 2,602 | `v1_{v1_en,v2_es,v3_mls_es}.json` |
| V2 | 19 | 2,849 | 2,451 | 2,761 | `v2_{…}.json` |
| V3 | 21 | 2,571 | 2,560 | — | `v3_{v1_en,v2_es}.json` |
| V3b | 5 | 2,621 | 2,483 | — | `v3b_{v1_en,v2_es}.json` |
| V3e | 14 | 2,619 | 2,551 | 2,686 | `v3e_{…}.json` |
| V5 (s42) | 21 | 2,774 | 2,686 | 2,840 | `v5_{…}.json` |
| V5 (s43) | 12 | 2,770 | 2,658 | 2,803 | `v5_s43_{…}.json` |
| V5 (s44) | 16 | 2,766 | 2,661 | 2,816 | `v5_s44_{…}.json` |

**Advertencia obligatoria sobre V5**: la semilla 42 es la más alta de las tres en
los tres sellados. El titular sale del máximo de tres corridas y el informe lo
tiene que decir donde reporte el número. Desviación estándar entre semillas:
0,015 (ES), 0,004 (EN), 0,019 (MLS). Las tres parten del mismo checkpoint de V2,
así que miden la varianza del **fine-tuning** (orden de los datos), no la de
inicialización (`decisions.md`, 08-09/09).

**V6 y V7 no van en esta tabla.** Su estimando es un contraste contra su propio
control, no un valor absoluto comparable con la línea del ablation. Mezclarlos en
la misma tabla es el error que el dashboard evita a propósito.

---

## V6 — contraste compuerta − placebo

| Endpoint preregistrado | Criterio | Resultado | Veredicto |
|---|---|---|---|
| P1, primaria, bucket [15,20] ES | ≥ +3 pp | +2,5 pp, p = 0,181 | **falla** |
| P2, costo acotado | — | pasa | pasa |
| P3, mecanismo ρ(g,SNR) | negativo | error de signo en el preregistro; el mecanismo se cumple con ρ ≈ −0,4, p < 1e−8 en los tres sellados | falla como está escrito |
| P4, degeneración | media de g fuera de los extremos | 0,57 a 0,65 | pasa |

Fuente: `docs/v6_compuerta.md` §1 y §5. Tag `7bcd30b`.
Por la regla de decisión preregistrada, la hipótesis **no queda sostenida**.

**Piso de ruido por checkpoint** (el hallazgo reutilizable, `v6_compuerta.md` §8):
sd = 0,0099 sobre `test_v2_es` y 0,0166 sobre `test_v1_en`, del mismo tamaño que
el efecto buscado. Desde el 15/09 el estimando primario del proyecto es la
trayectoria promediada, no el checkpoint seleccionado (`decisions.md`, 15/09).

---

## V7 — contraste compuerta − control, ambos desde cero

| Endpoint | Criterio preregistrado | Resultado | Veredicto |
|---|---|---|---|
| E1, contraste global ES, media ép. 15–20 | ≥ +0,050 | **+0,0795** | pasa |
| E2, costo en inglés | ≥ −0,020 | **+0,0620** | pasa |
| E3, mecanismo ρ(g,SNR) | negativo, \|ρ\| ≥ 0,15, tres sellados | −0,35 / −0,42 / −0,40 | pasa |
| E4, degeneración | media de g en (0,05, 0,99) | 0,42 a 0,51 | pasa |
| E5, ¿desde cero rinde más? | contra el +0,0187 de V6 | 4,3× | desde cero rinde |

Fuente: `results/v7_endpoints.json`, `docs/v7_compuerta_desde_cero.md`.
Sobre `test_v3_mls_es` el contraste es +0,0753. Patrón de signo `++++++` en los
tres sellados; archivos que favorecen la compuerta: 171/250 (ES), 175/250 (EN),
176/250 (MLS).

**Estado**: screening, no confirmación. El preregistro exige tres semillas antes
de escribirlo como aporte, y el estimando confirmatorio **excluye la semilla 42**,
que fue la que disparó la confirmación (`decisions.md`, 18/09).

---

## Arquitectura y configuración

| Dato | Valor | Fuente |
|---|---|---|
| Parámetros del modelo | 17.579.459 (17,58 M) | `models/crn.py`; coincide bit-exact con Fig. 5 de Tan & Wang 2018 |
| Parámetros en el LSTM | ~16,5 M (94 % del total) | `EXPERIMENTS.md` §V0 |
| Parámetros de la compuerta | 193 (+0,0011 %) | `v7_compuerta_desde_cero.md` §1 |
| Encoder | 5 conv2d, 1→16→32→64→128→256, kernel 2×3, stride (1,2) | `EXPERIMENTS.md` §V0 |
| Bottleneck | LSTM 2 capas, hidden 1024, unidireccional | ídem |
| n_fft / hop / win | 320 / 160 / 320, Hamming, F = 161 | `stft.py`, `EXPERIMENTS.md` §V0 |
| Latencia algorítmica | 10,0 ms (`center=True`) | medido; ver D6 en `DESVIACIONES.md` |

## Eficiencia

| Dato | Valor | Fuente |
|---|---|---|
| RTF mediana, i5-4460, 1 thread | 0,331 | `EXPERIMENTS.md` §V0, benchmarks |
| RTF p95, i5-4460, 1 thread | 0,340 | ídem |
| RTF mediana, RTX 4060 | 0,0148 | ídem |
| RTF p95, RTX 4060 | 0,0151 | ídem |

## Costo de entrenamiento

Generado el 19/09/2026 con `analysis/training_cost_report.py`.

| Variante | Épocas (best) | Horas | kWh | Eficiencia útil | USD nube |
|---|---|---|---|---|---|
| V1 | 30 (19) | 15,94 | 2,790 | 63,7 % | 5,58 |
| V2 | 20 (19) | 10,51 | 1,839 | 95,0 % | 3,68 |
| V3 | 30 (21) | 15,73 | 2,754 | 70,0 % | 5,51 |
| V3b | 10 (5) | 5,37 | 0,939 | 50,1 % | 1,88 |
| V3e | 25 (14) | 13,19 | 2,309 | 56,1 % | 4,62 |
| V5 (s42) | 25 (21) | 13,25 | 2,318 | 83,5 % | 4,64 |
| V5 (s43) | 25 (12) | 13,23 | 2,316 | 48,0 % | 4,63 |
| V5 (s44) | 25 (16) | 13,35 | 2,337 | 64,2 % | 4,67 |
| V6 compuerta | 6 (2) | 3,20 | 0,560 | 33,7 % | 1,12 |
| V6 placebo | 6 (2) | 3,12 | 0,546 | 33,3 % | 1,09 |
| V7 compuerta | 20 (18) | 10,60 | 1,854 | 89,8 % | 3,71 |
| V7 control | 20 (19) | 10,58 | 1,852 | 94,9 % | 3,70 |
| **Subtotal (12 corridas)** | | **128,09** | **22,415** | | **44,83** |

CO₂ del subtotal: 6,73 kg. Supuestos: 105 W de GPU + 70 W de sistema = 175 W;
0,3 kg CO₂/kWh (Cammesa, mix argentino 2024); USD 0,35/h (GCP T4).
"Eficiencia útil" es la fracción del tiempo total transcurrida hasta la mejor
época.

**El subtotal no es el total del proyecto.** Faltan, y hay que sumarlos antes de
escribir el capítulo 8:

| Falta | Brazos | Dónde está el `history.json` |
|---|---|---|
| V4 (sweep de α) | 5 | `checkpoints/v4_sweep/{alpha_070,alpha_080,alpha_090,alpha_095,control_mse}` |
| V4b (2×2 + placebo) | 3 | `checkpoints/v4b/{v4b_main,v4b_placebo,v4b_lr2e4}` |
| Sweep de lr de V3b | 5 | `checkpoints/v3_sweep/lr_*` |
| V0 | 1 | `checkpoints/v0` |
| Corridas smoke descartables | 4 | `checkpoints/{v4_smoke,v5_smoke,v5_smoke_lr5e5,v5_smoke_lr2e4}` |

**Nota metodológica para el capítulo 8**: V4 y V4b **no tienen un `history.json`
único** porque nunca fueron una corrida sino un diseño de varios brazos. El costo
de la línea de proxy perceptual es la suma de sus 8 brazos, y así es como hay que
reportarlo: es el precio de haber cerrado la línea con diseño experimental en vez
de con una sola corrida.

**Decisión pendiente**: si los sweeps y los smoke entran en el total del proyecto.
Argumento para incluirlos: son horas realmente gastadas y el tribunal pregunta por
el costo real. Argumento para separarlos: no producen resultados reportados.
Recomendación: dos totales, "corridas reportadas" y "total de proyecto".

## Hashes de sellado

| Sellado | SHA-256 acumulativo | Metadata |
|---|---|---|
| `test_v1_en` | `8eb360dc54343f1a04e8fdbbe3fbf6d2c5a801e32a2d5c647b14504eec400fa4` | `seal_test_metadata/test_v1_metadata.json` |
| `test_v2_es` | `96f33f837627ec3c30304c83e4d500d3c557bc23822cf16e12fa12b729e2c972` | `seal_test_metadata/test_v2_metadata.json` |
| `test_v3_mls_es` | `fa99a0faaf2d1c78b5c2d1ef79a513a9557bc4255ceadfb36230df3f5bd21306` | `seal_test_metadata/test_v3_mls_es_metadata.json` |

Los tres con seed 42, 250 pares, 500 WAV. `test_v3_mls_es` reusa las condiciones
de ruido de `test_v1_metadata.json` verbatim, que es lo que lo convierte en
control de canal.

## Preregistros con hash

| Preregistro | Archivo | Qué congela |
|---|---|---|
| MLS ES | `docs/preregistro_mls_es.sha256` | Predicciones P1 y P2 antes de bajar el dato |
| Compuerta V6 | `docs/preregistro_compuerta.sha256` | P1–P4 y la regla de decisión |
| Compuerta desde cero V7 | `docs/preregistro_v7_desde_cero.sha256` | E1–E5 y la tabla de desenlaces |
| Confirmación V7 | — | **no existe todavía**; borrador fuera del repo |
