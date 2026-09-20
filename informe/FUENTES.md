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

## Baseline no-DL: Butterworth pasabajo 4 kHz, orden 5

Corrido el 20/09/2026 sobre los tres sellados, en sus dos modos. Fuente:
`results/butterworth_{causal,zerophase}_{v1_en,v2_es,v3_mls_es}.json`.

| Sellado | Condición | PESQ-NB | PESQ-WB | STOI | SI-SDR |
|---|---|---|---|---|---|
| `test_v1_en` | Noisy | 2,152 | 1,583 | 0,852 | 7,86 dB |
| | Butter fase cero | 2,153 | 1,553 | 0,851 | 6,28 dB |
| | Butter causal | 2,163 | 1,586 | 0,852 | 0,36 dB |
| | **V2** | **2,849** | **2,211** | **0,908** | **14,44 dB** |
| `test_v2_es` | Noisy | 2,161 | 1,641 | 0,850 | 8,41 dB |
| | Butter fase cero | 2,163 | 1,652 | 0,849 | 7,05 dB |
| | Butter causal | 2,163 | 1,682 | 0,850 | 2,15 dB |
| | **V5** | **2,686** | **2,103** | **0,896** | **14,27 dB** |
| `test_v3_mls_es` | Noisy | 2,195 | 1,636 | 0,851 | 8,03 dB |
| | Butter fase cero | 2,198 | 1,620 | 0,850 | 6,99 dB |
| | Butter causal | 2,196 | 1,650 | 0,850 | 2,27 dB |
| | **V5** | **2,840** | **2,196** | **0,905** | **13,64 dB** |

### Por qué se reportan los dos modos

El pasabajo admite dos implementaciones y **ninguna de las dos favorece al
baseline de forma uniforme**, así que elegir una sola sería elegir el resultado:

- **Fase cero** (`sosfiltfilt`, filtra hacia adelante y hacia atrás): no
  introduce distorsión de fase, y por eso conserva SI-SDR —métrica sensible a la
  forma de onda—. Pero **no es causal**: ve el futuro de la señal, que es
  exactamente la restricción operativa a la que sí está sujeto el CRN. Es la
  implementación con la que se produjeron los números de V0.
- **Causal** (`sosfilt`, una sola pasada): cumple la restricción, y por eso es la
  comparación pareja y la que se reporta como primaria. Introduce distorsión de
  fase, que hunde SI-SDR (0,36 dB en inglés contra 6,28 dB del modo de fase cero).

El reparto es el opuesto en PESQ: **el modo causal da PESQ-WB más alto que el de
fase cero en los tres sellados** (1,586 contra 1,553; 1,682 contra 1,652; 1,650
contra 1,620), porque PESQ es poco sensible a fase y el filtrado bidireccional
atenúa el doble. De modo que el modo causal favorece al baseline en PESQ y el de
fase cero lo favorece en SI-SDR. Reportar los dos, y contrastar contra el mejor
de ambos en cada métrica, deja la afirmación de OP-6 sin depender de esa
elección.

### Lo que el baseline mide, además de OP-6

El pasabajo de 4 kHz es, en la práctica, **una operación nula sobre PESQ**:
+0,001 a +0,011 puntos de PESQ-NB sobre el ruidoso en los tres sellados, y
0,000 a −0,001 en STOI. Lo único que hace de forma apreciable es costar SI-SDR.

Es la respuesta empírica a la sugerencia recibida en la defensa del preproyecto
—aplicar un pasabajo donde la PSD cae 20 dB—, ahora medida sobre material sellado
y no argumentada por analogía con la literatura. Coherente con Braun & Tashev
2021, que reporta degradación de PESQ-WB de 0,15 a 0,30 puntos por filtrado
pasabajo, y con que DeepFilterNet2 no filtre su entrada.

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

Generado con `analysis/training_cost_report.py` los días 19 y 20/09/2026.

### Corridas reportadas, seleccionadas por `best.pt` (8)

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
| **Subtotal** | | **100,58** | **17,602** | | **35,20** |

### Corridas cuyo estimando es la trayectoria, no un checkpoint (4)

| Variante | Épocas | Horas | kWh | Eficiencia útil | USD nube |
|---|---|---|---|---|---|
| V6 compuerta | 6 | 3,20 | 0,560 | n/c | 1,12 |
| V6 placebo | 6 | 3,12 | 0,546 | n/c | 1,09 |
| V7 compuerta | 20 | 10,60 | 1,854 | n/c | 3,71 |
| V7 control | 20 | 10,58 | 1,852 | n/c | 3,70 |
| **Subtotal** | | **27,50** | **4,813** | | **9,63** |

### Línea de proxy perceptual — V4 y V4b, 8 brazos

V4 y V4b **no tienen un `history.json` único** porque nunca fueron una corrida
sino un diseño de varios brazos, evaluado época por época. El costo de la línea
es la suma, y así se reporta: es el precio de haberla cerrado con diseño
experimental en vez de con una sola corrida.

| Brazo | Épocas | Horas | kWh | USD nube |
|---|---|---|---|---|
| V4 `alpha_070` | 3 | 3,21 | 0,562 | 1,12 |
| V4 `alpha_080` | 3 | 3,27 | 0,572 | 1,14 |
| V4 `alpha_090` | 3 | 3,29 | 0,575 | 1,15 |
| V4 `alpha_095` | 3 | 3,23 | 0,566 | 1,13 |
| V4 `control_mse` | 3 | 1,59 | 0,278 | 0,56 |
| *subtotal V4* | | *14,59* | *2,553* | *5,11* |
| V4b `v4b_main` | 3 | 3,24 | 0,566 | 1,13 |
| V4b `v4b_placebo` | 3 | 1,56 | 0,272 | 0,54 |
| V4b `v4b_lr2e4` | 3 | 3,22 | 0,564 | 1,13 |
| *subtotal V4b* | | *8,02* | *1,403* | *2,81* |
| **Línea completa** | | **22,60** | **3,955** | **7,91** |

**Verificación de consistencia**: el subtotal de V4b da 8,02 h, que coincide con
las 8,02 h registradas en `decisions.md` el 01/09/2026 de forma independiente.

**Dato a interpretar, no sólo a tabular**: los dos brazos sin término perceptual
—`control_mse` (1,59 h) y `v4b_placebo` (1,56 h)— cuestan aproximadamente la
mitad por época que los brazos con Squim. El forward del proxy duplica el costo
del paso de entrenamiento. Es un argumento adicional contra la línea, además del
gaming.

### Totales

| | Horas GPU | kWh | kg CO₂ | USD nube |
|---|---|---|---|---|
| Corridas reportadas (12) | 128,09 | 22,415 | 6,72 | 44,83 |
| Línea de proxy (8 brazos) | 22,60 | 3,955 | 1,19 | 7,91 |
| **Total medido (20 corridas)** | **150,69** | **26,370** | **7,91** | **52,74** |

**Todavía sin contabilizar**: V0, los 5 brazos del sweep de lr de V3b
(`checkpoints/v3_sweep/lr_*`) y las 4 corridas smoke descartables
(`checkpoints/{v4_smoke,v5_smoke,v5_smoke_lr5e5,v5_smoke_lr2e4}`).

**Decisión pendiente**: si los sweeps y los smoke entran en el total del proyecto.
Argumento para incluirlos: son horas realmente gastadas y el tribunal pregunta por
el costo real. Argumento para separarlos: no producen resultados reportados.
Recomendación: mantener los dos totales y explicar la diferencia.

### Sobre la métrica de eficiencia útil

`efficiency_percent` es `tiempo_hasta_best / tiempo_total`, y **sólo tiene
sentido para las corridas cuyo resultado reportado sale de `best.pt`** — las 8 de
la primera tabla. Ahí mide entrenamiento gastado después del checkpoint que
efectivamente se usó.

En las demás **no se reporta** (`n/c`):

- **V6 y V7**: el estimando es la trayectoria promediada —las 6 épocas en V6, las
  15 a 20 en V7—. Las épocas posteriores al mínimo de `val_loss` no son
  desperdicio: son el dato. Leerlas como ineficiencia invierte el sentido del
  diseño.
- **V4 y V4b**: corridas de 3 épocas evaluadas época por época, precisamente
  porque con `mse_plus_squim` la selección por `val_loss` está contaminada. Cuatro
  de los ocho brazos dan 100 % sólo porque la mejor época fue la última de tres.

Supuestos del reporte: 105 W de GPU + 70 W de sistema = 175 W; 0,3 kg CO₂/kWh
(Cammesa, mix argentino 2024); USD 0,35/h (GCP T4). El consumo es estimado a
partir de potencia nominal sostenida, no medido con instrumento; el propio
reporte acota el error en ±15 %.

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
