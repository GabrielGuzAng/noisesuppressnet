# Capítulo 5 — Desarrollo por variante (V0 a V7)

**Extensión estimada**: 28 pp · **Estado**: parcial
**Fuentes primarias**: `docs/EXPERIMENTS.md`, `docs/decisions.md`,
`results/*.json`, `docs/v6_compuerta.md`, `docs/v7_compuerta_desde_cero.md`,
`training/config.py`.

El capítulo más largo. Una sección por experimento, con la misma plantilla en
todas: **pregunta / diseño / resultado / qué queda**. La plantilla ya está
validada: es la de las fichas del dashboard.

Los resultados negativos se desarrollan en el capítulo 6, pero **aparecen acá en
su lugar cronológico** con un resumen y el reenvío. No se los saltea: el orden
del capítulo es el orden en que se tomaron las decisiones.

---

## Estado de la materia prima

`docs/EXPERIMENTS.md` es la bitácora formal y **está incompleta**. Antes de
redactar este capítulo hay que cerrarla, porque es la fuente citable:

| Variante | Sección en `EXPERIMENTS.md` | Estado |
|---|---|---|
| V0 | sí, completa | lista |
| V1 | sí | lista |
| V2 | sí | lista |
| V3 | sí | lista |
| V3b | **no** | falta |
| V3e | **no** | falta |
| V4 | **no** | falta (está en `decisions.md`, 31/08) |
| V4b | **no** | falta (está en `decisions.md`, 01/09) |
| V5 | **no** | falta |
| V6 | sí | lista |
| V7 | sí | lista |

Cinco secciones faltantes. El material existe en `decisions.md` y en los commits;
lo que falta es la narrativa consolidada. La de V3→V3b→V3e está documentada en el
mensaje del commit `7f7884d` pero no en la bitácora.

---

## 5.1 V0 — Baseline y output collapse

- **Pregunta**: ¿el pipeline end-to-end funciona, y qué hace un CRN con datos
  insuficientes?
- **Diseño**: 200 pares, 10 épocas, MSE sobre magnitud.
- **Resultado**: PESQ-NB 1,417, STOI 0,794, SI-SDR 3,11 dB. **Por debajo del
  audio ruidoso** (2,050) y por debajo del baseline Butterworth (2,052).
  Output collapse severo: el modelo aprendió a predecir silencio uniforme en
  altas frecuencias.
- **Diagnóstico**: dataset insuficiente más underfitting extremo, confirmado por
  análisis de PSD (caída de 20 dB entre 0 y 8 kHz).
- **Fuente**: `results/summary_metrics.csv`; `EXPERIMENTS.md` §V0, "Diagnóstico:
  output collapse identificado"; `figures/psd_comparison.png`.
- **Qué queda**: el caso negativo que justifica el escalado del dataset. Y el
  único punto del proyecto donde existe comparación contra el Butterworth.
- **Falta**: nada para redactar. Figura de PSD ya disponible.

## 5.1.b Baseline no-DL: Butterworth pasabajo

- **Pregunta**: ¿cuánto de la mejora del CRN se consigue con procesamiento de
  señales clásico, y qué le hace al audio un pasabajo de 4 kHz?
- **Diseño**: Butterworth orden 5, corte en 4 kHz, sobre los tres sellados, en
  modo causal (primario) y de fase cero. n = 250 por sellado.
- **Resultado**: el filtro es **prácticamente una operación nula sobre PESQ**
  —entre +0,001 y +0,011 puntos de PESQ-NB sobre el ruidoso— y entre 0,000 y
  −0,001 en STOI. Lo único que mueve de forma apreciable es SI-SDR, hacia abajo:
  −7,50 dB en modo causal sobre `test_v1_en`.
- **Fuente**: `results/butterworth_{causal,zerophase}_{v1_en,v2_es,v3_mls_es}.json`;
  tabla completa en `FUENTES.md`.
- **Qué queda**: OP-6 cerrado, todas las variantes del CRN lo superan en los tres
  sellados. Y la respuesta empírica a la sugerencia de la defensa del preproyecto
  sobre aplicar un pasabajo: medida sobre material sellado, no argumentada por
  analogía.
- **Dato a no pasar por alto**: en `test_v2_es`, V1 supera al baseline en PESQ-NB
  por 0,167 puntos (2,330 contra 2,163). Es el margen más chico de toda la
  matriz, y aparece exactamente donde el modelo entrenado sólo en inglés está en
  su peor terreno. No cambia el veredicto de OP-6, pero es el dato que hace
  concreto por qué el fine-tuning por idioma no es opcional: sin él, el CRN se
  acerca al rendimiento de un pasabajo de orden 5.
- **Falta**: nada.

---

## 5.2 V1 — Escalado del dataset

- **Pregunta**: ¿cuánto del colapso de V0 era falta de datos?
- **Diseño**: 50.000 pares de entrenamiento + 2.000 de validación, LibriSpeech
  train-clean-100 con MUSAN y ESC-50. 30 épocas, misma loss que V0.
- **Resultado**: PESQ-NB 2,650 (Δ +0,497 sobre ruidoso), PESQ-WB 2,021,
  STOI 0,904, SI-SDR 13,80 dB. Mejor época: 19.
- **Fuente**: `results/v1_v1_en.json`; `results/v1_training_cost.json`.
- **Qué queda**: salto de +1,23 puntos PESQ sobre V0. Ganancia dual: volumen de
  datos y ruptura del colapso. **No es extrapolable linealmente** — la literatura
  muestra saturación logarítmica (Reddy 2020, Kolbæk 2020, Schröter 2022).
- **Reserva a declarar**: V1 no estaba del todo convergido (§4.7). Y se entrenó
  sin determinismo de cuDNN (§4.1).

## 5.3 V2 — Loss combinada MSE + SI-SDR

- **Pregunta**: ¿aporta información complementaria un término en dominio temporal?
- **Diseño**: única variable cambiada respecto de V1 — loss 0,7·MSE + 0,3·SI-SDR.
  20 épocas, mejor época 19.
- **Resultado**: PESQ-NB 2,849 (+0,199 sobre V1), PESQ-WB 2,211, STOI 0,908,
  SI-SDR 14,44 dB (+0,64 dB).
- **Fuente**: `results/v2_v1_en.json`; `results/v2_training_cost.json`.
- **Qué queda**: ganancia mayor en SNR alto — el término SI-SDR ayuda al
  procesamiento quirúrgico donde hay poco que remover. Confirmación empírica de
  Braun & Tashev 2021 §4.3. V2 es la baseline de todo lo que sigue.
- **Falta**: la justificación del reparto 0,7/0,3 (ver §2.4).

## 5.4 V3 — Fine-tuning agresivo EN→ES

- **Pregunta**: ¿el fine-tuning al español mejora en español, y cuánto cuesta en
  inglés?
- **Diseño**: full fine-tuning desde V1, mismo lr 2e-4, 30 épocas.
- **Resultado**: en `test_v2_es` PESQ-NB 2,560 (+0,230 sobre V1); en
  `test_v1_en` 2,571 (−0,079 sobre V1).
- **Fuente**: `results/v3_v2_es.json`, `results/v3_v1_en.json`.
- **Qué queda**: el fine-tuning funciona y el olvido es real. Consistente con
  Yosinski 2014 y McCloskey & Cohen 1989.
- **Falta**: nada. Sección ya escrita en `EXPERIMENTS.md`.

## 5.5 V3b — Learning rate conservador (hipótesis no sostenida)

- **Pregunta**: ¿un lr conservador mejora el balance entre adaptación y olvido?
- **Diseño**: lr 5e-5 elegido por sweep de 5 candidatos; 10 épocas.
- **Resultado**: +0,153 en español, −0,029 en inglés. PESQ-NB 2,483 (ES) y
  2,621 (EN).
- **Fuente**: `results/v3b_v2_es.json`, `results/v3b_v1_en.json`;
  `scripts/lr_sweep_v3b.py`.
- **Qué queda**: la hipótesis **no se sostuvo**. Redujo el olvido pero también la
  ganancia. Causa identificada: 10 épocas contra las 30 de V3 — épocas totales y
  agresividad del lr quedaron confundidas en el diseño (§4.8).
- **Falta**: sección en `EXPERIMENTS.md`.
- **Prohibido**: el "score compuesto" que ordenaba V3 contra V3b está retirado.
  Los dos ejes se reportan por separado.

## 5.6 V3e — Desconfundir épocas y learning rate

- **Pregunta**: si se desconfunden las dos variables, ¿se puede tener la ganancia
  de V3 con el olvido de V3b?
- **Diseño**: lr 1e-4 —el candidato del sweep que no había convergido a 5 épocas—
  extendido a 25 épocas con decay tardío en la 12, no en la 4.
- **Resultado**: +0,221 en español, **−0,031** en inglés. PESQ-NB 2,551 (ES),
  2,619 (EN), 2,686 (MLS).
- **Fuente**: `results/v3e_{v1_en,v2_es,v3_mls_es}.json`. Tag `v3e.0.0`.
- **Qué queda**: checkpoint recomendado para V5. `checkpoints/v3e/best.pt`.
- **Falta**: sección en `EXPERIMENTS.md`, con la narrativa completa de las tres
  iteraciones y la discusión de por qué V3b no superó a V3.
- **Prohibido**: el −0,032. El valor correcto es −0,031.

## 5.7 V4 y V4b — Proxy perceptual

- **Resumen acá, desarrollo completo en el capítulo 6.**
- **Pregunta**: ¿se puede optimizar contra un proxy diferenciable de PESQ?
- **Resultado**: no con el proxy congelado. Gaming desde la época 1 en las seis
  corridas con término perceptual. El protocolo estabilizado de Xu recupera un
  tercio del daño; dos tercios sobreviven.
- **Fuente**: `decisions.md`, "Cierre de V4" (31/08) y "V4b" (01/09).
  Tags `v4.0.0`, `v4b.0.0`.
- **Falta**: secciones en `EXPERIMENTS.md`.

## 5.8 V5 — La propuesta completa, y sus réplicas

- **Pregunta**: ¿cuánto rinde la mejor receta de fine-tuning aplicada sobre la
  mejor baseline?
- **Diseño**: V2 fine-tuneado a español con la receta de V3e. Réplicas con
  semillas 43 y 44 desde el mismo checkpoint de V2.
- **Resultado** (PESQ-NB, semillas 42/43/44):
  español 2,686 / 2,658 / 2,661 (sd 0,015);
  inglés 2,774 / 2,770 / 2,766 (sd 0,004);
  audiolibro 2,840 / 2,803 / 2,816 (sd 0,019).
- **Fuente**: `results/v5_{,s43_,s44_}{v1_en,v2_es,v3_mls_es}.json`.
- **Obligatorio al reportar**: **la semilla 42 es la más alta de las tres en los
  tres sellados.** El titular sale del máximo de tres corridas y hay que decirlo
  en la misma tabla donde aparece el número.
- **Qué queda, y qué no**: las réplicas miden la varianza del fine-tuning —orden
  de los datos—, **no** la de inicialización: las tres parten del mismo
  checkpoint. No habilitan afirmar nada sobre robustez a la inicialización.
- **Fuente de esa distinción**: `decisions.md`, 08-09/09.
- **Barrido de época**: el pico está en la 18, no en la de mínima `val_loss`.
  Cuarta evidencia del mismo problema de selección (§4.7). **No se seleccionó
  nada con esto**: está medido sobre el sellado.
- **Falta**: sección en `EXPERIMENTS.md`.

## 5.9 V6 — Compuerta de paso directo desde V2 (negativo)

- **Pregunta**: ¿una compuerta convexa y causal por banda le da a la red un
  camino de identidad que evite degradar audio ya limpio?
- **Diseño**: `M_out = g·M̂ + (1−g)·M_noisy`, 193 parámetros sobre 17.579.459
  (+0,0011 %). Tratamiento/placebo, 6 épocas de fine-tuning desde V2, entrenando
  sólo en inglés.
- **Resultado**: **el endpoint primario falla** — +2,5 pp contra un umbral de
  +3 pp, p = 0,181. El mecanismo se cumple: ρ(g, SNR) ≈ −0,4, p < 1e−8 en los
  tres sellados. P3 falla por un error de signo del propio preregistro.
- **Fuente**: `docs/v6_compuerta.md`; tag `7bcd30b`.
- **Qué queda**: por la regla preregistrada, la hipótesis no queda sostenida. Y
  el subproducto vale más que el veredicto: **el piso de ruido por checkpoint**
  (§4.6).
- **Prohibido**: el costo en inglés de −0,011 con p = 0,005. Retractado.

## 5.10 V7 — La misma compuerta desde la inicialización

- **Pregunta**: ¿el problema de V6 era la compuerta, o haberla agregado tarde
  sobre una red ya entrenada sin ella?
- **Diseño**: compuerta presente desde la inicialización, 20 épocas desde cero
  contra un control desde cero.
- **Resultado**: los cuatro endpoints preregistrados pasan. E1 = +0,0795 sobre un
  umbral de +0,050; E2 = +0,0620 sin costo en inglés; E3 con ρ de −0,35/−0,42/−0,40;
  E4 con media de g entre 0,42 y 0,51. E5: desde cero rinde 4,3× lo de V6.
- **Fuente**: `results/v7_endpoints.json`; `docs/v7_compuerta_desde_cero.md`.
- **El dato que le pega a la hipótesis central**: en el bucket [15,20] dB del
  español, el control desde cero **replica la patología de V1 y V2** —degrada
  audio que ya estaba limpio— y la compuerta la cruza a cero.
- **Tres reservas explícitas** (`v7_compuerta_desde_cero.md` §6): una sola semilla
  por brazo; el mecanismo propuesto no es la única explicación en pie; la
  dispersión entre épocas es más ancha de lo que V6 había medido.
- **Falta, y bloquea el cierre del capítulo**: la confirmación con tres semillas
  (~45 h de GPU). La infraestructura está lista (`CONFIG_V7_{GATE,CONTROL}_S{43,44}`,
  `scripts/run_v7_seeds.sh`). **El preregistro no está cerrado ni hasheado**, y
  tiene una decisión sin resolver: si la semilla 42 entra o no en el estimando
  confirmatorio habiendo sido la que disparó la confirmación.
- **Si no se corre**: V7 se escribe como screening positivo con sus tres reservas,
  no como aporte arquitectónico. Es defendible; lo que no lo es, es escribirlo
  como aporte sin la confirmación que el propio preregistro exige.

## 5.11 Tabla consolidada

- **Afirma**: una tabla con las ocho variantes de la línea principal sobre los
  tres sellados, cuatro métricas, con deltas contra ruidoso.
- **Fuente**: `FUENTES.md`, tabla de PESQ-NB; los JSON para el resto.
- **Regla de construcción**: sólo la línea principal del ablation (ruidoso, V1,
  V2, V3, V3b, V3e, V5). **V6 y V7 no entran**: su estimando es un contraste
  contra su propio control, no un valor absoluto comparable. Es el mismo criterio
  con que se construyó el dashboard.
- **Falta**: generar la tabla. `scripts/generate_dashboard.py` ya tiene la lógica.
