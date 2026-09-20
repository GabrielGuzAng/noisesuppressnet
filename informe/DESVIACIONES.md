# Anteproyecto comprometido contra proyecto ejecutado

Insumo transversal. El capítulo 1 lo usa para el estado de cada objetivo; el
capítulo 8 lo usa para el desvío contra la línea base de gestión.

Fuente del lado comprometido: `DocsProfesores/Anteproyecto Gabriel Guzmán.pdf`,
defendido el 30/06/2026 (secciones 2.1, 2.4, 3.1, 3.2, 6.2, 6.3, 6.4).
Fuente del lado ejecutado: `docs/decisions.md`, `docs/EXPERIMENTS.md`,
`results/*.json`, `training/config.py`.

Un tribunal no penaliza el desvío; penaliza el desvío no declarado. Cada fila de
abajo va al informe con su justificación, no se omite ninguna.

---

## D1 — El eje de loss perceptual se cerró como resultado negativo

**Comprometido**: el anteproyecto declara dos ejes de innovación (sección 2.1), y
el segundo es el uso de PESQNet (Xu et al., 2022) como pérdida perceptual
diferenciable. El hito H6 congela "V4 — PESQNet loss sobre dataset inglés" y el
H7 congela "V5 — propuesta completa (español + PESQNet) ★". La estrella es del
original: V5 era el entregable central del proyecto.

**Ejecutado**: PESQNet se sustituyó por TorchAudio-Squim (Kumar et al., 2023) por
disponibilidad de implementación. V4 (31/08) y V4b (01/09) caracterizaron el
gaming del proxy y la línea quedó cerrada. V5 se redefinió: es V2 fine-tuneado a
español con la receta de V3e, **sin término perceptual**.

**Qué decir**: uno de los dos ejes comprometidos no se sostuvo empíricamente. El
proyecto lo midió con un diseño 2×2 más placebo, lo caracterizó y lo cerró. Eso
es el capítulo 6, y es contenido evaluable, no una falla de gestión. Lo que sí
hay que decir explícito es que **la propuesta central del anteproyecto cambió de
contenido manteniendo el nombre V5**.

**Riesgo de tribunal**: alto. Es la pregunta más probable de toda la defensa.
Conviene que el capítulo 1 lo declare en la primera página, no que aparezca
recién en el 6.

---

## D2 — El KPI propio cambió de definición

**Comprometido**: sección 6.3, KPI marcado con estrella —
`Pearson r (loss, PESQ)`, "alineación loss/percepción — KPI PROPIO", con umbral
de alerta `r < 0,3 al final del entrenamiento`.

**Ejecutado**: el diagnóstico que terminó rindiendo es la **brecha
`pesq_hat − PESQ-WB real`** contra el MAE nominal de Squim (0,142; Kumar 2023,
Tabla 2). Brechas observadas de +2,36 y +2,77, entre 16× y 20× ese error.

**Qué decir**: el KPI propio existe y funcionó —detectó el gaming desde la
época 1 en las seis corridas con término perceptual—, pero su forma operativa no
es la correlación de Pearson que se prometió. Hay que documentar la transición y
por qué la brecha absoluta es mejor diagnóstico que la correlación: una
correlación alta es compatible con un sesgo constante grande, que es exactamente
la firma del gaming.

**Pendiente de verificar antes de redactar**: si `evaluation/monitor_correlation.py`
llegó a producir valores de Pearson r reportables. Si los hay, entran como
evidencia de la transición.

---

## D3 — El dataset en español no es el comprometido

**Comprometido**: sección 2.1, "fine-tuning de una arquitectura pre-entrenada en
inglés a un dataset validado en español (DNS-Challenge)".

**Ejecutado**: Common Voice ES v26 (1.680.810 clips, 2.278,7 h) para
entrenamiento, y Multilingual LibriSpeech ES para el tercer sellado.

**Qué decir**: el cambio habilitó algo que el plan original no tenía — separar
idioma de canal. Common Voice es habla leída de micrófono heterogéneo; MLS es
audiolibro, mismo canal que LibriSpeech. Con los dos sellados en español se
responde la objeción de Wang et al. 2022 sobre confusión idioma/canal, que con un
solo dataset en español no se podía responder. El desvío mejoró el diseño.

**Costo del cambio**: ~23 % de los clips de Common Voice miden menos de 4 s, lo
que hace que `pad_or_crop` loopee el clip en vez de recortar. Es una diferencia
real contra LibriSpeech y va declarada como limitación (`decisions.md`,
18/08/2026).

---

## D4 — El test set sellado tiene 250 pares, no 300, y hay tres

**Comprometido**: sección 6.4, "test set sellado (300 pares) generado una única
vez y no regenerable".

**Ejecutado**: tres sellados de 250 pares cada uno.

| sellado | fecha | contenido | hash SHA-256 (prefijo) |
|---|---|---|---|
| `test_v1_en` | 26/07/2026 | LibriSpeech EN | `8eb360dc…` |
| `test_v2_es` | 20/08/2026 | Common Voice ES | `96f33f83…` |
| `test_v3_mls_es` | 06/09/2026 | MLS ES, condiciones de ruido de v1 copiadas verbatim | `fa99a0fa…` |

**Qué decir**: la regla de fondo —sellar una vez, no regenerar— se cumplió en los
tres. El número bajó de 300 a 250 y la cantidad de conjuntos subió de uno a tres;
750 pares sellados en total contra los 300 comprometidos. Justificar el 250: cinco
buckets de SNR × 50 pares, que es lo que fija la potencia por bucket del análisis
estratificado.

---

## D5 — Se ejecutaron ocho variantes, no cinco

**Comprometido**: sección 3.1, "cinco variantes experimentales (V1 a V5)".

**Ejecutado**: V0, V1, V2, V3, V3b, V3e, V4, V4b, V5 (con réplicas de semilla 43 y
44), V6 y V7 (con brazo de control propio en cada uno). Trece tags en el
repositorio.

**Qué decir**: el ablation planificado se cumplió y además se abrieron dos líneas
que el plan no tenía: la iteración de recetas de fine-tuning (V3/V3b/V3e) y la
compuerta de paso directo (V6/V7). Corresponde justificar por qué se financiaron
con horas del proyecto, y decir qué se sacó del plan para pagarlas — GCRN quedó
descartado explícitamente el 06/09.

---

## D6 — La causalidad estricta tiene 10 ms de lookahead

**Comprometido**: sección 1.1, el sistema "no utiliza información del futuro".
Objetivo de calidad OP-5: causalidad estricta, `diff = 0.00e+00`.

**Ejecutado**: la causalidad a nivel de frame es bit-exacta y `test_causality.py`
la verifica. Pero la STFT usa `center=True`, lo que agrega 10,0 ms de lookahead a
nivel de señal. Medido, no estimado.

**Qué decir**: esta es la corrección que más conviene que haga el autor antes de
que la pregunte el tribunal. La latencia algorítmica es de 10 ms, no cero; sigue
siendo compatible con tiempo real, y el enunciado correcto es "causal a nivel de
frame con 10 ms de latencia algorítmica", no "sin lookahead". El OP-5 se cumple
en lo que mide el test; el enunciado del abstract del anteproyecto es el que hay
que corregir.

---

## D7 — Faltan métricas que estaban dentro del alcance

**Comprometido**: sección 3.1, métricas intrusivas PESQ, STOI, SI-SDR y **SegSNR**,
y no intrusivas **DNSMOS** y TorchAudio-Squim.

**Ejecutado**: `evaluation/metrics.py` implementa PESQ-NB, PESQ-WB, STOI y SI-SDR.
Squim se usó como término de loss y como objeto de estudio del gaming, no como
métrica de reporte. SegSNR y DNSMOS no están.

**Decisión pendiente**: implementarlas y re-evaluar las ocho variantes sobre los
tres sellados, o declarar la reducción de alcance. El costo de re-evaluar no es
despreciable pero tampoco prohibitivo; la decisión es de Gabriel.

---

## D8 — Falta el benchmark contra RNNoise y DeepFilterNet2

**Comprometido**: sección 3.1, dentro del alcance, explícito con cita de
Valin 2018 y Schröter et al. 2022.

**Ejecutado**: nada. Figura en el pendiente de mediano plazo de `CLAUDE.md`.

**Qué decir**: es el punto de alcance incumplido más visible para un tribunal,
porque es lo que ubica el trabajo contra el estado del arte publicado. Si no se
corre, la justificación tiene que ser mejor que "no hubo tiempo".

---

## D9 — Hitos: el cronograma se cumplió, con una excepción de contenido

| Hito | Fecha plan | Fecha real | Desvío |
|---|---|---|---|
| H1 preproyecto defendido | 30/06/2026 | 30/06/2026 | en fecha |
| H2 pipeline listo | 22/07/2026 | — | verificar |
| H3 V1 congelada | 31/07/2026 | 02/08/2026 (reporte de costo) | +2 d |
| H4 V2 congelada | 11/08/2026 | 06/08/2026 | −5 d |
| H5 V3 congelada | 20/08/2026 | 21/08/2026 | +1 d |
| H6 V4 congelada | 30/08/2026 | 31/08/2026 | +1 d, **resultado negativo** |
| H7 V5 congelada ★ | 15/09/2026 | 09/09/2026 | −6 d, **contenido distinto (D1)** |
| H8 evaluación completa | 10/10/2026 | en curso | — |

**Qué decir**: el cronograma se sostuvo con desvíos de días, no de semanas, y dos
hitos se adelantaron. El desvío relevante no es de tiempo sino de contenido: H7
se congeló en fecha pero con una V5 que no es la que el hito define. Eso se dice
en el capítulo 8 y se cruza con D1.

---

## D10 — Documentos de gestión faltantes o inconsistentes

- El anteproyecto (sección 3.3) cita **"EDT v3 — NoiseSuppressNet"** con 65
  paquetes de trabajo. En `DocsProfesores/` sólo hay `EDT_v2_NoiseSuppressNet.docx`.
- `CLAUDE.md` lista la presentación de la defensa del preproyecto entre los
  documentos entregados. No está en el directorio.
- Hay dos documentos en `DocsProfesores/` que ninguna fuente del proyecto
  menciona: `Gestion_Tiempos.docx` y `NoiseSuppressNet_Descripcion_FODA (1).docx`.
  Hay que ver qué son y si entran al capítulo 8 o a anexos.

**Bloquea**: capítulo 8 completo.
