# Ledger de afirmaciones

Insumo transversal. **No es un resumen del proyecto: es una tabla de decisiones.**
Cada fila fija qué se puede afirmar, con qué nivel de evidencia, de qué archivo
sale el número, con qué hedge se escribe y a qué documento va.

**Creado**: 19/09/2026. **Revisado**: 20/09/2026 (reconciliación de E7, veredicto
sobre D-1, escalamiento de D-7, y la calibración de hedging de §4.1.1 del informe).
Rol: `paper_writer` (ROLES.md).
**Regla de verificación aplicada**: cada número de esta tabla se leyó del archivo
en `results/`, `seal_test_metadata/` o del código, no del documento que lo cita.
Donde documento y archivo discrepan, la fila lleva el del archivo y la
discrepancia queda anotada en la sección 11. Se encontraron **doce**, y una
contradicción de diseño que no es documental y tiene bloque propio (§4.b).

---

## 1. Escala de niveles

| nivel | qué significa | cómo se enuncia |
|---|---|---|
| preregistrado y confirmado | hipótesis y umbral escritos y hasheados antes del dato, fuera de muestra | afirmación directa |
| preregistrado y fallado | el criterio literal no se cumplió | se reporta el fracaso, no se reformula el criterio |
| screening | preregistrado con n=1 por brazo, o umbral grueso a propósito | "consistente con", "sugiere", nunca "demuestra" |
| exploratorio | la hipótesis salió de estos mismos datos | "generación de hipótesis"; requiere confirmación fuera de muestra |
| descriptivo | patrón sin test, o con checkpoints no independientes | se muestra el patrón, sin p-valor de respaldo |

La escala no tiene nivel para una verificación determinista (un test que pasa o
falla sin inferencia). Esas filas van como **descriptivo** y llevan la aclaración
de que no corresponde p-valor por construcción.

## 2. Regla de lectura de los p-valores, y va en el cuerpo del paper

Todos los p-valores apareados de este proyecto —F1 a F5, los de V5, los de V2
contra V1— se calculan **sobre archivos dentro de un único par de checkpoints**.
Miden si el efecto es consistente entre los 250 archivos del sellado. **No miden
si volver a entrenar reproduce el efecto**: para eso hacen falta réplicas de
semilla, y sólo V5 las tiene (n=3, y miden orden de datos, no inicialización).

Consecuencia de redacción: `p = 2e−41` no habilita "el fine-tuning al español
mejora de forma robusta", habilita "en este par de checkpoints, el 90 % de los
archivos mejora". La diferencia es la que separa un resultado de un titular.

## 2.b Alcance de reproducibilidad de todo número de este ledger

`evaluation/evaluate_variant.py` no activa el determinismo de cuDNN —el trainer sí,
desde V3b—, así que **ningún JSON de `results/` es bit-reproducible**: re-evaluar la
misma variante, con el mismo checkpoint y el mismo código, da un archivo distinto
(informe §4.1.1, medido el 20/09/2026; la variación entre dos corridas del mismo
código es indistinguible de la variación contra el archivo commiteado).

**Calibración que se aplica a todas las filas**: donde este ledger dice que un
número está medido, el alcance es **reproducible a la precisión reportada (tres
decimales)**, no bit-exacto. Las diferencias medidas están entre tres y cinco
órdenes de magnitud por debajo de esa precisión y muy por debajo del piso de ruido
por checkpoint, así que no se mueve ninguna conclusión; lo que no se puede escribir
es "bit-reproducible" sobre un resultado de evaluación.

Una distinción que hay que mantener, porque las dos afirmaciones conviven: las
verificaciones deterministas de A1, A2 y A5 **corren en CPU**, donde el resultado es
exacto y por eso el test puede exigir `diff = 0,00e+00`. La evaluación corre en GPU,
donde el forward del mismo modelo sobre el mismo audio difiere entre 3e−08 y 7e−07.
No hay contradicción: son dos caminos de cómputo distintos, y conviene decirlo así.

## 3. Exclusiones de este ledger

| qué | por qué | fila |
|---|---|---|
| V7 y la compuerta de paso directo (V6 y V7) | la confirmación con tres semillas está corriendo y no hay resultado | X1, única |
| RNNoise y DeepFilterNet2 | no corridos | ninguna; ver §13 |
| DNSMOS y SegSNR | no implementadas en `evaluation/metrics.py` | ninguna; ver §13 |
| ASR / WER downstream | no corrido | ninguna; ver §13 |
| Cualquier afirmación de estado del arte que requiera esos números | no hay con qué sostenerla | ninguna |

No hay filas condicionales ("si se corriera, diría"). Lo que no está medido no
entra, ni siquiera como hipótesis con hedge.

---

## 4. Bloque A — Sistema causal y eficiencia

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| A1 | La red es causal a nivel de frame de forma bit-exacta: modificar la entrada en t≥50 no cambia ninguna salida anterior (`diff = 0,00e+00`). | descriptivo (verificación determinista; sin p-valor por construcción) | `tests/test_causality.py`, 24 tests, **corridos y pasados el 19/09/2026** | "La causalidad a nivel de frame se verifica bit-exacta: la diferencia medida es 0,00e+00." | los dos |
| A2 | La latencia algorítmica del sistema es de 10,0 ms (160 muestras = `n_fft − hop`), impuesta por el solapamiento de la síntesis y no por el modo de framing del análisis: es idéntica con padding causal, verificado sobre cinco configuraciones de `n_fft`/`hop`. | descriptivo (verificación determinista) | `tests/test_causality.py::test_signal_lookahead_equals_nfft_minus_hop` y el test que exige 160 muestras ≡ 10,0 ms; `decisions.md` 07/09/2026 | "El sistema es causal a nivel de frame con 10,0 ms de latencia algorítmica, medidos. La configuración STFT del paper base tiene la misma latencia." | los dos |
| A3 | El sistema corre en tiempo real con margen de 3× sobre la CPU objetivo: RTF 0,331 mediana y 0,340 p95 en un i5-4460 monohilo; 0,0148 / 0,0151 en RTX 4060. | descriptivo | `EXPERIMENTS.md` §V0 (20 corridas, 5 de warm-up). **Sin artefacto en `results/`**: `benchmarks/measure_rtf.py` existe pero su salida no quedó guardada | "Sobre un Intel i5-4460 monohilo el RTF es 0,331 en mediana y 0,340 en p95, sobre 20 corridas." | los dos |
| A4 | El modelo tiene 17.579.459 parámetros, que es el orden que reporta la Fig. 5 de Tan & Wang 2018 (17,58 M). | descriptivo | `models/crn.py`, **recontado el 19/09/2026** con `sum(p.numel())` | "La implementación tiene 17.579.459 parámetros, coincidentes en el orden reportado por el paper base." | los dos |
| A5 | La STFT es invertible dentro del error numérico: error de round-trip 9,54e-07. | descriptivo (verificación determinista) | `tests/test_stft.py`, **reejecutado el 19/09/2026**: reproduce 9.54e-07. Es un módulo ejecutable, no recolecta tests bajo pytest, así que no entra en ninguna corrida de la suite | "El round-trip STFT tiene un error de 9,54e-07, por debajo del umbral de 1e-5." | los dos |

**Prohibido en este bloque**: "sin lookahead", "latencia cero", "no utiliza
información del futuro". Son 10,0 ms medidos (ver A2 y `DESVIACIONES.md` D6).

## 4.b Bloque M — Validez del material de evaluación (transversal, y es lo más grave que encontró esta revisión)

Salió de verificar la condición que pedía D-1 —demostrar que no hay solapamiento
entre entrenamiento y sellado— y no es un problema de V0: es del sellado en inglés.

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| M1 | **`test_v1_en` no es disjunto del material de entrenamiento en inglés.** `scripts/seal_test_set_en.py` (línea 126) y `datasets/make_mixtures.py` (línea 180) recolectan la voz del **mismo directorio**, `data/raw/clean_speech/en/librispeech/LibriSpeech`, que contiene únicamente `train-clean-100` (28.539 clips, 251 hablantes), y sortean con `random.choice` sin exclusión ni split. Los 250 clips sellados y sus 159 hablantes salen de ese pool. | descriptivo (verificación determinista sobre código y metadata, 20/09/2026) | las dos líneas de código citadas; `seal_test_metadata/test_v1_metadata.json` (250 clips distintos, 159 hablantes, todos bajo `train-clean-100/`); conteo del pool en disco | "El conjunto sellado en inglés se construyó sobre el mismo subconjunto de LibriSpeech que las mixturas de entrenamiento, sin exclusión de archivos ni de hablantes." | los dos, en metodología **y** en limitaciones |
| M2 | Bajo el modelo de sorteo que el propio código implementa, la exposición esperada es del **83,8 %** de las emisiones selladas para V1 y V2 (≈210 de 250) y del **100 %** de los 159 hablantes; para V0, de **0,87 %** (≈2,2 de 250 clips) y ~100 de 159 hablantes. | descriptivo (cálculo cerrado sobre el pool real, no medición) | recomputado el 20/09: `1 − (1 − 1/28539)^n` con n = 52.000 (V1/V2) y n = 250 (V0); exposición por hablante con la distribución real de clips por hablante | "La exposición no se puede verificar par por par —las mixturas no guardan procedencia— pero sí acotar: con 52.000 extracciones sobre 28.539 clips, se espera que el 84 % de las emisiones selladas haya aparecido en entrenamiento." | los dos |
| M3 | **La contaminación no es simétrica entre idiomas.** Los dos sellados en español sí son disjuntos: `test_v2_es` sale del split oficial `test` de Common Voice, con 0 hablantes y 0 frases en común con `train`/`dev` verificado; `test_v3_mls_es` sale de Multilingual LibriSpeech, corpus que nunca se usó para entrenar. | descriptivo | `datasets/make_mixtures.py` líneas 194-225 (manifests disjuntos); `decisions.md` 17/08; `seal_test_metadata/test_v3_mls_es_metadata.json → source` | "El material de evaluación en español es disjunto del de entrenamiento; el de inglés no lo es. Toda comparación entre idiomas lleva incorporada esa asimetría." | los dos |

**Qué sobrevive y qué no.** Hay que escribirlo con esta granularidad, porque la
tentación de las dos puntas —"no invalida nada" y "invalida todo"— es falsa:

- **No se ve afectado**: ningún contraste entre variantes evaluadas sobre el mismo
  sellado. La contaminación es idéntica para los dos brazos de cualquier
  comparación, así que B3, C1, C2, F1 a F5 y todo el bloque E mantienen su
  estimando. Es la misma lógica por la que el placebo de V4b funciona.
- **Queda como cota superior**: todo valor absoluto en inglés (B2, C3 en su columna
  de inglés, y los márgenes de OP-6 en `test_v1_en`).
- **Se cae**: cualquier afirmación de generalización a hablantes o emisiones no
  vistas en inglés. El proyecto no midió eso y no puede afirmarlo.
- **Queda declarado como confusor de la afirmación central**: el contraste
  inglés/español compara un sellado contaminado contra dos limpios. La pendiente
  plana contra el SNR en inglés (E1, ρ = −0,065) admite como explicación
  alternativa que el modelo ya vio esas emisiones. **D1 es el que menos sufre**
  —contrasta dos sellados en español entre sí, los dos limpios, con las condiciones
  de ruido idénticas— y por eso sigue siendo la fila fuerte; pero E1, que apoya la
  interacción sobre el eje inglés, baja de grado.

**Reparación, con su costo.** Sellar un cuarto conjunto en inglés desde un split
disjunto —`dev-clean` o `test-clean`, que **no están descargados**: en disco sólo
hay `train-clean-100`— y reevaluar las variantes sobre él. Cuesta una descarga de
~350 MB, un sellado y ~8 evaluaciones de 250 pares; nada de entrenamiento. Es la
única forma de convertir la cota superior en una medición. No se puede reparar
hacia atrás: reconstruir qué clips vio V1 es imposible, porque las mixturas no
guardan procedencia y `make_mixtures.py` no sembró `random` hasta el 19/08/2026.

## 5. Bloque B — Ablation en inglés (V0 → V1 → V2)

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| B1 | Con 200 pares de entrenamiento el modelo no supera al audio sin procesar en ninguna de las tres métricas medidas, y colapsa a la media del dataset (ratio `std_out/std_target` = 0,30). | descriptivo | `results/summary_metrics.csv` (PESQ-NB 1,417 vs 2,050 del ruidoso; STOI 0,794 vs 0,842; SI-SDR 3,11 vs 5,13 dB), sobre **el val set de 50 clips de V0, no sobre sellado**; PSD en `EXPERIMENTS.md` §V0 | "En el régimen de 200 pares el modelo colapsa: no supera al ruidoso en ninguna métrica sobre su conjunto de validación de 50 clips." | los dos |
| B2 | Con 50.000 pares el modelo supera al audio ruidoso en las cuatro métricas del sellado en inglés: PESQ-NB 2,650 (+0,497), PESQ-WB 2,021 (+0,437), STOI 0,904 (+0,051), SI-SDR 13,80 dB (+5,95). | descriptivo (un checkpoint) | `results/v1_v1_en.json → global.*` | "Escalado a 50.000 pares, el modelo mejora las cuatro métricas sobre el sellado de 250 pares en inglés." | los dos |
| B3 | La loss combinada MSE + SI-SDR aporta sobre MSE puro: +0,200 PESQ-NB y +0,64 dB SI-SDR sobre `test_v1_en`, con mejora en las cuatro métricas y en el 90,4 % de los 250 archivos. | descriptivo (un checkpoint por brazo, sin placebo) | `results/v2_v1_en.json` y `v1_v1_en.json → global`; fracción y test apareado recomputados de `all_pairs` el 19/09 (Wilcoxon p = 1,2e−37) | "Con un checkpoint por brazo, la loss combinada mejora 0,200 puntos de PESQ-NB sobre MSE puro, y mejora el 90,4 % de los archivos del sellado." | los dos |
| B4 | La selección de checkpoint por mínimo de `val_loss` MSE no está alineada con PESQ: V1 más dos épocas a lr 2e-5 con la misma loss da 2,716 / 2,102 / 0,906 / 13,95, mejor en las cuatro métricas, con peor `val_loss` (0,0836 contra 0,0724). | descriptivo | `results/v4b/v4b_placebo_epoch_02_v1_en.json → global`; `v1_v1_en.json`; `decisions.md` 01/09 | "Los valores reportados de V1 son levemente pesimistas: dos épocas más con la misma loss mejoran las cuatro métricas pese a empeorar la pérdida de validación." | los dos (limitaciones) |

| B5 | Todas las variantes del CRN superan al baseline pasabajo Butterworth de 4 kHz en PESQ-NB y STOI sobre los tres sellados, contra el mejor de sus dos modos en cada métrica; el margen va de +0,167 (V1 sobre español crowdsourced) a +0,687 (V2 sobre inglés). El pasabajo es una operación nula sobre PESQ: +0,001 a +0,011 contra el ruidoso. | descriptivo | `results/butterworth_{causal,zerophase}_{v1_en,v2_es,v3_mls_es}.json`, corridos el 20/09; márgenes recomputados el 20/09 contra los ocho JSON de variante | "Sobre los tres conjuntos sellados, todas las variantes superan al pasabajo en PESQ y en STOI. El pasabajo no mejora el PESQ del audio ruidoso en ninguno de sus dos modos: lo único que hace de forma apreciable es costar SI-SDR." | los dos |

**No ledgereable**: el salto "V0 → V1 = +1,23 PESQ". Compara V0 sobre su val de
50 clips contra V1 sobre el sellado de 250. No es el mismo estimando ni el mismo
material. Ver §11, D-1.

## 6. Bloque C — Adaptación al español (V3 / V3b / V3e / V5)

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| C1 | El fine-tuning sobre español mejora el PESQ-NB medido sobre español: +0,230 (V3), +0,221 (V3e) y +0,152 (V3b) contra V1 sobre `test_v2_es`. | descriptivo (un checkpoint por variante) | `results/v3_v2_es.json`, `v3e_v2_es.json`, `v3b_v2_es.json` contra `v1_v2_es.json → global.pesq_nb.est_mean` | "Las tres recetas de fine-tuning al español mejoran el PESQ-NB sobre el sellado en español, entre +0,152 y +0,230." | los dos |
| C2 | Ese fine-tuning tiene un costo medio en inglés de −0,079 (V3), −0,031 (V3e) y −0,028 (V3b) sobre `test_v1_en`. | descriptivo | mismos JSON contra `v1_v1_en.json` | "El costo medio en inglés va de −0,028 a −0,079 puntos de PESQ-NB según la agresividad del learning rate." | los dos |
| C3 | V5 —V2 fine-tuneado a español con la receta de V3e— da PESQ-NB 2,686 en español crowdsourced, 2,774 en inglés y 2,840 en español de audiolibro; mejora +0,136 sobre V3e en español y +0,235 sobre V2. | descriptivo, **con advertencia de semilla obligatoria** | `results/v5_{v1_en,v2_es,v3_mls_es}.json`; contrastes apareados recomputados el 19/09 (Wilcoxon p = 2,3e−17 contra V3e, 9,8e−30 contra V2) | "V5 alcanza 2,686 de PESQ-NB en español. El valor corresponde a la semilla 42, que es la más alta de las tres corridas, y se reporta junto con las otras dos." | los dos |
| C4 | La varianza entre semillas del fine-tuning es sd 0,015 (español crowdsourced), 0,004 (inglés) y 0,019 (español audiolibro), sobre n=3; las tres corridas parten del mismo checkpoint, así que miden el orden de los datos y no la inicialización. | descriptivo | `results/v5_{,s43_,s44_}*.json`; sd muestral recomputada el 19/09 | "Las tres réplicas acotan la varianza de orden de datos en 0,004 a 0,019 puntos. No acotan la varianza de inicialización, que no se midió." | los dos |
| C5 | V5 cruza el criterio mínimo OP-1 del plan de calidad (PESQ-NB ≥ 2,5) en los tres sellados. | descriptivo | los tres JSON de V5; criterio en anteproyecto §6.2 | "V5 supera el criterio mínimo comprometido de PESQ ≥ 2,5 en los tres conjuntos sellados." | informe |
| C6 | V5 pierde −0,075 contra su propio punto de partida en inglés, con 16,8 % de archivos que caen más de 0,2, y gana +0,125 contra V1 en inglés, con 9,2 %. | descriptivo | `v5_v1_en.json` contra `v2_v1_en.json` y `v1_v1_en.json`; fracciones recomputadas de `all_pairs` el 19/09 (Wilcoxon p = 0,0039 y 1,3e−15) | "V5 queda por debajo de V2 en inglés en 0,075 puntos, y por encima de V1 en 0,125. Las dos comparaciones son ciertas y responden a preguntas distintas." | los dos |
| C7 | La hipótesis de V3b —un learning rate conservador mejora el balance entre adaptación y olvido— no se sostuvo, y su diseño no permite atribuir: 10 épocas contra las 30 de V3 dejan lr y presupuesto confundidos. | descriptivo | `results/v3b_*.json`; `decisions.md`, línea V3 | "La hipótesis del learning rate conservador no se sostuvo, y el diseño confunde learning rate con presupuesto de épocas, de modo que la causa no es atribuible." | informe; una línea en la discusión del paper |

**Prohibidos en este bloque**: el "score compuesto" que ordenaba V3/V3b/V3e, y el
−0,032 de V3e. El valor es **−0,031** (C2). Además: el learning rate de V3b fue
elegido por ese score compuesto hoy retirado, así que la procedencia de la receta
de V3b se declara, no se omite.

## 7. Bloque D — Control preregistrado de idioma contra canal

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| D1 | La pendiente de la ganancia contra el SNR sigue al idioma y no al canal de grabación: ρ = −0,065 (n.s.) en inglés/audiolibro, −0,262 en español/crowdsourced y −0,212 en español/audiolibro, con IC95 agrupado por hablante [−0,309, −0,115]. | **preregistrado y confirmado** (criterio: ρ ≤ −0,15 con IC sin cruzar cero) | `results/reanalysis_stats.json → families.F6.p1`; hash en `docs/preregistro_mls_es.sha256` | "Con las condiciones de ruido fijadas y el canal cambiado, la pendiente contra el SNR se mantiene negativa (ρ = −0,212, IC95 [−0,309, −0,115]); con el idioma cambiado es indistinguible de cero. En este régimen el idioma no es despreciable frente al canal." | los dos |
| D2 | El tercer sellado reusa las condiciones de ruido del primero de forma verbatim: los 250 pares coinciden en archivo de ruido, SNR, offset de ruido y categoría. | descriptivo (verificación determinista) | `seal_test_metadata/test_v3_mls_es_metadata.json` contra `test_v1_metadata.json`, **cotejado par por par el 19/09/2026: 250/250 en los cuatro campos** | "El tercer conjunto sellado reusa las 250 condiciones de ruido del primero sin modificarlas, verificado campo por campo." | los dos |
| D3 | La ganancia del fine-tuning transfiere entre canales de forma atenuada: +0,221 en crowdsourced contra +0,084 en audiolibro (IC95 agrupado [+0,057, +0,113]), con 74,8 % de archivos que mejoran (IC95 [0,688, 0,805]). El umbral compuesto preregistrado pedía media > +0,10 **y** proporción > 0,70: la proporción pasa, la media no. | **preregistrado, criterio de media fallado**; desenlace PARCIAL, declarado de antemano | `results/reanalysis_stats.json → families.F6.p2` | "El criterio preregistrado de adaptación lingüística no se cumplió por completo: la proporción de archivos que mejoran lo supera, la magnitud media no. Lo aprendido sobre Common Voice es en parte español y en parte ese canal." | los dos |
| D4 | La retención entre canales distingue el tipo de mejora, no la receta de fine-tuning: la mejora agnóstica al idioma (V2 − V1) transfiere al 132 %, y los dos fine-tunings al español (V3e − V1 y V5 − V2) al 38 % y 34 %. | exploratorio (hipótesis posterior a la retractación, sobre estos mismos datos) | recomputado el 19/09 de `results/{v1,v2,v3e,v5}_{v2_es,v3_mls_es}.json`, cada efecto contra su propio punto de partida. **No usar `reanalysis_stats.json → F6.retention_by_recipe`**: publica el estimando viejo (ver §11, D-2) | "Medido cada efecto desde su propio punto de partida, la mejora agnóstica al idioma transfiere entre canales sin pérdida, y los fine-tunings al español retienen entre un tercio y dos quintos. Es generación de hipótesis, no confirmación." | los dos |
| D5 | El tercer sellado tiene 40 hablantes y esa estructura es inseparable del canal que se quiere fijar —un corpus de audiolibros son pocos lectores leyendo mucho—; la dirección del confusor se declaró antes de sellar: conservadora para D1, ambigua para D3. | descriptivo | `seal_test_metadata/test_v3_mls_es_metadata.json → n_speakers = 40`. **El segundo sellado no registra hablante**: su metadata sólo tiene ruta de clip (249 clips distintos en 250 pares), así que el "~248 hablantes" que circula en `decisions.md` no es verificable desde el material sellado | "El control de canal tiene 40 hablantes y la limitación se declaró antes de generar el dato: favorece la conclusión de D3 y no la de D1." | los dos |

## 8. Bloque E — Estadística del efecto por SNR y del olvido

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| E1 | El aporte se enuncia como interacción: la ganancia no depende del SNR en inglés (ρ = −0,065, p_Holm = 0,305) y decae con el SNR en español (ρ = −0,262, p_Holm = 8,3e−05); la interacción apareada sobrevive Holm (ρ = +0,173, p_Holm = 0,012). | exploratorio | `results/reanalysis_stats.json → families.F1` | "La ganancia decae con el SNR en español y no en inglés; la interacción sobrevive la corrección por familia. Es exploratorio: la hipótesis salió de estos datos." | los dos |
| E2 | El negativo puntual del bucket [15,20] dB no es sostenible por sí solo: con n=50 el test de signo da 24/50 (p = 0,89), y sobre el tercer sellado ese mismo bucket es positivo (V1 +0,104; V2 +0,330). | descriptivo | `analysis/hueco_bucket_checks.py`, **corrido el 19/09** (24/50, p = 0,8877). Los valores del tercer sellado **ya están publicados**: `docs/v7_compuerta_desde_cero.md` líneas 207-208 y la sección V7 de `EXPERIMENTS.md`, donde V1 y V2 son las filas de referencia de esa tabla. Son citables aunque las filas de la compuerta de ese documento estén excluidas por X1 | "El bucket de SNR alto no sostiene por sí solo la afirmación: el test de signo no tiene potencia con 50 pares, y el signo negativo no aparece en el canal de audiolibro." | los dos (salvedad obligatoria) |
| E3 | Los modelos entrenados sólo en inglés degradan activamente el audio limpio en español **crowdsourced**: en el bucket [15,20] dB de `test_v2_es`, V1 da −0,167 de PESQ-NB y −3,49 dB de SI-SDR, y V2 −0,087 y −3,58 dB. Es transversal a la loss. | descriptivo | `results/v1_v2_es.json` y `v2_v2_es.json → by_bucket[4]` | "Sobre habla limpia en español crowdsourced, los dos modelos entrenados en inglés degradan la señal; ocurre con MSE puro y con la loss combinada, así que no es un artefacto de una variante." **No se escribe sin la salvedad de E2 en el mismo párrafo.** | los dos, **con el canal escrito en la oración** |
| E4 | El olvido en inglés es selectivo y no difuso: la mediana no se mueve y el signo es una moneda (51,2 / 47,2 / 52,8 % de archivos que mejoran), pero la media cae por una cola izquierda (asimetría −3,5 / −2,2 / −2,6). El estimando correcto es la fracción de archivos que se rompen: 17,6 % (V3), 11,6 % (V3e) y 6,8 % (V3b), monótona con la agresividad del learning rate. | exploratorio | `results/reanalysis_stats.json → families.F3` | "El olvido no se lee en la media: se concentra en una cola de entre 6,8 % y 17,6 % de archivos, monótona con el learning rate. Es exploratorio." | los dos |
| E5 | Esa cola no es ruido de medición: contra controles del mismo idioma la fracción rota es 0,0–0,4 %; el McNemar apareado da c = 0 en los tres casos (b = 44 / 17 / 29, p_Holm 1,0e−12 / 4,6e−05 / 2,2e−08); los archivos rotos se repiten entre variantes entre 4 y 7 veces por encima del azar (p_Holm ≤ 3,5e−06); sólo 54 de 250 se rompen en alguna variante. | exploratorio | `results/reanalysis_stats.json → families.F5.a`, `.F5.b` y el perfil estratificado | "Contra controles del mismo idioma y con el mismo protocolo, la cola desaparece; y los archivos que se rompen son los mismos entre variantes muy por encima del azar." | los dos |
| E6 | Hay un confusor medido y descartado por estratificación, no por un control marginal: los archivos que se rompen tienen PESQ basal más alto bajo V1 (+0,276, IC95 [+0,077, +0,480]); dentro del cuartil basal más alto los controles siguen en 0,0 % y los tratamientos en 9,5–28,6 %, y el perfil por cuartil no es monótono. | exploratorio | `results/reanalysis_stats.json → families.F5.c`, `.F5.d` y el perfil por cuartil | "El nivel basal es un confusor real y está medido; la estratificación lo descarta, y el perfil por cuartil no tiene la forma que tendría un efecto techo." | los dos |
| E7 | Normalizado por el margen disponible (techo nominal 4,5 menos el PESQ del ruidoso), el rendimiento es plano en inglés (24,7 / 25,0 / 22,3 % en los tres buckets superiores), se derrumba en español crowdsourced (14,8 / 3,4 / −11,4 %) y decae sin cambiar de signo en español de audiolibro (24,8 / 21,1 / 7,0 %). | exploratorio | `decisions.md` 06/09 ya publica las dos primeras series (tabla de margen normalizado, líneas 893-895). La tercera —español de audiolibro— es **la columna que falta en esa tabla**, recomputada el 20/09 de `results/v1_v3_mls_es.json → all_pairs` con el mismo cociente de medias por bucket. No es un hallazgo nuevo: es esa tabla completada sobre el tercer sellado | "Con el margen disponible igualado, el inglés recupera una fracción constante en todo el rango y el español decae en los dos canales: hasta cambiar de signo en el crowdsourced y sin cambiarlo en el de audiolibro. La caída del delta crudo a SNR alto no se explica por efecto techo." **La columna de inglés se lee con M1: ese sellado está contaminado.** | los dos |

**Reconciliación de E2 y E7 con lo ya escrito.** Ninguna de las dos filas aporta un
hecho que el proyecto no tuviera: la diferencia es que el hecho está en un documento
y la afirmación que lo contradice está en otro. Los deltas crudos del bucket alto
sobre el tercer sellado se publicaron con V7 (+0,104 para V1 y +0,330 para V2), y la
tabla de margen normalizado se publicó el 06/09 con dos de sus tres columnas. Lo que
falta es que eso baje a las afirmaciones: el §9.1 del informe enuncia hoy la
degradación de audio limpio "en español" sin el calificativo de canal, y la entrada
del 06/09 concluye sobre "el español" con la única columna que cambia de signo. La
corrección es de enunciado, no de número, y no obliga a retractar nada.

**Salvedad de E1 que hay que escribir**: el apareamiento entre `test_v1_en` y
`test_v2_es` comparte el SNR en los 250 pares pero **sólo 18 de 250 comparten el
archivo de ruido** (`reanalysis_stats.json → integrity.noise_file_match_en_es`).
El contraste inglés/español-crowdsourced confunde idioma, canal y sorteo de
ruido; es exactamente el hueco que cierra D2, y por eso D1 es la fila fuerte y
E1 la exploratoria.

## 9. Bloque F — El negativo del proxy perceptual (V4 / V4b)

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| F1 | Un proxy perceptual congelado se gamea desde la primera época: las cuatro corridas con término Squim degradan contra el audio sin procesar, con dosis-respuesta en el peso del término (−0,343 / −0,806 / −0,943 / −0,846 para α = 0,95 / 0,90 / 0,80 / 0,70), mientras el control con MSE puro y el mismo presupuesto da +0,081. | descriptivo, con control interno | `results/v4_sweep/*_v1_en.json → global.pesq_nb.delta_mean` | "Con el proxy congelado, las cuatro corridas quedan por debajo del audio sin procesar y el control con MSE puro no. La degradación crece con el peso del término perceptual." | los dos |
| F2 | El protocolo estabilizado que prescribe la literatura, sin reentrenar el proxy, recupera aproximadamente un tercio del daño y no lo elimina: el daño atribuible al término perceptual es −0,887 en el régimen inestable y −0,598 en el estabilizado, cada uno contra su propio control. | descriptivo, con placebo de idéntico protocolo | `results/v4_sweep/{control_mse,alpha_090}_v1_en.json` y `results/v4b/v4b_{placebo,main}_epoch_03_v1_en.json` | "Arrancar de un modelo convergido y bajar el learning rate diez veces recupera un tercio del daño; dos tercios sobreviven. Lo que falta es el reentrenamiento del proxy." | los dos |
| F3 | Sin el placebo la conclusión se invierte: contra el audio sin procesar el término perceptual cuesta −0,059 y parece inocuo; contra el placebo con idéntico protocolo (+0,539) cuesta −0,598. | descriptivo / metodológico | `results/v4b/v4b_comparison.json`, filas `v4b_main` y `v4b_placebo`, época 3 | "El control con protocolo idéntico cambia el signo de la lectura: la comparación contra el audio sin procesar habría llevado a la conclusión opuesta." | los dos |
| F4 | La brecha entre el PESQ estimado por el proxy y el PESQ-WB real diagnostica el gaming sin correr la evaluación completa: +2,36 y +2,77 contra un error nominal del proxy de 0,142, entre 16 y 20 veces, y creciendo monótonamente por época (2,04 → 2,30 → 2,36 y 2,59 → 2,70 → 2,77). | descriptivo | `results/v4b/v4b_comparison.json → pesq_hat` y `pesq_wb`; MAE nominal de Kumar et al. 2023, Tabla 2 | "La brecha entre el valor estimado por el proxy y el PESQ-WB real crece época a época hasta 16–20 veces el error nominal del proxy. Es el diagnóstico barato del gaming." | los dos |
| F5 | Con una loss que incluye el proxy, la selección de `best.pt` por mínima `val_loss` elige el checkpoint más gameado, porque el término perceptual entra en la pérdida de validación. | descriptivo / metodológico | `decisions.md`, "Cierre de V4" (31/08); los cinco checkpoints del Paso 1 se evaluaron así | "El criterio de selección de checkpoint queda contaminado por lo mismo que se quiere medir; con esas losses hay que evaluar época por época." | los dos (limitaciones) |

## 10. Bloque G — Método, reproducibilidad y costo

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| G1 | Hay tres conjuntos de test sellados, de 250 pares cada uno, cinco buckets de SNR × 50 pares, semilla 42 y hash SHA-256 acumulativo, sellados una sola vez y no regenerados. | descriptivo (verificación determinista) | los tres `seal_test_metadata/test_*_metadata.json` y sus `*_hash.txt`, **cotejados el 19/09: 250 pares y 50 por bucket en los tres** | "El material de evaluación son tres conjuntos sellados de 250 pares, con hash publicado, generados una única vez." | los dos |
| G2 | Las predicciones del control de idioma contra canal se hashearon antes de que el dato existiera: el hash es de las 16:22 del 06/09/2026 y el sellado del conjunto, de las 16:40 del mismo día. | descriptivo (verificación determinista) | `docs/preregistro_mls_es.sha256` (mtime) contra `test_v3_mls_es_metadata.json → generated_at` | "Las predicciones y los criterios de decisión se comprometieron por hash dieciocho minutos antes de que el conjunto de test existiera." | los dos |
| G3 | El análisis estadístico está cubierto por 358 tests escritos por alguien distinto del que escribió el análisis. | descriptivo | `tests/test_reanalysis_stats.py`, **358 recolectados el 19/09/2026** | "El código del análisis está cubierto por 358 tests, escritos bajo la regla de que quien escribe el análisis no escribe sus tests." | los dos |
| G4 | La reproducibilidad bit-exacta no cubre todas las variantes: `cudnn_deterministic` se incorporó recién en el sweep de V3b, así que V1, V2 y V3 se entrenaron sólo con semilla fija y no son bit-exactos. No se corrigió retroactivamente. | descriptivo / limitación | `decisions.md` 22/08; `training/trainer.py` | "V1, V2 y V3 son reproducibles en inicialización y orden de datos pero no bit-exactos; las variantes posteriores sí." | los dos (limitaciones) |
| G5 | El costo de rendimiento del determinismo de cuDNN que se había registrado no reproduce: 31,0 a 32,0 min por época con el flag activo y sin él. | descriptivo | `decisions.md` 07/09, tabla de cuatro corridas | "La penalización de rendimiento atribuida al determinismo no se reproduce en mediciones posteriores." | informe |
| G6 | Una corrida cortada se reanuda de forma bit-exacta, verificado por test con cada entrenamiento en su propio proceso. | descriptivo (verificación determinista) | `tests/test_resume.py`; `decisions.md` 16/09 | "La reanudación tras un corte produce checkpoints bit-idénticos a los de la corrida sin cortar." | informe; una línea de setup en el paper |
| G7 | El dataset en español tiene splits oficiales sin fuga de hablante ni de frase, y un 22,9 % de sus clips mide menos que la duración de la mixtura, lo que activa la repetición del clip en vez del recorte. | descriptivo | `decisions.md` 17-18/08, de `scripts/analyze_cv26_es.py`. **Sin artefacto en `results/`**; los manifests viven en `data/interim/`, fuera del repositorio | "Los splits no comparten hablantes ni frases. En aproximadamente uno de cada cuatro pares en español la voz limpia es una repetición del clip, y no un recorte; es una diferencia real contra el material en inglés." | los dos (limitaciones) |
| G8 | El costo medido del proyecto es 150,69 h de GPU, 26,370 kWh, 7,91 kg de CO₂ y USD 52,74 de equivalente en nube, sobre 20 corridas: 12 reportadas (128,09 h) más los 8 brazos de la línea de proxy perceptual (22,60 h). El consumo es estimado a partir de potencia nominal sostenida, no medido con instrumento, con error acotado en ±15 %. | descriptivo | `informe/FUENTES.md`, tablas de costo del 19-20/09, generadas con `analysis/training_cost_report.py`; verificación independiente: el subtotal de V4b da 8,02 h y coincide con lo registrado en `decisions.md` el 01/09 | "El proyecto consumió 150,69 h de GPU medidas en 20 corridas, bajo supuestos de potencia declarados y con el consumo estimado, no instrumentado." | informe |
| G9 | Ese total sigue sin ser el del proyecto: faltan diez corridas sin reporte generado (V0, los cinco brazos del sweep de learning rate de V3b y las cuatro corridas de humo descartables). | descriptivo | `informe/FUENTES.md`, sección de costo; directorios de `checkpoints/` | "El costo se reporta en dos totales, corridas reportadas y total de proyecto. La línea del proxy perceptual no tiene una corrida única: son ocho brazos, y ése es el precio de haberla cerrado con diseño experimental." | informe |

| G10 | La evaluación nunca fue bit-reproducible: `evaluate_variant.py` no activa el determinismo de cuDNN, y dos re-evaluaciones de la misma variante con código idéntico difieren entre sí tanto como cualquiera difiere del JSON commiteado. A tres decimales —la precisión de reporte— los cuatro agregados son idénticos en las tres corridas. | descriptivo | informe §4.1.1, medido el 20/09/2026, en tres pasos: forward repetido, métricas del ruidoso bit-idénticas por no pasar por GPU, y dos reevaluaciones completas de V1 | "Los resultados son reproducibles a la precisión reportada, no bit-exactos. El hallazgo no mueve ningún número: las diferencias están entre tres y cinco órdenes de magnitud por debajo de esa precisión." | los dos (metodología y limitaciones) |
**Nota sobre G8**: el total incluye las horas de la línea de la compuerta. Es una
afirmación de costo, no una afirmación sobre esa línea.

---

## X. Fila única de lo excluido

| # | Qué | Estado | Qué se hace con esto |
|---|---|---|---|
| X1 | V6 y V7 — la compuerta causal de paso directo. | **Pendiente de confirmación. No se ledgerea todavía.** La confirmación está corriendo al 19/09/2026 (`logs/v7_seeds.log`, brazo de control de la semilla 43 en la época 18 de 20). Su preregistro está hasheado en `docs/preregistro_v7_semillas.sha256` (19/09, todavía sin commitear) y declara tres semillas nuevas —43, 44 y 45, seis corridas— con la 42 fuera del estimando confirmatorio. | Ninguna fila de este ledger depende de X1. Cuando haya resultado se abre un bloque propio y se decide destino. Hasta entonces no se escribe ni como aporte ni como hallazgo preliminar en ningún documento que salga del repositorio. |

**Una fila de método queda en zona gris y la decisión es de Gabriel.** El piso de
ruido por checkpoint —sd 0,0099 sobre `test_v2_es` y 0,0166 sobre `test_v1_en`,
`docs/v6_compuerta.md` §8— se midió **dentro del diseño de V6**, pero es una
afirmación sobre el ruido entre checkpoints consecutivos, no sobre la compuerta,
y no depende de la confirmación que está corriendo. Es lo que justifica que el
estimando primario del proyecto haya pasado del checkpoint seleccionado a la
trayectoria promediada (`decisions.md`, 15/09), y es la cuarta evidencia de lo
que ya dicen B4 y F5. Si la exclusión se lee literal, esta fila se cae y con ella
el argumento de por qué cambió el estimando; el resto del ledger no se mueve.

---

## 11. Discrepancias encontradas entre documentos y archivos

Doce. La regla aplicada fue que gana el archivo, con una excepción declarada.

| id | Dónde | Documento dice | Archivo dice | Resolución |
|---|---|---|---|---|
| D-1 | `CLAUDE.md`, "Del salto V0→V1" | ganancia de +1,23 puntos de PESQ | 1,417 sale del val de 50 clips de V0 (`summary_metrics.csv`) y 2,650 del sellado de 250 (`v1_v1_en.json`) | **No es un contraste válido tal como está.** Reparable, con condiciones: ver §11.b |
| D-2 | `results/reanalysis_stats.json → F6.retention_by_recipe` | — | sigue publicando el estimando retractado el 09/09 (ganancia definida contra V1 para las dos recetas) | **Única excepción a la regla**: acá el archivo es el desactualizado y la corrección documentada es la que vale. El ledger usa D4, recomputado de los JSON de variante. Pendiente real: `analysis/reanalysis_stats.py` sigue calculando ese campo mal, y por la regla 2 del proyecto el arreglo y sus tests no los escribe el mismo agente |
| D-3 | `docs/EXPERIMENTS.md` línea 146 | 17.582.977 parámetros | 17.579.459, recontado sobre `models/crn.py` el 19/09 | Gana el código. `FUENTES.md` ya tiene el valor correcto; hay que corregir `EXPERIMENTS.md` |
| D-4 | `CLAUDE.md` e `informe/06` | costo de V3b en inglés: −0,029 | −0,0282 (`v3b_v1_en.json` contra `v1_v1_en.json`) | **−0,028**. Es la misma clase de error que el −0,032 de V3e ya retractado |
| D-5 | `decisions.md` 07/09 | IC95 de la ganancia en audiolibro [+0,056, +0,112]; proporción [0,686, 0,806] | [+0,057, +0,113] y [0,688, 0,805] (`reanalysis_stats.json → F6.p2`) | Gana el JSON |
| D-6 | `decisions.md`, "Test set sellado v1" | "200 pares", y "4 buckets" seguido de una lista de cinco | 250 pares, 5 buckets × 50 (`test_v1_hash.txt` y el metadata) | Gana el metadata. Es prosa vieja de julio, no afecta ningún número reportado |
| D-7 | `CLAUDE.md`, estructura del repositorio y `DESVIACIONES.md` D2 | `evaluation/monitor_correlation.py` como el KPI propio, con la verificación "pendiente" | **el archivo no existe** | **No es una discrepancia documental: es un objetivo comprometido sin instrumento.** Escalado a §11.a |
| D-8 | `CLAUDE.md`, "Sprint tercer sellado (13/09/2026)" | 13/09 | `generated_at` del metadata: 06/09/2026 16:40 | Gana el metadata. La fecha importa porque es la que se compara contra el hash del preregistro (G2) |
| D-9 | `CLAUDE.md` "En curso" y `FUENTES.md`, tabla de preregistros | la confirmación de V7 no está lanzada y su preregistro "no existe todavía" | corriendo desde el 19/09 a las 00:03 (`logs/v7_seeds.log`); el preregistro está hasheado en `docs/preregistro_v7_semillas.sha256` desde el 19/09 a las 20:36, sin commitear | Los dos documentos quedaron desactualizados el mismo día. No afecta a ninguna fila: X1 sigue siendo "pendiente" en cualquiera de las versiones |
| D-10 | `decisions.md` 18/09 | el estimando confirmatorio de V7 es el promedio de las semillas 43 y 44, con n=2 y ~45 h | el preregistro hasheado declara **tres** semillas nuevas (43, 44 y 45), seis corridas, y el promedio no ponderado de los tres estimandos | Gana el preregistro, que es posterior y está comprometido por hash. La entrada del 18/09 quedó superada por su propio cierre; conviene anotarlo ahí, porque un lector del informe va a encontrar primero el n=2 |
| D-11 | `DESVIACIONES.md` D5, título | "Se ejecutaron ocho variantes, no cinco" | el cuerpo de esa misma entrada lista once (V0, V1, V2, V3, V3b, V3e, V4, V4b, V5, V6, V7), y hay trece tags en el repositorio | Once variantes, diez con tag. El título de D5 quedó desalineado con su propio cuerpo y alimenta el capítulo 8 |
| D-12 | `informe/01`, §1.3, cierre de OP-6 | "todas las variantes lo superan… por márgenes de entre 0,49 y 0,69 puntos de PESQ-NB" | el mínimo sobre las ocho variantes y los tres sellados es **+0,167** (V1 sobre `test_v2_es`); el máximo es +0,687 (V2 sobre `test_v1_en`). El rango 0,52–0,69 es el de la mejor variante de cada sellado, no el de todas | La afirmación "todas superan" es correcta y está verificada; el rango que la acompaña no. Se reporta el rango completo, o se dice "la mejor variante de cada sellado" |

### 11.a — D-7 escalado: el tercer eje del aporte no tiene instrumento

D-7 entró a la tabla como una fila más y no corresponde. El KPI propio no es un
detalle de implementación: es **uno de los tres ejes diferenciadores** que
`CLAUDE.md` declara para el proyecto, y en el anteproyecto (§6.3) está marcado con
estrella y con umbral de alerta —`Pearson r (loss, PESQ) < 0,3` al final del
entrenamiento—. Lo que se verificó es que `evaluation/monitor_correlation.py` no
existe en el repositorio y que **ningún valor de Pearson r se calculó nunca**, ni
durante el entrenamiento ni después.

**Qué puede seguir afirmando el proyecto, en orden de fuerza decreciente.** Las tres
primeras están sostenidas; la cuarta no, y es la que hay que dejar de escribir.

1. **Que el riesgo que motivaba el KPI se materializó y quedó medido.** Eso está en
   F1 a F5 y no depende del instrumento prometido.
2. **Que el diagnóstico que lo detectó es la brecha `pesq_hat − PESQ-WB real`**, con
   su magnitud contra el error nominal del proxy (F4), y que creció monótonamente
   por época en las dos corridas con término perceptual.
3. **Que la brecha es mejor diagnóstico que la correlación, y por qué**: una
   correlación alta es compatible con un sesgo constante grande, que es exactamente
   la firma del gaming. Ese argumento es del proyecto y se sostiene solo.
4. **Lo que NO se puede afirmar**: que el proyecto haya construido, usado o validado
   un KPI de alineación entre la pérdida y la percepción. No se midió alineación
   —que es correlación, un estimando de forma— sino **sesgo**, que es un estimando
   de nivel. Son cosas distintas, y la sustitución **no fue diseñada: se descubrió
   después**, mirando los `history.json` de corridas que ya estaban hechas.

**Un límite de la brecha que hay que declarar, y que no estaba escrito.** `pesq_hat`
sale de `val_squim` del `history.json` —la media de Squim sobre las 2.000 mixturas de
**validación**— y el PESQ-WB real con el que se compara está medido sobre los 250
pares del **sellado**. No es una comparación apareada archivo por archivo, sino una
diferencia entre dos medias sobre dos conjuntos distintos de la misma distribución.
Con una brecha de +2,36 contra un error nominal de 0,142 eso no cambia la conclusión,
pero el enunciado correcto dice "sobre validación" y "sobre el sellado", no "en los
mismos archivos".

**Reparación, si se quiere recuperar el KPI comprometido.** Es barata y no requiere
reentrenar: correr el modelo sobre un conjunto de archivos, pasar sus salidas por
Squim **guardando el valor por archivo**, y correlacionar contra el PESQ real por
archivo. Hoy eso no se puede hacer con lo que hay en disco, porque no existe ningún
`pesq_hat` por archivo: `scripts/v4b_protocol_check_evaluate.py` sólo lee el escalar
por época. Con eso se obtiene el `r` prometido para los checkpoints que sobrevivan.
**Lo que ninguna reparación puede dar es la función**: el KPI se comprometió como
monitor con umbral de alerta durante el entrenamiento, y reconstruirlo en septiembre
sobre checkpoints viejos produce el número, no la vigilancia. Si se hace, se escribe
como verificación posterior; decir que el KPI alertó del gaming sería falso, porque
las decisiones de V4 y V4b se tomaron con la brecha.

### 11.b — D-1 es reparable en la aritmética, no en la lógica, y conviene repararlo igual

La pregunta es si evaluar `checkpoints/v0/best.pt` sobre los sellados legitima el
delta. Respuesta en tres partes.

**Lo barato está disponible.** El checkpoint existe (`checkpoints/v0/best.pt`,
211 MB, del 27/06/2026) y evaluarlo sobre los tres sellados son tres corridas de 250
pares, sin entrenamiento. Eso convierte el contraste en uno sobre material idéntico,
que es lo que hoy no es.

**La condición que pediste no se puede cumplir, y no por descuido de esta
verificación: el dato no existe.** Los 200 pares de entrenamiento de V0 no son
identificables ni reconstruibles, por tres razones acumulativas:

1. `data/processed/train` fue sobrescrito con los 50.000 pares de V1 el 25/07/2026;
   los archivos de V0 no están.
2. Las mixturas no guardan procedencia: cada `pair_NNNN/` tiene `clean.wav` y
   `noisy.wav` y nada más. No hay manifest.
3. No son regenerables: `datasets/make_mixtures.py` no llamaba a `random.seed()` y
   `collect_files` devolvía el orden no determinista de `rglob()` hasta el
   19/08/2026 (`decisions.md`, esa fecha). El sorteo de V0 no se puede repetir.

**Lo que sí se puede hacer es acotar, y el resultado da vuelta la objeción.** Bajo
el sorteo que el código implementa —`random.choice` uniforme con reposición sobre
28.539 clips— la exposición esperada de V0 a las emisiones selladas es **0,87 %, unos
2,2 clips de 250**. La de V1, con 52.000 extracciones, es **83,8 %, unos 210 de 250**
(M2). O sea: **el brazo cuya contaminación no se puede descartar es el que casi no
tiene, y el que ya está reportado es el que la tiene entera.**

**Veredicto y recomendación.** Correrlo. El delta resultante es un contraste sobre
material idéntico y se escribe con dos declaraciones al lado, ninguna opcional:
(a) la identidad de los pares de entrenamiento de V0 es irrecuperable, así que la
ausencia de solapamiento no está demostrada sino acotada en esperanza; (b) la
exposición es asimétrica y favorece a V1 en dos órdenes de magnitud, de modo que el
delta medido es una **cota superior** de la mejora atribuible al escalado de datos.
Con esas dos frases el número es defendible; sin ellas, no. Y mientras no se corra,
el enunciado que sí está habilitado hoy es el de B1 y B2 por separado, cada uno sobre
su propio material.

---

Tres huecos de trazabilidad que no son discrepancias pero bloquean citas:

- **A3 (RTF) no tiene artefacto en `results/`.** Sostiene el objetivo de calidad
  OP-4 y hoy su única fuente es prosa de `EXPERIMENTS.md`. Correr
  `benchmarks/measure_rtf.py` y guardar la salida cierra el hueco.
- **G7 (dataset en español) tampoco.** Los manifests están fuera del repositorio.
- **V3b no tiene tag**, mientras `EXPERIMENTS.md` declara que cada entrada
  corresponde a un tag. Son trece tags: tres de sellado y diez de variante, y
  V3b es la única variante corrida que quedó sin etiquetar (commit `49235c4`).
- **El segundo sellado no registra hablante** (ver D5), así que la inferencia
  agrupada por hablante que usa D1 y D3 sobre el tercer sellado no se puede
  replicar sobre el segundo. Los números del segundo sellado son sin agrupar.

---

## 12. (a) El espinazo del paper, según lo que el ledger sostiene

*Reescrito el 20/09 tras el bloque M: la validez del material cambia cuál es la fila
que aguanta el peso, aunque no cambia cuál es el aporte.*

El paper no es sobre un modelo mejor: es sobre **qué hace falta para poder afirmar
que una adaptación lingüística funcionó, y qué pasa cuando esa exigencia se aplica a
los propios resultados**. La fila que aguanta el peso es D1, la única preregistrada y
confirmada, y la revisión del material la dejó **más sólida en términos relativos**:
contrasta dos conjuntos sellados en español que son los dos disjuntos del
entrenamiento, con las 250 condiciones de ruido idénticas verificadas campo por campo
(D2), de modo que no la toca la contaminación que sí afecta al sellado en inglés (M1).
Con el canal cambiado y el idioma fijo la pendiente contra el SNR se mantiene
negativa; con el idioma cambiado es indistinguible de cero. A su lado va D3, que es
preregistrada y **falla** la mitad de su criterio: la adaptación transfiere entre
canales atenuada, lo que obliga a reformular el aporte —lo que se aprendió es en parte
idioma y en parte canal— y esa reformulación, hecha contra un criterio escrito antes
del dato, es más publicable que el resultado que se buscaba. El segundo pilar es el
negativo del proxy perceptual (F1 a F5): no es "no funcionó" sino atribución causal
con placebo de idéntico protocolo, con dos tercios del daño sobreviviendo al protocolo
estabilizado de la literatura y con la demostración de que sin el placebo la lectura se
invierte (F3). El tercer pilar es metodológico y recorre todo: cuatro evidencias
independientes de que seleccionar checkpoint por mínima pérdida de validación no está
alineado con la métrica que se reporta (B4, F5, las réplicas de C4 y el piso de ruido
de la zona gris). Lo que sostiene el conjunto es el **diseño**, no la magnitud: cada
intervención tiene su control con idéntico protocolo, y en dos ocasiones el control dio
vuelta la conclusión.

**Y hay una cuarta pata que apareció el 20/09 y que conviene que el paper use en vez de
sufrir**: el trabajo encontró, auditando su propia infraestructura, que el sellado en
inglés comparte pool con el entrenamiento (M1) y que la evaluación nunca fue
bit-reproducible (G10). Las dos salieron de verificaciones rutinarias, las dos están
medidas y acotadas, y ninguna de las dos mueve un contraste entre variantes. Un paper
que declara esto y muestra exactamente qué afirmación se cae —la generalización a
hablantes no vistos en inglés— y cuál no —todo contraste sobre el mismo sellado— está
haciendo lo mismo que hace con el placebo de V4b, un nivel más arriba. Es coherente con
el resto: el aporte del trabajo es el aparato de control, y el aparato incluye
auditarse el material.

Todo lo que está por debajo de eso —el ablation en inglés (B2, B3, B5), la adaptación
al español (C1 a C3), el reanálisis (E1 a E7)— entra como descriptivo o exploratorio
declarado, con un checkpoint por brazo, y la sección de límites lo dice en el cuerpo:
sin réplicas de semilla ningún número de esas filas está protegido contra la varianza
de entrenamiento, el único que la midió (C4) descubrió que el titular sale del máximo
de tres corridas, y los valores absolutos en inglés son cotas superiores hasta que
exista un sellado disjunto.

## 13. (b) Afirmaciones del anteproyecto sin ninguna fila que las sostenga

Esto es lo que hay que declarar como reducción de alcance. Ninguna tiene fila en
§4 a §10 porque no hay número que la respalde.

| # | Comprometido | Estado real | Dónde se declara |
|---|---|---|---|
| R1 | Pérdida perceptual diferenciable (PESQNet) como segundo eje de innovación, y V5 como "propuesta completa (español + PESQNet)" | El eje se cerró como negativo (F1–F5). V5 conserva el nombre del hito y cambia de contenido: es V2 fine-tuneado a español, sin término perceptual | `DESVIACIONES.md` D1; informe §1.5 |
| **R2** | KPI propio: correlación de Pearson entre la loss y el PESQ, con umbral de alerta r < 0,3. Es uno de los **tres ejes diferenciadores** que declara `CLAUDE.md`, y va con estrella en el anteproyecto §6.3 | **No existe y nunca se calculó.** El diagnóstico que rindió (F4) mide sesgo, no alineación: es otro estimando, no una variante del prometido, y se descubrió después en vez de diseñarse. Reparable sin reentrenar, con el límite de que ninguna reparación devuelve la función de monitor | §11.a, que es donde queda escrito qué se puede seguir afirmando; `DESVIACIONES.md` D2 |
| R3 | Comparativa cuantitativa contra RNNoise y DeepFilterNet2 | No corrida. Es el incumplimiento más visible para un tribunal, porque es lo que ubica el trabajo contra el estado del arte publicado | `DESVIACIONES.md` D8; informe §00, problema 3 |
| R4 | Métricas DNSMOS y SegSNR dentro del alcance de evaluación | No implementadas. `evaluation/metrics.py` tiene PESQ-NB, PESQ-WB, STOI y SI-SDR | `DESVIACIONES.md` D7 |
| ~~R5~~ | Objetivo de calidad OP-6: superioridad sobre el baseline pasabajo | **Resuelto el 20/09/2026**, después de la primera versión de este ledger: el baseline se corrió sobre los tres sellados en sus dos modos y todas las variantes lo superan en PESQ y STOI (fila B5). Sale de esta lista y entra al ledger. Queda el matiz de D-12 sobre el rango, y el de M1 sobre el margen en inglés | `FUENTES.md`, sección Butterworth; informe §1.3 |
| R6 | Test set sellado de 300 pares | 250 por conjunto, tres conjuntos, 750 en total. El desvío es favorable y hay que justificar el 250 por la potencia por bucket | `DESVIACIONES.md` D4 |
| R7 | Cinco variantes experimentales (V1 a V5) | Once variantes contando las iteraciones de receta (V0, V1, V2, V3, V3b, V3e, V4, V4b, V5, V6, V7), más dos réplicas de semilla y los brazos de control de V4, V4b, V6 y V7. Corresponde decir qué se sacó del plan para financiarlas: GCRN, descartado el 06/09 | `DESVIACIONES.md` D5, con la salvedad de D-11 |
| R8 | El sistema "no utiliza información del futuro" | Son 10,0 ms de lookahead medidos (A2). El enunciado correcto es causal a nivel de frame con 10 ms de latencia algorítmica | `DESVIACIONES.md` D6 |
| R9 | "Más del 85 % de los trabajos evaluados corresponden a LibriSpeech, WSJ0 o DNS Challenge" | Afirmación cuantitativa sobre la literatura sin metodología registrada. O se recupera el conteo o se reformula en términos cualitativos | informe §1.1 |
| R10 | Auditoría auditiva subjetiva sobre muestra representativa, y la escucha dirigida de cierre de V4, que su propio plan declaraba obligatoria | No se hicieron, y está declarado por escrito: `docs/proxy_enganado.html` §09 dice "no escuchamos los audios… sabemos cómo se ve la firma del gaming en las métricas, no cómo suena en nuestro modelo". `scripts/generate_listening_samples.py` existe y no tiene salida documentada | `CLAUDE.md`, pendientes de mediano plazo; `decisions.md` 30/08, Fase 8 |
| R11 | Evaluación downstream sobre ASR / WER | Fuera de alcance comprometido en sentido estricto, pero figura como trabajo futuro y es la pregunta natural del tribunal sobre utilidad | informe §9.4 |

De las once, R5 quedó resuelta el 20/09 y **R1, R2 y R10 son las que cambian lo que
el informe puede afirmar sobre sus propios objetivos**; las demás son desvíos de
alcance declarables sin consecuencia sobre los resultados. R3 y R4 son decisiones
pendientes de Gabriel: implementar y re-evaluar, o declarar la reducción.

**Y hay una duodécima que no viene del anteproyecto sino de esta revisión**: la
validez del sellado en inglés (M1 a M3). No es una reducción de alcance —nadie la
prometió ni la incumplió— pero se declara en el mismo lugar y con el mismo tono,
porque es la limitación que un tribunal técnico va a encontrar antes que ninguna
otra si mira las dos líneas de código.
