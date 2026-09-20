# Ledger de afirmaciones

Insumo transversal. **No es un resumen del proyecto: es una tabla de decisiones.**
Cada fila fija qué se puede afirmar, con qué nivel de evidencia, de qué archivo
sale el número, con qué hedge se escribe y a qué documento va.

**Creado**: 19/09/2026. Rol: `paper_writer` (ROLES.md).
**Regla de verificación aplicada**: cada número de esta tabla se leyó del archivo
en `results/`, `seal_test_metadata/` o del código, no del documento que lo cita.
Donde documento y archivo discrepan, la fila lleva el del archivo y la
discrepancia queda anotada en la sección 11. Se encontraron **once**.

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

## 5. Bloque B — Ablation en inglés (V0 → V1 → V2)

| # | Afirmación | Nivel | Fuente | Enunciado habilitado | Destino |
|---|---|---|---|---|---|
| B1 | Con 200 pares de entrenamiento el modelo no supera al audio sin procesar en ninguna de las tres métricas medidas, y colapsa a la media del dataset (ratio `std_out/std_target` = 0,30). | descriptivo | `results/summary_metrics.csv` (PESQ-NB 1,417 vs 2,050 del ruidoso; STOI 0,794 vs 0,842; SI-SDR 3,11 vs 5,13 dB), sobre **el val set de 50 clips de V0, no sobre sellado**; PSD en `EXPERIMENTS.md` §V0 | "En el régimen de 200 pares el modelo colapsa: no supera al ruidoso en ninguna métrica sobre su conjunto de validación de 50 clips." | los dos |
| B2 | Con 50.000 pares el modelo supera al audio ruidoso en las cuatro métricas del sellado en inglés: PESQ-NB 2,650 (+0,497), PESQ-WB 2,021 (+0,437), STOI 0,904 (+0,051), SI-SDR 13,80 dB (+5,95). | descriptivo (un checkpoint) | `results/v1_v1_en.json → global.*` | "Escalado a 50.000 pares, el modelo mejora las cuatro métricas sobre el sellado de 250 pares en inglés." | los dos |
| B3 | La loss combinada MSE + SI-SDR aporta sobre MSE puro: +0,200 PESQ-NB y +0,64 dB SI-SDR sobre `test_v1_en`, con mejora en las cuatro métricas y en el 90,4 % de los 250 archivos. | descriptivo (un checkpoint por brazo, sin placebo) | `results/v2_v1_en.json` y `v1_v1_en.json → global`; fracción y test apareado recomputados de `all_pairs` el 19/09 (Wilcoxon p = 1,2e−37) | "Con un checkpoint por brazo, la loss combinada mejora 0,200 puntos de PESQ-NB sobre MSE puro, y mejora el 90,4 % de los archivos del sellado." | los dos |
| B4 | La selección de checkpoint por mínimo de `val_loss` MSE no está alineada con PESQ: V1 más dos épocas a lr 2e-5 con la misma loss da 2,716 / 2,102 / 0,906 / 13,95, mejor en las cuatro métricas, con peor `val_loss` (0,0836 contra 0,0724). | descriptivo | `results/v4b/v4b_placebo_epoch_02_v1_en.json → global`; `v1_v1_en.json`; `decisions.md` 01/09 | "Los valores reportados de V1 son levemente pesimistas: dos épocas más con la misma loss mejoran las cuatro métricas pese a empeorar la pérdida de validación." | los dos (limitaciones) |

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
| E2 | El negativo puntual del bucket [15,20] dB no es sostenible por sí solo: con n=50 el test de signo da 24/50 (p = 0,89), y sobre el tercer sellado ese mismo bucket es positivo (V1 +0,104; V2 +0,330). | descriptivo | `analysis/hueco_bucket_checks.py`, **corrido el 19/09** (24/50, p = 0,8877); `results/v1_v3_mls_es.json → by_bucket[4]` | "El bucket de SNR alto no sostiene por sí solo la afirmación: el test de signo no tiene potencia con 50 pares, y el signo negativo no aparece en el canal de audiolibro." | los dos (salvedad obligatoria) |
| E3 | Los modelos entrenados sólo en inglés degradan activamente el audio limpio en español **crowdsourced**: en el bucket [15,20] dB de `test_v2_es`, V1 da −0,167 de PESQ-NB y −3,49 dB de SI-SDR, y V2 −0,087 y −3,58 dB. Es transversal a la loss. | descriptivo | `results/v1_v2_es.json` y `v2_v2_es.json → by_bucket[4]` | "Sobre habla limpia en español crowdsourced, los dos modelos entrenados en inglés degradan la señal; ocurre con MSE puro y con la loss combinada, así que no es un artefacto de una variante." **No se escribe sin la salvedad de E2 en el mismo párrafo.** | los dos, **con el canal escrito en la oración** |
| E4 | El olvido en inglés es selectivo y no difuso: la mediana no se mueve y el signo es una moneda (51,2 / 47,2 / 52,8 % de archivos que mejoran), pero la media cae por una cola izquierda (asimetría −3,5 / −2,2 / −2,6). El estimando correcto es la fracción de archivos que se rompen: 17,6 % (V3), 11,6 % (V3e) y 6,8 % (V3b), monótona con la agresividad del learning rate. | exploratorio | `results/reanalysis_stats.json → families.F3` | "El olvido no se lee en la media: se concentra en una cola de entre 6,8 % y 17,6 % de archivos, monótona con el learning rate. Es exploratorio." | los dos |
| E5 | Esa cola no es ruido de medición: contra controles del mismo idioma la fracción rota es 0,0–0,4 %; el McNemar apareado da c = 0 en los tres casos (b = 44 / 17 / 29, p_Holm 1,0e−12 / 4,6e−05 / 2,2e−08); los archivos rotos se repiten entre variantes entre 4 y 7 veces por encima del azar (p_Holm ≤ 3,5e−06); sólo 54 de 250 se rompen en alguna variante. | exploratorio | `results/reanalysis_stats.json → families.F5.a`, `.F5.b` y el perfil estratificado | "Contra controles del mismo idioma y con el mismo protocolo, la cola desaparece; y los archivos que se rompen son los mismos entre variantes muy por encima del azar." | los dos |
| E6 | Hay un confusor medido y descartado por estratificación, no por un control marginal: los archivos que se rompen tienen PESQ basal más alto bajo V1 (+0,276, IC95 [+0,077, +0,480]); dentro del cuartil basal más alto los controles siguen en 0,0 % y los tratamientos en 9,5–28,6 %, y el perfil por cuartil no es monótono. | exploratorio | `results/reanalysis_stats.json → families.F5.c`, `.F5.d` y el perfil por cuartil | "El nivel basal es un confusor real y está medido; la estratificación lo descarta, y el perfil por cuartil no tiene la forma que tendría un efecto techo." | los dos |
| E7 | Normalizado por el margen disponible (techo nominal 4,5 menos el PESQ del ruidoso), el rendimiento es plano en inglés (24,7 / 25,0 / 22,3 % en los tres buckets superiores), se derrumba en español crowdsourced (14,8 / 3,4 / −11,4 %) y decae sin cambiar de signo en español de audiolibro (24,8 / 21,1 / 7,0 %). | exploratorio | recomputado el 19/09 de `results/v1_{v1_en,v2_es,v3_mls_es}.json` y `v3e_v2_es.json → all_pairs`, como cociente de medias por bucket. **La tercera serie no está en ningún documento del proyecto** | "Con el margen disponible igualado, el inglés recupera una fracción constante en todo el rango y el español no. La caída del delta crudo a SNR alto no se explica por efecto techo." | los dos |

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
| G8 | El costo de cómputo de las doce corridas reportadas es 128,09 h de GPU, 22,415 kWh, 6,73 kg de CO₂ y USD 44,83 de equivalente en nube, con los supuestos declarados (175 W, 0,3 kg CO₂/kWh, USD 0,35/h). | descriptivo | los doce `results/*_training_cost.json`, **sumados el 19/09** | "Las corridas reportadas consumieron 128,09 h de GPU y 22,4 kWh, bajo los supuestos declarados." | informe |
| G9 | Ese subtotal no es el costo del proyecto: faltan dieciocho brazos sin reporte generado (V0, cinco de V4, tres de V4b, cinco del sweep de learning rate de V3b y cuatro corridas de humo). | descriptivo | `informe/FUENTES.md`, sección de costo; directorios de `checkpoints/` | "El costo se reporta en dos totales, corridas reportadas y total de proyecto. La línea del proxy perceptual no tiene una corrida única: son ocho brazos, y ése es el precio de haberla cerrado con diseño experimental." | informe |

**Nota sobre G8**: el subtotal incluye las horas de la línea de la compuerta. Es
una afirmación de costo, no una afirmación sobre esa línea.

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

Once. La regla aplicada fue que gana el archivo, con una excepción declarada.

| id | Dónde | Documento dice | Archivo dice | Resolución |
|---|---|---|---|---|
| D-1 | `CLAUDE.md`, "Del salto V0→V1" | ganancia de +1,23 puntos de PESQ | 1,417 sale del val de 50 clips de V0 (`summary_metrics.csv`) y 2,650 del sellado de 250 (`v1_v1_en.json`) | **No es un contraste válido.** No entra al ledger ni al informe como delta. V0 y V1 se reportan cada uno sobre el material en que se midió |
| D-2 | `results/reanalysis_stats.json → F6.retention_by_recipe` | — | sigue publicando el estimando retractado el 09/09 (ganancia definida contra V1 para las dos recetas) | **Única excepción a la regla**: acá el archivo es el desactualizado y la corrección documentada es la que vale. El ledger usa D4, recomputado de los JSON de variante. Pendiente real: `analysis/reanalysis_stats.py` sigue calculando ese campo mal, y por la regla 2 del proyecto el arreglo y sus tests no los escribe el mismo agente |
| D-3 | `docs/EXPERIMENTS.md` línea 146 | 17.582.977 parámetros | 17.579.459, recontado sobre `models/crn.py` el 19/09 | Gana el código. `FUENTES.md` ya tiene el valor correcto; hay que corregir `EXPERIMENTS.md` |
| D-4 | `CLAUDE.md` e `informe/06` | costo de V3b en inglés: −0,029 | −0,0282 (`v3b_v1_en.json` contra `v1_v1_en.json`) | **−0,028**. Es la misma clase de error que el −0,032 de V3e ya retractado |
| D-5 | `decisions.md` 07/09 | IC95 de la ganancia en audiolibro [+0,056, +0,112]; proporción [0,686, 0,806] | [+0,057, +0,113] y [0,688, 0,805] (`reanalysis_stats.json → F6.p2`) | Gana el JSON |
| D-6 | `decisions.md`, "Test set sellado v1" | "200 pares", y "4 buckets" seguido de una lista de cinco | 250 pares, 5 buckets × 50 (`test_v1_hash.txt` y el metadata) | Gana el metadata. Es prosa vieja de julio, no afecta ningún número reportado |
| D-7 | `CLAUDE.md`, estructura del repositorio y `DESVIACIONES.md` D2 | `evaluation/monitor_correlation.py` como el KPI propio, con la verificación "pendiente" | **el archivo no existe** | Queda resuelto el pendiente de D2: el KPI comprometido nunca produjo valores de Pearson. Lo que existe es su reemplazo (F4). Ver §13 |
| D-8 | `CLAUDE.md`, "Sprint tercer sellado (13/09/2026)" | 13/09 | `generated_at` del metadata: 06/09/2026 16:40 | Gana el metadata. La fecha importa porque es la que se compara contra el hash del preregistro (G2) |
| D-9 | `CLAUDE.md` "En curso" y `FUENTES.md`, tabla de preregistros | la confirmación de V7 no está lanzada y su preregistro "no existe todavía" | corriendo desde el 19/09 a las 00:03 (`logs/v7_seeds.log`); el preregistro está hasheado en `docs/preregistro_v7_semillas.sha256` desde el 19/09 a las 20:36, sin commitear | Los dos documentos quedaron desactualizados el mismo día. No afecta a ninguna fila: X1 sigue siendo "pendiente" en cualquiera de las versiones |
| D-10 | `decisions.md` 18/09 | el estimando confirmatorio de V7 es el promedio de las semillas 43 y 44, con n=2 y ~45 h | el preregistro hasheado declara **tres** semillas nuevas (43, 44 y 45), seis corridas, y el promedio no ponderado de los tres estimandos | Gana el preregistro, que es posterior y está comprometido por hash. La entrada del 18/09 quedó superada por su propio cierre; conviene anotarlo ahí, porque un lector del informe va a encontrar primero el n=2 |
| D-11 | `DESVIACIONES.md` D5, título | "Se ejecutaron ocho variantes, no cinco" | el cuerpo de esa misma entrada lista once (V0, V1, V2, V3, V3b, V3e, V4, V4b, V5, V6, V7), y hay trece tags en el repositorio | Once variantes, diez con tag. El título de D5 quedó desalineado con su propio cuerpo y alimenta el capítulo 8 |

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

El paper no es sobre un modelo mejor: es sobre **qué hace falta para poder
afirmar que una adaptación lingüística funcionó, y qué pasa cuando esa exigencia
se aplica a los propios resultados**. La fila que aguanta el peso es D1, la única
preregistrada y confirmada: con las condiciones de ruido copiadas verbatim entre
dos conjuntos sellados (D2, verificada par por par), la pendiente de la ganancia
contra el SNR sigue al idioma y no al canal de grabación, que es la objeción de
Wang et al. 2022 contestada en las unidades del trabajo. A su lado va D3, que es
preregistrada y **falla** la mitad de su criterio: la adaptación transfiere entre
canales atenuada, lo que obliga a reformular el aporte —lo que se aprendió es en
parte idioma y en parte canal— y esa reformulación, hecha contra un criterio
escrito antes del dato, es más publicable que el resultado que se buscaba. El
segundo pilar es el negativo del proxy perceptual (F1 a F5), que no es "no
funcionó" sino atribución causal con placebo de idéntico protocolo: dos tercios
del daño sobreviven al protocolo estabilizado de la literatura, con un mecanismo
medido —la brecha contra el error nominal del proxy— y con la demostración de que
sin el placebo la lectura se invierte (F3). El tercer pilar es metodológico y
recorre todo: cuatro evidencias independientes de que seleccionar checkpoint por
mínima pérdida de validación no está alineado con la métrica que se reporta (B4,
F5, las réplicas de C4, y el piso de ruido de la zona gris). Lo que sostiene el
conjunto es el *diseño*, no la magnitud: cada intervención tiene su control con
idéntico protocolo, y en dos ocasiones el control dio vuelta la conclusión. Todo
lo que está por debajo de eso —el ablation en inglés (B2, B3), la adaptación al
español (C1 a C3), el reanálisis (E1 a E7)— entra como descriptivo o exploratorio
declarado, con un checkpoint por brazo, y la sección de límites lo dice en el
cuerpo: sin réplicas de semilla, ningún número de esas filas está protegido
contra la varianza de entrenamiento, y el único que la midió (C4) descubrió que
el titular del proyecto sale del máximo de tres corridas.

## 13. (b) Afirmaciones del anteproyecto sin ninguna fila que las sostenga

Esto es lo que hay que declarar como reducción de alcance. Ninguna tiene fila en
§4 a §10 porque no hay número que la respalde.

| # | Comprometido | Estado real | Dónde se declara |
|---|---|---|---|
| R1 | Pérdida perceptual diferenciable (PESQNet) como segundo eje de innovación, y V5 como "propuesta completa (español + PESQNet)" | El eje se cerró como negativo (F1–F5). V5 conserva el nombre del hito y cambia de contenido: es V2 fine-tuneado a español, sin término perceptual | `DESVIACIONES.md` D1; informe §1.5 |
| R2 | KPI propio: correlación de Pearson entre la loss y el PESQ, con umbral de alerta r < 0,3 | **No existe.** `evaluation/monitor_correlation.py` no está en el repositorio y nunca produjo valores de Pearson. El diagnóstico que rindió es otro (F4), y es una métrica distinta, no una variante de la prometida | `DESVIACIONES.md` D2, cuyo "pendiente de verificar" queda resuelto por D-7 de §11 |
| R3 | Comparativa cuantitativa contra RNNoise y DeepFilterNet2 | No corrida. Es el incumplimiento más visible para un tribunal, porque es lo que ubica el trabajo contra el estado del arte publicado | `DESVIACIONES.md` D8; informe §00, problema 3 |
| R4 | Métricas DNSMOS y SegSNR dentro del alcance de evaluación | No implementadas. `evaluation/metrics.py` tiene PESQ-NB, PESQ-WB, STOI y SI-SDR | `DESVIACIONES.md` D7 |
| R5 | Objetivo de calidad OP-6: superioridad sobre el baseline pasabajo | **No medido sobre material sellado.** La única comparación existente es del 27/06, sobre el val de 50 clips de V0, y ahí el modelo pierde (1,417 contra 2,052). Además el pasabajo de referencia es de fase cero, que no cumple la restricción causal a la que sí está sujeto el modelo | informe §00, problema 1; §1.3 |
| R6 | Test set sellado de 300 pares | 250 por conjunto, tres conjuntos, 750 en total. El desvío es favorable y hay que justificar el 250 por la potencia por bucket | `DESVIACIONES.md` D4 |
| R7 | Cinco variantes experimentales (V1 a V5) | Once variantes contando las iteraciones de receta (V0, V1, V2, V3, V3b, V3e, V4, V4b, V5, V6, V7), más dos réplicas de semilla y los brazos de control de V4, V4b, V6 y V7. Corresponde decir qué se sacó del plan para financiarlas: GCRN, descartado el 06/09 | `DESVIACIONES.md` D5, con la salvedad de D-11 |
| R8 | El sistema "no utiliza información del futuro" | Son 10,0 ms de lookahead medidos (A2). El enunciado correcto es causal a nivel de frame con 10 ms de latencia algorítmica | `DESVIACIONES.md` D6 |
| R9 | "Más del 85 % de los trabajos evaluados corresponden a LibriSpeech, WSJ0 o DNS Challenge" | Afirmación cuantitativa sobre la literatura sin metodología registrada. O se recupera el conteo o se reformula en términos cualitativos | informe §1.1 |
| R10 | Auditoría auditiva subjetiva sobre muestra representativa, y la escucha dirigida de cierre de V4, que su propio plan declaraba obligatoria | No se hicieron, y está declarado por escrito: `docs/proxy_enganado.html` §09 dice "no escuchamos los audios… sabemos cómo se ve la firma del gaming en las métricas, no cómo suena en nuestro modelo". `scripts/generate_listening_samples.py` existe y no tiene salida documentada | `CLAUDE.md`, pendientes de mediano plazo; `decisions.md` 30/08, Fase 8 |
| R11 | Evaluación downstream sobre ASR / WER | Fuera de alcance comprometido en sentido estricto, pero figura como trabajo futuro y es la pregunta natural del tribunal sobre utilidad | informe §9.4 |

De estas once, **R1, R2, R5 y R10 son las que cambian lo que el informe puede
afirmar sobre sus propios objetivos**; las demás son desvíos de alcance
declarables sin consecuencia sobre los resultados. R3 y R4 son decisiones
pendientes de Gabriel: implementar y re-evaluar, o declarar la reducción.
