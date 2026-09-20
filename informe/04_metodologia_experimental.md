# Capítulo 4 — Metodología experimental

**Extensión estimada**: 16 pp · **Estado**: redactable
**Fuentes primarias**: `decisions.md`, `scripts/seal_test_set_*.py`,
`scripts/verify_test_set.py`, `docs/preregistro_*.sha256`, `training/trainer.py`,
`tests/`.

**Acá va el aparato de rigor.** Es el capítulo que más separa este informe de un
trabajo de grado promedio, y el que hace defendibles los capítulos 5, 6 y 7. Se
escribe antes que ellos, porque todos citan hacia acá.

---

## 4.1 Reproducibilidad

- **Afirma**: seed 42 en todas las operaciones aleatorias; determinismo de cuDNN
  activado; datasets versionados con hash SHA-256.
- **Fuente**: `training/trainer.py`; `training/config.py`, flag
  `cudnn_deterministic`.
- **Afirma, y hay que decirlo sin adorno**: **V1, V2 y V3 se entrenaron sin
  determinismo de cuDNN.** Sólo tenían `torch.manual_seed`, pese a que la
  documentación del proyecto afirmaba lo contrario. El bug se cerró antes de V3b.
  No se corrigió retroactivamente: es una limitación real de esos tres resultados
  y va declarada.
- **Fuente**: `decisions.md`, 22/08/2026 y entrada de cierre del bug.
- **Falta**: nada. Declararlo cuesta menos que si lo encuentra el tribunal.

### 4.1.1 La evaluación tampoco es bit-reproducible (medido el 20/09/2026)

- **Afirma**: `evaluation/evaluate_variant.py` **no activa el determinismo de
  cuDNN**. El trainer lo hace desde V3b; el evaluador nunca lo hizo. En
  consecuencia, **ningún JSON de `results/` es bit-reproducible**: re-evaluar la
  misma variante sobre el mismo sellado con el mismo checkpoint da un archivo
  distinto.
- **Cómo se estableció**, en tres pasos, porque la conclusión no se deduce del
  código sino de la medición:

  1. Dos pasadas del forward del mismo modelo, sobre el mismo audio, en el mismo
     proceso, difieren en los 6 pares probados, con `max|a−b|` entre 3e−08 y
     7e−07. El modelo es 94 % LSTM y los kernels de cuDNN para RNN no son
     deterministas por defecto.
  2. Las métricas del audio **ruidoso** —el único camino que no pasa por la
     GPU— salen bit-idénticas en las cuatro métricas y en las tres corridas.
     Sólo difiere lo que pasa por el modelo.
  3. Dos re-evaluaciones completas de V1 sobre `test_v1_en` con **código
     idéntico** difieren entre sí tanto como cualquiera de ellas difiere del
     JSON commiteado: 1163 diferencias contra 1159, con la misma mediana de
     2,384e−07. La variación entre corridas del mismo código es indistinguible
     de la variación contra el archivo sellado.

- **Magnitud**: por par, máximo 7,6e−05 en PESQ-WB. En los agregados globales,
  entre 6,8e−10 (STOI) y 2,0e−07 (PESQ-WB); el SI-SDR global no se mueve, por
  cancelación al promediar. **A tres decimales, que es la precisión con que el
  informe reporta, los cuatro agregados son idénticos en las tres corridas.**
- **Qué NO invalida**: ningún número reportado. Las diferencias están entre tres
  y cinco órdenes de magnitud por debajo de la precisión de reporte, y muy por
  debajo del piso de ruido por checkpoint del §4.6 (sd 0,0099), que es el que
  gobierna qué efectos el proyecto puede resolver.
- **Qué sí invalida**: la afirmación de que los resultados son bit-reproducibles.
  Son reproducibles **a la precisión reportada**, que es una afirmación más débil
  y es la que corresponde escribir.
- **Decisión pendiente**: activar el determinismo de cuDNN en el evaluador. A
  favor, es la única forma de que una prueba de no-regresión sobre esta
  herramienta pueda pasar, y el trainer ya lo hace. En contra, cambia los kernels,
  así que las corridas futuras tampoco reproducirían los JSON ya commiteados: se
  cambia "irreproducible hacia atrás y hacia adelante" por "irreproducible hacia
  atrás, reproducible desde acá".
- **Nota de honestidad para el capítulo 6**: este hallazgo salió de una prueba de
  no-regresión rutinaria, al agregar soporte de baselines al evaluador. La prueba
  no encontró lo que buscaba —el cambio no introdujo regresión— y encontró algo
  que llevaba en el repositorio desde V0.

## 4.2 Sellado de los conjuntos de prueba

- **Afirma**: tres conjuntos de 250 pares, sellados una vez, no regenerables. La
  integridad se verifica por hash acumulativo SHA-256 con
  `scripts/verify_test_set.py`.
- **Diseño**: 5 buckets de SNR × 50 pares. El 50 por bucket es lo que fija la
  potencia del análisis estratificado del capítulo 7; no es un número arbitrario.
- **Fuente**: hashes y metadata en `FUENTES.md`; `scripts/seal_test_set_en.py`,
  `_es.py`, `_mls_es.py`.
- **Afirma además**: `test_v3_mls_es` reusa las condiciones de ruido de
  `test_v1_metadata.json` **verbatim**. Eso es lo que lo convierte en control de
  canal y no en un tercer conjunto cualquiera.
- **Falta**: nada.

## 4.3 Preregistro

- **Afirma**: desde septiembre, toda hipótesis confirmatoria se preregistra con
  hash SHA-256 **antes** de que exista el dato, incluyendo predicciones puntuales,
  endpoints, umbrales y la regla de decisión.
- **Fuente**: `docs/preregistro_mls_es.sha256`, `preregistro_compuerta.sha256`,
  `preregistro_v7_desde_cero.sha256`.
- **Afirma además, y es lo que le da credibilidad al mecanismo**: el preregistro
  de V6 tenía **un error de signo en P3**, escrito por el autor. El endpoint
  falló como estaba redactado aunque el mecanismo se cumpliera con ρ ≈ −0,4,
  p < 1e−8. Se reportó el fallo según la regla preregistrada en vez de corregir el
  signo a posteriori.
- **Fuente**: `v6_compuerta.md` §6.
- **Afirma también**: desde el 18/09 un preregistro lo valida alguien que no lo
  escribió.
- **Fuente**: `decisions.md`, 18/09.
- **Falta**: nada. Este apartado es de los más valiosos del informe: un
  preregistro que sólo se cumple cuando conviene no es un preregistro, y acá hay
  la evidencia de que se cumplió cuando no convenía.

## 4.4 Determinismo de cuDNN: el costo medido, y su corrección

- **Afirma**: el determinismo de cuDNN se activa por defecto y se desactiva sólo
  en corridas exploratorias descartables (sweeps).
- **Cuidado**: la primera medición del costo (~2× tiempo por época, atribuido al
  94 % de parámetros en el LSTM) **no reprodujo** en una verificación posterior.
- **Fuente**: `decisions.md`, 22/08 (medición original) y 07/09 ("Corrección: el
  costo del `cudnn_deterministic` no reproduce").
- **Falta**: leer la entrada del 07/09 y decidir qué número va al informe. Hasta
  entonces, **el 2× no se escribe**.

## 4.5 Protocolo tratamiento / control

- **Afirma**: una intervención no se compara contra una variante anterior sino
  contra su propio control, entrenado con el mismo protocolo, la misma cantidad de
  épocas y la misma semilla, cambiando únicamente la variable bajo estudio.
- **Fuente**: `decisions.md`, 15/09 ("V7 reentrena su propio control en vez de
  comparar contra V2"); `v6_compuerta.md` §3.
- **Afirma además, y es la lección que lo motivó**: en V4b, el contraste contra
  el audio ruidoso daba −0,059 y parecía "casi arreglado"; contra el placebo con
  idéntico protocolo, el término perceptual costaba −0,598. **Sin el control, la
  conclusión hubiera sido la opuesta.**
- **Fuente**: `decisions.md`, "V4b" (01/09).
- **Falta**: nada.

## 4.6 El estimando primario es la trayectoria, no el checkpoint

- **Afirma**: el contraste entre dos ramas varía entre checkpoints con una
  desviación estándar del mismo orden que el efecto buscado — sd 0,0099 sobre
  `test_v2_es` y 0,0166 sobre `test_v1_en`. En consecuencia, **ninguna comparación
  de un solo checkpoint puede resolver efectos de esta magnitud**, y desde el
  15/09 el estimando primario del proyecto es la media sobre una ventana de
  épocas.
- **Fuente**: `v6_compuerta.md` §8; `decisions.md`, 15/09.
- **Afirma como consecuencia directa**: el costo en inglés de V6 se retractó
  porque no sobrevivía a la trayectoria (ver `FUENTES.md`, "Prohibidos").
- **Falta**: nada. Este apartado es metodológicamente el más fuerte del informe y
  merece una figura: el contraste por época, con la banda del piso de ruido.

## 4.7 Selección de checkpoint

- **Afirma**: seleccionar por mínima `val_loss` no está alineado con PESQ, y con
  losses que incluyen un proxy perceptual está directamente contaminado — más
  gameado da menor `val_loss` y por lo tanto "mejor" checkpoint.
- **Evidencia**: V1 no estaba convergido; dos épocas más a lr 2e-5 dan PESQ-NB
  2,716 contra 2,650, mejor en las cuatro métricas, pese a peor val MSE. Y el
  barrido de época de V5 ubica el pico en la 18, no en la de mínima `val_loss`.
- **Fuente**: `decisions.md`, 01/09 (hallazgo lateral de V4b) y entrada de V5.
- **Mitigación**: `scripts/select_by_val_pesq.py` selecciona por PESQ sobre
  validación, **nunca sobre los sellados**.
- **Falta**: la decisión abierta de si se re-reporta V1 con el checkpoint del
  placebo de V4b. Afecta todas las comparaciones V1→V2/V3/V3e y los números de V1
  ya están taggeados. **Es decisión de Gabriel y el capítulo no cierra sin ella.**

## 4.8 Diseño del ablation

- **Afirma**: una variable por vez entre variantes, misma semilla, mismos
  hiperparámetros de optimización salvo cuando la variante es sobre el
  fine-tuning. Las desviaciones se documentan.
- **Afirma, y es una desviación a declarar**: en V3b quedaron confundidas dos
  variables — 10 épocas contra las 30 de V3, y lr conservador. No se pudo aislar
  cuál pesaba. V3e se diseñó para desconfundirlas.
- **Fuente**: `CLAUDE.md`, restricción 3; `decisions.md`, línea V3.
- **Falta**: nada. Una violación del propio protocolo, detectada y corregida por
  el autor, vale más ante un tribunal que un protocolo sin incidentes.

## 4.9 Generación de mixturas

- **Afirma**: pares (ruidoso, limpio) generados a SNR controlado en cinco buckets,
  con balanceo de género y splits disjuntos de hablante y frase.
- **Fuente**: `datasets/make_mixtures.py`; `scripts/analyze_cv26_es.py`
  (train 320.708 / dev 13.139 / test 12.620 clips, 0 leakage).
- **Limitación a declarar**: ~23 % de los clips de Common Voice ES miden menos de
  4 s, y en esos casos `pad_or_crop` loopea el clip en vez de recortar un segmento.
  Ocurre en aproximadamente 1 de cada 4 pares en español y casi nunca en
  LibriSpeech. No es un bug, es una diferencia real entre datasets.
- **Fuente**: `decisions.md`, 18/08/2026.
- **Bug corregido a declarar**: `collect_files` dependía del orden no determinista
  de `rglob()` y faltaba `random.seed(42)` al inicio. Corregido.
- **Falta**: nada.

## 4.10 Verificación por tests

- **Afirma**: inventario de tests y qué garantiza cada uno.

| Test | Garantiza |
|---|---|
| `test_causality.py` | Causalidad bit-exact a nivel de frame |
| `test_stft.py` | Round-trip de la STFT |
| `test_pipeline.py`, `test_pipeline_real.py` | End-to-end |
| `test_dataloader.py` | Integridad del DataLoader |
| `test_squim_differenciable.py` | Squim propaga gradiente |
| `test_reanalysis_stats.py` | 358 tests sobre el análisis estadístico |
| `test_f6_channel_control.py` | Control preregistrado idioma/canal |
| `test_resume.py` | Reanudación bit-exacta, cada entrenamiento en su propio proceso |

- **Falta**: correr la suite completa y anotar el resultado. Si algo falla hoy, se
  arregla antes de citar la tabla.
- **Regla heredada de los agentes, que vale la pena declarar**: nunca escribe el
  mismo autor el análisis y sus tests. El modo de falla dominante no es que el
  código se rompa, es que pase.
