# V6 — Compuerta causal de paso directo

**Estado: resultado negativo sobre el endpoint primario preregistrado, con
mecanismo confirmado y un hallazgo de diseño reutilizable.**
Corridas del 14-15/09/2026. Documento separado; se integra a `EXPERIMENTS.md`
y `decisions.md` cuando se decida dónde va cada parte.

---

## 1. Resumen

Se agregó al CRN una compuerta convexa, causal y por banda que le da a la red un
camino de identidad hacia la entrada:

    M_out[t,f] = g[t,f] · M_hat[t,f] + (1 − g[t,f]) · M_noisy[t,f]

193 parámetros sobre 17.579.459 (+0,0011 %). Diseño tratamiento/placebo, seis
épocas de fine-tuning desde `checkpoints/v2/best.pt`, entrenando **solo en
inglés**.

| | resultado |
|---|---|
| **P1** (primaria) | **falla**: +2,5 pp contra un umbral de +3 pp, p = 0,181 |
| **P2** (costo acotado) | pasa |
| **P3** (mecanismo) | **falla como está escrito**, por un error de signo mío en el preregistro. El mecanismo se cumple con ρ ≈ −0,4, p < 1e−8 |
| **P4** (degeneración) | pasa: media de `g` entre 0,57 y 0,65 |

Por la regla de decisión preregistrada, **la hipótesis no queda sostenida**.

Lo más valioso que salió no fue el veredicto sino la medición del **piso de ruido
por checkpoint**: sd = 0,0099 sobre `test_v2_es` y 0,0166 sobre `test_v1_en`, del
mismo tamaño que el efecto buscado. Ninguna comparación de un solo checkpoint
puede resolver efectos de esta magnitud, y eso condiciona todo diseño futuro.

---

## 2. La hipótesis

El CRN de Tan & Wang 2018 hace **mapeo espectral directo**: la salida sale de
`softplus(bn1_t(conv1_t(d2)))` y no toca nunca la magnitud de entrada. Para dejar
un bin como está hay que reconstruirlo exactamente desde el cuello de botella
LSTM. **No existe un camino barato para expresar "acá no hagas nada".**

Se verificó leyendo el código oficial de **Tan & Wang 2020 (GCRN)**, el sucesor
publicado del paper base, que **tampoco tiene camino de identidad**: sus
compuertas son GLU entre capas (`conv1(x) * sigmoid(conv2(x))`, gating de
features internas) y la salida sale por `fc1`/`fc2` sin ningún término que
dependa de la entrada. La propuesta es ortogonal a las dos arquitecturas.

**Hipótesis:** esa ausencia es el mecanismo por el cual el modelo *degrada* voz
limpia fuera de dominio. Evidencia propia: en el bucket [15,20] dB de
`test_v2_es`, V1 recupera −11,4 % del margen disponible y V2 −5,9 %; negativos,
es decir que consumen margen en la dirección equivocada. En inglés, mismo
bucket, V2 recupera +38,7 %.

**Salvedad sobre la novedad.** Mezclar salida realzada con entrada ruidosa no es
inédito: es un truco conocido de mitigación de sobre-supresión. Lo que sería
aporte no es la capa sino la afirmación medida: que una compuerta aprendida,
causal y por banda acota la degradación fuera de dominio, cuantificada con el
estimando de margen normalizado sobre sellados preregistrados.

---

## 3. El diseño, y por qué no fue "V6 contra V2"

La comparación ingenua está confundida, y de las dos formas posibles:

- **Fine-tunear V2 con compuerta y comparar contra V2** mezcla la compuerta con
  las épocas extra. El proyecto ya lo midió: el placebo de V4b (V1 + 2 épocas a
  lr 2e-5, sin término perceptual) superó a V1 en las cuatro métricas.
- **Entrenar desde cero y comparar contra el V2 existente** también confunde: V2
  se entrenó antes de que el trainer tuviera `cudnn_deterministic`.

Se adoptó el **protocolo de V4b: tratamiento + placebo**, con una sola variable.

| | arranca de | arquitectura | épocas | lr | loss | datos |
|---|---|---|---|---|---|---|
| placebo | `checkpoints/v2/best.pt` | CRN sin compuerta | 6 | 2e-5 | mse_plus_sisdr α 0,7 | `data/processed` (EN) |
| tratamiento | `checkpoints/v2/best.pt` | CRN con compuerta | 6 | 2e-5 backbone / 1e-3 compuerta | idéntica | idéntico |

Semilla 42, `cudnn_deterministic=True`, `save_every_n_epochs=1`, sin scheduler.
**El estimando es tratamiento − placebo**, apareado archivo por archivo.

Preregistro en `~/nosiesuppressnet-oracle/preregistro_compuerta.md`, hash
`cc4ce698…` en `docs/preregistro_compuerta.sha256`, escrito **antes** de
modificar `models/crn.py` y de que existiera ninguna corrida.

---

## 4. Implementación

La cabeza espeja `conv1_t`: `ConvTranspose2d(32, 1, kernel=(2,3), stride=(1,2),
padding=(1,0))` sobre `d2`, los mismos features causales, seguida de sigmoide.
Sin BatchNorm, que pelearía con la inicialización del sesgo.

**Inicialización preservadora del nulo.** Sesgo +3,0 (`g₀` = 0,953) y pesos con
`std=0,01`. El tratamiento arranca casi igual al placebo —perturbación relativa
medida sobre audio real: 0,07— y la compuerta solo puede *aprender a replegarse*
si la loss lo premia. Un sesgo más alto la dejaría con gradiente casi muerto.

**Corrección durante la verificación.** El pad temporal metía un logit cero en el
frame 0, o sea `g = sigmoid(0) = 0,5` exacto: una mezcla 50/50 en el primer frame
sin importar lo aprendido. Ahora el pad lleva el sesgo aprendido, así que sin
features disponibles el frame 0 cae en el valor por defecto del modelo.

**Verificaciones previas al entrenamiento:**

- `CRN(gate=False)` es **bit-idéntico** al CRN original: cargando
  `checkpoints/v2/best.pt` en las dos clases, `max|diff| = 0,000e+00`.
- **Causalidad de frame con la compuerta activa**: `tests/test_causality.py`
  parametrizado con y sin compuerta, 24 tests pasan. La compuerta se predice
  desde `d2` y mezcla con la entrada del mismo frame, así que no puede introducir
  lookahead — ahora verificado, no asumido.
- Smoke de 1 época sobre 2.000 pares antes de comprometer las 6 h.

---

## 5. Resultados

**Entrenamiento.** 6 épocas por rama, 31,8 a 32,9 min/época. Los dos `best.pt`
cayeron en la **época 2** por mínimo de `val_loss`. `g` no degeneró: 0,624 →
0,592 → 0,526 → 0,558 → 0,585 → 0,530.

**Global, PESQ-NB sobre el checkpoint seleccionado:**

| test set | V2 | placebo | compuerta | comp − plac | plac − V2 |
|---|---|---|---|---|---|
| inglés, LibriSpeech | 2,8493 | 2,8963 | 2,8856 | −0,0106 (p=5e−03) | +0,0470 (p=2e−18) |
| español, Common Voice | 2,4511 | 2,4635 | 2,4775 | +0,0141 (p=0,72) | +0,0124 (p=3e−03) |
| español, audiolibro | 2,7615 | 2,8089 | 2,8055 | −0,0034 (p=0,21) | +0,0474 (p=8e−14) |

**El placebo se ganó sus 3,3 horas.** Contra V2, el tratamiento en inglés da
2,8856 contra 2,8493 = **+0,036 y parece una mejora**. Contra el placebo con
idéntico protocolo, **cuesta −0,011**. Las seis épocas extra a lr 2e-5 valen
+0,047 por sí solas. Sin ese control la conclusión habría sido la opuesta a la
correcta — la misma lección de V4b, replicada sobre otra intervención.

**P1 — bucket [15,20] dB de `test_v2_es`, % del margen recuperado:**

| | V1 | V2 | placebo | compuerta | contraste |
|---|---|---|---|---|---|
| margen | −11,4 % | −5,9 % | −3,5 % | −1,0 % | **+2,5 pp**, p = 0,181 |

Umbral preregistrado: ≥ +3 pp con p < 0,05. **Falla.** El criterio fuerte
(alcanzar margen ≥ 0) también falla, aunque el tratamiento queda mucho más cerca
de cero que V2.

**P3 — el mecanismo, con el signo que corresponde:**

| media de `g` | [−5,0] | [0,5] | [5,10] | [10,15] | [15,20] | ρ(g, SNR) |
|---|---|---|---|---|---|---|
| inglés / audiolibro | 0,658 | 0,607 | 0,586 | 0,534 | 0,447 | −0,403 (p=4e−11) |
| español / crowdsourced | 0,713 | 0,678 | 0,666 | 0,601 | 0,570 | −0,351 (p=1e−08) |
| español / audiolibro | 0,692 | 0,634 | 0,606 | 0,541 | 0,499 | −0,409 (p=2e−11) |

**La compuerta aprendió exactamente lo que se buscaba**: enhancement pleno con la
entrada sucia, repliegue con la entrada limpia, monótono y replicado en los tres
sellados.

**Métricas secundarias, compuerta − placebo:**

| | PESQ-WB | STOI | SI-SDR |
|---|---|---|---|
| inglés | +0,0005 (n.s.) | −0,0000 (n.s.) | **+0,26 dB** (p=3e−10) |
| español / crowds. | +0,0231 (p=0,052) | **+0,0046** (p=6e−07) | **+0,23 dB** (p=2e−03) |
| español / audio. | +0,0083 (n.s.) | **+0,0025** (p=5e−06) | **+0,25 dB** (p=2e−09) |

Coherente con el mecanismo: dejar pasar la entrada preserva fidelidad de forma de
onda e inteligibilidad, y deja ruido residual que PESQ castiga.

---

## 6. El error de signo en P3

El preregistro dice *"`g` tiene que crecer con el SNR"*, con el criterio
`media(g | [15,20]) > media(g | [−5,0])`. **Eso contradice el mecanismo que el
mismo documento describe dos secciones antes**, donde con `g₀ = 0,953` la
compuerta *"solo puede aprender a replegarse"* — y replegarse es `g` bajando.

Como `M_out = g·M̂ + (1−g)·M_noisy`, `g = 1` es enhancement puro y `g = 0` es la
entrada sin tocar. Replegarse sobre entrada limpia significa **`g` baja cuando
sube el SNR**, o sea **ρ(g, SNR) negativo**. La predicción está escrita al revés.

**No se revisa el veredicto.** El criterio literal falló y el preregistro existe
justamente para que no se pueda arreglar el enunciado después de ver el dato. Lo
que corresponde es dejar asentadas las dos cosas: el criterio falló, y el error
es interno al documento y contradictorio con su propia sección 1.

**Lección de proceso:** el preregistro se escribió sin validador, que es la
regla 2 de `CLAUDE.md` —nunca el mismo agente escribe el análisis y su
verificación—. Un validador leyendo el preregistro probablemente agarraba el
signo. La misma omisión que había dejado a F6 sin tests.

---

## 7. La trayectoria por época, y el costo en inglés retractado

`best.pt` cayó en la época 2 en las dos ramas. Como `g` siguió moviéndose hasta
la sexta, se evaluaron las seis épocas de las dos ramas sobre los dos sellados
que cargan los endpoints. **No se seleccionó nada con esto**: elegir una época
mirando el sellado sería contaminar el endpoint primario.

Contraste compuerta − placebo:

| | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | media | signo |
|---|---|---|---|---|---|---|---|---|
| global ES | +0,025 | +0,014 | +0,013 | +0,023 | +0,032 | +0,005 | **+0,019** | ++++++ |
| global EN | −0,001 | −0,011 | −0,013 | +0,015 | +0,031 | +0,001 | **+0,004** | −−−+++ |
| [15,20] ES | +5,6 pp | +2,5 pp | +3,5 pp | +5,5 pp | +6,1 pp | −0,0 pp | **+3,9 pp** | +++++− |
| [15,20] EN | +0,3 pp | −1,2 pp | −1,5 pp | +0,2 pp | +1,9 pp | −0,5 pp | **−0,1 pp** | +−−++− |

**Se retracta el costo en inglés.** El −0,011 con p = 0,005 del checkpoint
seleccionado no sobrevive a la trayectoria: a lo largo de las seis épocas el
contraste inglés rebota entre −0,013 y +0,031, con media +0,004 y 3 de 6
positivas. Está centrado en cero. La lectura de "trade-off entre idiomas" que se
escribió la primera mañana **no se sostiene**.

**La época 2 fue una mala mano en las dos direcciones a la vez**: de las peores
para español (+2,5 pp contra una media de +3,9) y de las peores para inglés
(−1,2 pp contra una media de −0,1). Un único checkpoint desafortunado gobernó la
interpretación inicial.

**Y la hipótesis que motivó el barrido tampoco se sostiene.** Se corrió creyendo
que el efecto estaba subestimado por evaluar temprano y crecería con las épocas.
No crece: ρ(época) = −0,20, p = 0,70. No hay tendencia, hay rebote sin
estructura.

Detalle mecánico que sí refuerza el diagnóstico: el margen propio de la
compuerta en [15,20] español **empeora** con las épocas (−1,0 % → −4,0 %,
ρ = −0,77, p = 0,07). Más entrenamiento en inglés no ayuda; refuerza el problema.

---

## 8. El hallazgo reutilizable: el piso de ruido por checkpoint

Contraste por checkpoint, sobre las seis épocas:

| sellado | media | sd | rango | amplitud |
|---|---|---|---|---|
| `test_v2_es` | +0,0187 | **0,0099** | [+0,005, +0,032] | 0,027 |
| `test_v1_en` | +0,0036 | **0,0166** | [−0,013, +0,031] | 0,044 |

**El ruido por checkpoint es del mismo tamaño que el efecto buscado.** Ninguna
comparación de un solo checkpoint puede resolver efectos de esta magnitud, caiga
donde caiga la selección. Eso invalida el **diseño**, no la hipótesis, y vale
para cualquier comparación futura de este orden en el proyecto.

Dos consecuencias operativas:

1. **La selección por mínimo de `val_loss` tiene que reemplazarse.** La meseta de
   V2 rebota ±0,02 época a época sin tendencia (últimas ocho: −0,0399, −0,0372,
   −0,0534, −0,0403, −0,0412, −0,0530, −0,0655, −0,0447); el mínimo en la 19 es
   un golpe de suerte, no convergencia. Seguir seleccionando así es dejar que el
   azar elija el resultado. Es la cuarta evidencia del mismo problema, después
   del placebo de V4b y las dos réplicas de semilla de V5.
2. **Los diseños futuros necesitan promediar** sobre épocas, semillas o ambas, y
   declarar ese estimando por adelantado.

---

## 9. Qué queda en pie y qué no

**En pie:**

- El mecanismo existe y es fuerte: ρ(g, SNR) ≈ −0,4 con p < 1e−8 en los tres
  sellados, replicado.
- El patrón descriptivo es favorable: 6 de 6 épocas positivas en el global
  español, 5 de 6 en el bucket [15,20].
- SI-SDR y STOI mejoran de forma consistente.
- No hay costo detectable en inglés.

**No en pie:**

- **P1, el endpoint primario preregistrado, falla.** No se revisa.
- El costo en inglés que se reportó la primera mañana: retractado.
- La hipótesis de que el efecto crecería con las épocas: no se sostiene.
- El estimando que sí tiene señal —el promedio sobre la trayectoria— **no estaba
  preregistrado**, y los seis checkpoints **no son muestras independientes**. Es
  generación de hipótesis, no confirmación. El "6 de 6" es descriptivo, no un
  p-valor.

**Diagnóstico.** La compuerta aprendió a detectar *"la entrada está limpia"* —una
señal de SNR disponible en el entrenamiento inglés— y no *"estoy fuera de mi
dominio"*, que no existe en datos de un solo idioma. El camino de identidad está
y funciona; falta el controlador que sepa cuándo usarlo. **Deja de ser un
problema de arquitectura y pasa a ser uno de señal de entrenamiento.**

---

## 10. Qué sigue

**V7 corrió, y el diagnóstico de arriba resultó ser el punto.** La compuerta
presente desde la inicialización, entrenada desde cero contra un control desde
cero, 20 épocas cada uno. El screening pasó: el contraste promediado sobre las
épocas 15-20 da **+0,0795** en español contra un umbral de +0,050, y **+0,062**
en inglés, o sea sin costo en el idioma de entrenamiento. El mecanismo se repite
con el mismo signo y la misma fuerza que acá. **Documento completo:
`docs/v7_compuerta_desde_cero.md`.**

Lo que eso le hace a la lectura de V6: el camino de identidad **no** era
insuficiente. Lo insuficiente era atornillarlo a un backbone que ya había
convergido 20 épocas sin él, que es exactamente la limitación que esta sección
anticipaba —seis épocas a lr 2e-5 no alcanzan para reorganizarse— pero que no se
podía separar del resto hasta tener la contrafactual corrida. V6 no queda
retractado: su endpoint primario falló y sigue fallando. Queda **acotado**: falla
para la compuerta agregada tarde, no para la compuerta.

**Precondición cumplida.** `scripts/select_by_val_pesq.py` reemplaza la selección
por mínimo de `val_loss` por PESQ sobre el set de validación, nunca sobre los
sellados. Tamaño de muestra medido: la sd de la diferencia apareada entre
checkpoints es 0,059 —no los 0,73 del PESQ crudo—, así que con 300 pares el error
estándar queda en 0,0034.

**Idea derivada, para su propia celda:** condicionar la compuerta sobre el
residuo |M̂ − M_noisy| además de `d2`. "Voy a cambiar mucho una entrada que se ve
limpia" es la firma del sobre-procesamiento fuera de dominio, y es una señal
interna que no requiere datos del idioma objetivo. Salió del resultado de V6 y
ataca el diagnóstico en vez de la división del trabajo.

---

## Archivos

- `models/crn.py` — `CRN(gate=...)`; con `gate=False` es bit-idéntico al original
- `training/config.py` — `CONFIG_V6_GATE`, `CONFIG_V6_PLACEBO`, `CONFIG_V6_SMOKE`
- `training/trainer.py` — soporte de compuerta, grupos de lr, guarda de `g` por época
- `evaluation/evaluate_variant.py` — deduce la arquitectura del checkpoint, guarda `g` por par
- `tests/test_causality.py` — parametrizado con y sin compuerta
- `scripts/run_v6_gate.sh`, `scripts/run_v6_epoch_sweep.sh`
- `results/v6_gate_*.json`, `results/v6_placebo_*.json`, `results/v6_epochs/`
- `docs/preregistro_compuerta.sha256`
- `logs/v6_gate.log`, `logs/v6_epoch_sweep.log`
- Continuación: `docs/v7_compuerta_desde_cero.md`
