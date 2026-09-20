# V7 — La compuerta de paso directo, entrenada desde cero

**Estado: screening positivo sobre el endpoint primario preregistrado, con tres
reservas escritas y una confirmación pendiente.**
Corridas del 15 al 18/09/2026. Continuación de `docs/v6_compuerta.md`; se integra
a `EXPERIMENTS.md` y `decisions.md` cuando se decida dónde va cada parte.

---

## 1. Resumen

La misma compuerta de V6 —convexa, causal, por banda— pero **presente desde la
inicialización** y entrenada desde cero contra un control desde cero, 20 épocas
cada brazo.

    M_out[t,f] = g[t,f] · M_hat[t,f] + (1 − g[t,f]) · M_noisy[t,f]

| endpoint | criterio preregistrado | resultado | |
|---|---|---|---|
| **E1** contraste global ES, media ép. 15-20 | ≥ +0,050 | **+0,0795** | pasa |
| **E2** costo en inglés | ≥ −0,020 | **+0,0620** | pasa |
| **E3** mecanismo, ρ(g,SNR) | negativo, \|ρ\| ≥ 0,15, tres sellados | −0,35 / −0,42 / −0,40 | pasa |
| **E4** degeneración | media de g en (0,05 , 0,99) | media de g 0,42 a 0,51 | pasa |
| **E5** ¿desde cero rinde más que tarde? | contra el +0,0187 de V6 | 4,3× | desde cero rinde |

Por la tabla de desenlaces del preregistro, el resultado cae en la fila
*"E1 ≥ +0,050 y E3 se cumple"*: **la compuerta desde cero es un aporte
arquitectónico real, y corresponde confirmarlo con tres semillas antes de
escribirlo como tal.**

Y el dato que le pega directo a la hipótesis central del proyecto: en el bucket
de SNR alto [15,20] dB del español, **el control desde cero replica la patología
de V1 y V2** —degrada audio que ya estaba limpio— y la compuerta la cruza a cero.

---

## 2. Qué preguntaba, y por qué no es repetir V6

V6 atornilló la compuerta a un backbone que había convergido 20 épocas sin ella.
Ese backbone aprendió mapeo espectral directo —sintetizar la magnitud limpia
desde cero en cada bin— porque era la única solución disponible sin camino de
identidad. Seis épocas a lr 2e-5 no alcanzan para reorganizarse alrededor de la
nueva libertad.

**V7 pregunta otra cosa: con el camino de identidad disponible desde la
inicialización, ¿la red aprende una división del trabajo cualitativamente
distinta?** Por ejemplo, el decoder especializándose en corrección residual en
vez de síntesis completa, porque la identidad ya carga la parte que está bien.

Es **screening, no confirmación, y el umbral lo refleja**. La banda de ruido
estaba medida en V6: el contraste por checkpoint tiene sd 0,0099 sobre
`test_v2_es` y 0,0166 sobre `test_v1_en`. Un par de corridas no puede resolver un
efecto del tamaño que V6 insinuó (+0,019 promediado sobre la trayectoria). El
umbral se fijó en 3× esa sd: **+0,050**. Sólo puede detectar un efecto grande, y
eso es deliberado — si la hipótesis de la división del trabajo es cierta, el
efecto tiene que ser grande, o "división distinta" es una distinción sin
diferencia.

Preregistro: `~/nosiesuppressnet-oracle/preregistro_compuerta_desde_cero.md`,
hash `4f918035…` en `docs/preregistro_v7_desde_cero.sha256`, escrito el 15/09 a
las 12:32 **antes** de crear las configs y de correr nada desde cero. Incluye la
convención de signo escrita con ejemplo numérico, para que no se repita el error
de P3 de V6.

---

## 3. El diseño

| | arquitectura | init | épocas | lr | loss | datos |
|---|---|---|---|---|---|---|
| control | CRN sin compuerta | aleatoria | 20 | 2e-4, StepLR(2 · 0,98) | mse_plus_sisdr α 0,7 | `data/processed` (EN) |
| tratamiento | CRN **con compuerta** | aleatoria | 20 | idéntico | idéntica | idéntico |

Semilla 42, `cudnn_deterministic=True`, `save_every_n_epochs=1` en las dos.
Receta de optimización: la de V2.

**Una sola variable declarada entre brazos: la compuerta.** Dos decisiones de
diseño que sostienen eso:

- **El control se reentrena y no se reusa V2.** V2 se entrenó antes de que el
  trainer tuviera `cudnn_deterministic`, así que su trayectoria de punto flotante
  es otra — del orden de cambiar la semilla, sd 0,004 a 0,019 medido en las
  réplicas de V5, o sea del mismo tamaño que el efecto buscado. Costo: 10,6 h de
  GPU. Detalle en `decisions.md` (15/09/2026).
- **Sin `gate_lr` separado.** En V6 la cabeza de compuerta llevaba lr propio
  (1e-3) porque era nueva sobre un backbone convergido. Desde cero todo es nuevo:
  un solo lr de 2e-4 para todos los parámetros. Una diferencia menos.

**Una asimetría que no estaba declarada, encontrada el 19/09 al validar el
preregistro de las semillas.** Los dos brazos **no ven el mismo orden de datos**.
La inicialización del backbone sí es bit-idéntica entre `CRN(gate=False)` y
`CRN(gate=True)` —verificado, `max|diff| = 0,000e+00`, porque la cabeza de
compuerta se construye última—, pero construir esa cabeza consume draws del
generador global de CPU, y `RandomSampler` saca de ahí su semilla de época.
Medido con semilla 43, las semillas de sampler de las tres primeras épocas son
`[164201, 823800, 936888]` sin compuerta y `[884710, 469486, 162857]` con ella:
el barajado difiere en todas las épocas.

Es incómodo porque el argumento para reentrenar el control en vez de reusar V2 es
que una diferencia de trayectoria del orden de 0,004-0,019 contamina el contraste,
y las réplicas de V5 miden precisamente la componente de orden de datos. Ese ruido
se evitó por un lado y quedó metido por otro.

**Qué le hace al resultado y qué se hace al respecto.** Cada contraste por semilla
carga un término de ruido del orden de la sd de orden de datos de V5, un orden de
magnitud por debajo del efecto medido (+0,0795), así que no lo explica. **No se
corrige**: cambiarlo ahora haría que las réplicas corran un protocolo distinto al
de la semilla 42 y no serían comparables. Queda declarado como término de ruido
conocido que las réplicas absorben. Si alguna vez se corrige, es un experimento
nuevo.

El sesgo de la compuerta se inicializa igual que en V6 (+3,0 → g₀ = 0,953), para
que la arquitectura sea idéntica y la única diferencia entre V6 y V7 sea
desde-cero contra fine-tuning.

**El estimando es la trayectoria promediada, no un checkpoint.** El contraste
global de PESQ-NB, compuerta − control, apareado archivo por archivo, promediado
sobre las épocas 15 a 20. Está declarado así en el preregistro y la razón es el
piso de ruido de V6: un checkpoint único no resuelve este tamaño de efecto caiga
donde caiga la selección.

---

## 4. La interrupción, y por qué importa para el resultado

**Dos cortes de energía el 16/09.** El primero a las 06:50 mató V7control a mitad
de la época 15 de 20. El segundo a las 13:08 cortó el barrido de evaluación.
Ninguno de los dos fue el proceso: el journal del sistema se corta en seco sin
secuencia de apagado, sin Xid del driver, sin OOM y sin evento térmico, y el ext4
de `sdb1` necesitó replay del journal en los dos arranques siguientes.

**Las épocas perdidas eran exactamente las que cargan el endpoint primario.** Eso
convirtió "reanudar o reentrenar" en una decisión metodológica y no operativa:
una reanudación *equivalente pero no idéntica* mete una perturbación de punto
flotante del mismo orden que el efecto buscado, que es el mismo argumento por el
que el control se reentrenó en vez de reusar V2. **O la reanudación es
bit-exacta, o el experimento se cae.**

`Trainer._resume` repone las tres piezas: pesos y momentos de Adam (vienen en el
checkpoint), la fase del `StepLR` (el lr ya viene en `opt_state` y el scheduler es
multiplicativo, así que sólo falta `last_epoch` — este es el bug que agarró el
test) y el stream del RNG global de CPU, que no se guarda y por eso se **replica**
reconstruyendo un iterador de train y uno de val por época ya hecha.
`tests/test_resume.py` verifica checkpoints bit-idénticos corriendo cada
entrenamiento en su propio proceso, porque reanudar tras un corte es
necesariamente un proceso nuevo.

Evidencia sobre la corrida real, además del test: la secuencia de lr de las
épocas 15-20 del brazo reanudado coincide exactamente con la del brazo que corrió
sin cortes, la train loss engancha sin escalón en el empalme (−0,0867 → −0,0886)
y la val loss no muestra discontinuidad. Cualquier perturbación residual sería
del orden del piso de ruido por checkpoint (0,01-0,017), no de +0,08.

Queda dicho igual, porque es un confusor real aunque esté descartado: **E1 se
apoya sobre las seis épocas del control que se perdieron y se recalcularon.**

---

## 5. Resultados

**Costo.** 31,8 min/época en los dos brazos: 10,60 h el tratamiento, 10,58 h el
control, más ~1,5 h de barrido de evaluación (36 evaluaciones sobre los tres
sellados). Total ~22,7 h de GPU.

### E1 — primaria, `test_v2_es`

Contraste global de PESQ-NB, compuerta − control, apareado:

| | ep15 | ep16 | ep17 | ep18 | ep19 | ep20 | **media** |
|---|---|---|---|---|---|---|---|
| contraste | +0,108 | +0,128 | +0,080 | +0,068 | +0,034 | +0,059 | **+0,0795** |
| p apareado | 1,6e−06 | 3,8e−08 | 1,4e−04 | 9,8e−04 | 0,10 | 7,0e−04 | |

Seis de seis positivas. sd entre épocas 0,0340, rango [+0,034, +0,128].
171 de 250 archivos favorecen a la compuerta.

### E2 — costo en el idioma de entrenamiento, `test_v1_en`

| | ep15 | ep16 | ep17 | ep18 | ep19 | ep20 | **media** |
|---|---|---|---|---|---|---|---|
| contraste | +0,078 | +0,050 | +0,043 | +0,049 | +0,071 | +0,081 | **+0,0620** |

Seis de seis positivas, sd 0,0167. **No hay costo: hay ganancia.** El umbral era
"que no pierda más de 0,020" y el resultado es que gana tanto como en español.
175 de 250 archivos.

### Lo que el contraste global no dice: el bucket de SNR alto

Δ PESQ-NB contra el audio sin procesar, promediado sobre las épocas 15-20:

| `test_v2_es` | [−5,0] | [0,5] | [5,10] | [10,15] | **[15,20]** |
|---|---|---|---|---|---|
| control desde cero | +0,349 | +0,359 | +0,464 | +0,160 | **−0,084** |
| compuerta | +0,442 | +0,452 | +0,523 | +0,223 | **+0,006** |

**El control replica la patología.** Entrenado solo en inglés, degrada voz que ya
estaba limpia en español — igual que V1 (−0,167) y V2 (−0,087) en ese mismo
bucket. No es un artefacto de una variante puntual: es lo que hace esta
arquitectura, con esta receta, fuera de su dominio de entrenamiento. La compuerta
lo cruza a cero.

**Dónde cae la degradación, con los tres sellados a la vista.** Δ PESQ-NB contra
noisy en el mismo bucket [15,20] dB:

| | inglés / audiolibro | español / audiolibro | español / crowdsourced |
|---|---|---|---|
| V1 | +0,343 | +0,104 | −0,167 |
| V2 | +0,595 | +0,330 | −0,087 |
| V7 control | +0,561 | +0,318 | −0,084 |
| V7 compuerta | +0,647 | +0,412 | **+0,006** |

Manteniendo el canal y cambiando el idioma, la ganancia en SNR alto se desploma
pero sigue positiva; el cruce a negativo aparece cuando además cambia el canal.
Las dos cosas empujan. La atribución inferencial —que la **pendiente** contra el
SNR sigue al idioma y no al canal, ρ −0,212 en audiolibro español contra −0,065
en inglés— está en el control preregistrado del 07/09, en `decisions.md`; acá
sólo se muestra el descriptivo por bucket y se deja constancia de que V7 lo
replica con un modelo entrenado desde cero.

En inglés los cinco buckets son fuertemente positivos en los dos brazos, con la
compuerta arriba en todos.

### El tercer sellado, `test_v3_mls_es`

Español de audiolibro, mismo paradigma de grabación que el inglés de
entrenamiento y las condiciones de ruido copiadas verbatim de `test_v1_en`.
Entra porque E3 pide los tres sellados, y de paso da el contraste sobre el
tercero:

| | ep15 | ep16 | ep17 | ep18 | ep19 | ep20 | **media** |
|---|---|---|---|---|---|---|---|
| contraste | +0,084 | +0,075 | +0,053 | +0,076 | +0,087 | +0,078 | **+0,0753** |

Seis de seis positivas y la sd entre épocas más chica de las tres (0,0119).
176 de 250 archivos.

**Los tres estimandos juntos**: +0,0795 (español crowdsourced) · +0,0753 (español
audiolibro) · +0,0620 (inglés). El orden va en la dirección que predice la
hipótesis —más efecto cuanto más lejos del dominio de entrenamiento— pero **la
diferencia entre los tres (0,0175 de punta a punta) es menor que la sd entre
épocas del sellado primario (0,0340)**. La dirección es consistente; la
separación no es resoluble con estos datos.

### E3 — el mecanismo, en los tres sellados

ρ(g, SNR) sobre la rama con compuerta, promediado sobre las épocas 15-20:

| sellado | ρ medio | rango entre épocas | media de g |
|---|---|---|---|
| español, Common Voice | **−0,350** | [−0,375 · −0,334] | 0,507 |
| inglés, LibriSpeech | **−0,415** | [−0,441 · −0,404] | 0,423 |
| español, audiolibro | **−0,399** | [−0,430 · −0,376] | 0,451 |

Criterio preregistrado: ρ negativo con |ρ| ≥ 0,15 en los tres. **Se cumple con
holgura, en las seis épocas de los tres sellados.** El signo es el que la
sección 3 del preregistro fijó con ejemplo numérico: `g` baja cuando sube el
SNR, o sea la compuerta se repliega y deja pasar la entrada cuando la entrada ya
está limpia.

Es el mismo mecanismo que V6 midió (ρ ≈ −0,4) — la diferencia entre los dos
experimentos nunca fue si la compuerta aprende a replegarse, sino si eso alcanza
para mover el endpoint.

### E4 — sin degeneración

Media de g entre 0,423 y 0,507 según el sellado, y entre 0,401 y 0,538 según la
época. El criterio pedía quedar dentro de (0,05 · 0,99): no colapsa ni a
enhancement puro ni a identidad pura. Arrancó en 0,953 por la inicialización
preservadora del nulo y bajó hasta estabilizarse cerca de 0,45.

---

## 6. Las tres reservas

El resultado es positivo y el preregistro lo declara como tal. Estas tres cosas
van escritas al lado, no en un apéndice.

### 6.1 Es una sola semilla por brazo

El preregistro lo dice de entrada: esto es screening. Las réplicas de V5 midieron
sd entre semillas de 0,004 a 0,019 sobre este tipo de contraste, y E1 es de 4 a 20
veces eso — improbable como ruido de semilla, pero no descartado. **V5 también
mostró que la semilla 42 fue la más alta de sus tres corridas en los tres
sellados**, así que el proyecto ya tiene un antecedente propio de que la 42
tiende a caer del lado bueno.

Hay una diferencia con V5 que hace la réplica más necesaria, no menos: acá las dos
ramas entrenan **desde cero**, así que la semilla mueve también la
inicialización, no sólo el orden de los datos.

### 6.2 El mecanismo propuesto no es la única explicación en pie

ρ(g, SNR) es negativo y fuerte en los tres sellados: la compuerta aprendió a
replegarse sobre entrada limpia, tal como se predijo. Pero el contraste es
**parejo entre buckets** —en español, +0,093 / +0,094 / +0,059 / +0,063 / +0,090
de SNR bajo a alto— y en inglés la compuerta gana en todos lados. Si el efecto
fuera sólo "protege la entrada limpia fuera de dominio", debería concentrarse en
el extremo de SNR alto del español, y no lo hace.

El tercer sellado aprieta más esa tuerca. Si el efecto fuera "protege cuando está
fuera de dominio", tendría que escalar con la distancia al dominio: máximo en
español crowdsourced (cambia idioma **y** canal), intermedio en español de
audiolibro (cambia sólo el idioma), mínimo en inglés. Sale +0,0795 / +0,0753 /
+0,0620: **el orden es el correcto pero el rango entero es más chico que el ruido
entre épocas del sellado primario.** Con estos datos no se puede afirmar que el
efecto dependa del dominio.

El brazo con compuerta además **ajusta mejor**: `val_mse` promedio de las épocas
15-20 de 0,0899 contra 0,1026 del control. Eso es consistente con algo más
prosaico y bien conocido: un camino de identidad facilita la optimización, como
cualquier conexión residual. Las dos cosas pueden convivir, pero **la afirmación
"la compuerta acota la degradación fuera de dominio" está sobredeterminada por la
evidencia disponible** y no se puede escribir sola.

Qué la separaría: un brazo con el camino de identidad pero sin condicionamiento
—una mezcla con `g` constante aprendida, sin dependencia de `d2`— aislaría "ayuda
a optimizar" de "sabe cuándo replegarse".

### 6.3 La dispersión entre épocas es más ancha de lo que V6 midió

sd entre épocas de 0,0340 en español, contra los 0,0099 que midió V6 sobre su
propia trayectoria. El umbral de +0,050 se fijó con la sd vieja. E1 lo pasa, pero
el margen sobre el umbral es de aproximadamente dos errores estándar de la media,
no de cinco — y esos seis checkpoints **no son muestras independientes**, así que
ni siquiera ese cálculo es del todo honesto. El p-valor sobre el promedio por
archivo se reporta como descriptivo.

---

## 7. Qué queda en pie y qué no

**En pie:**

- E1 y E2 pasan sus umbrales preregistrados, con 6 de 6 épocas positivas en los dos.
- El mecanismo se replica: ρ(g, SNR) negativo y fuerte en los tres sellados, con
  el mismo signo y magnitud que en V6, pero ahora con el endpoint primario a favor.
- **El control desde cero replica la patología de V1/V2 en SNR alto en español**, lo
  que la convierte en una propiedad de la arquitectura y la receta, y no de una
  variante puntual.
- V6 queda **acotado, no retractado**: su endpoint falló y sigue fallando, pero
  falla para la compuerta agregada tarde, no para la compuerta.

**No en pie, o no todavía:**

- "La compuerta desde cero mejora el modelo" como afirmación confirmada. Es
  screening con n=1 por brazo.
- La atribución del efecto exclusivamente a la división del trabajo (ver 6.2).
- Cualquier lectura del efecto por bucket como si fuera un patrón resuelto: el
  contraste es parejo y la forma por bucket no tiene estructura clara.

---

## 8. Qué sigue

**La confirmación con tres semillas, ~45 h de GPU.** Es la consecuencia que el
preregistro de V7 ata al desenlace obtenido. La infraestructura está lista:
`CONFIG_V7_{GATE,CONTROL}_S{43,44}` en `training/config.py` y
`scripts/run_v7_seeds.sh`, que espera si hay otro job de GPU y reanuda solo desde
el último checkpoint periódico si un corte lo mata.

**Falta el preregistro de la confirmación**, y tiene una decisión abierta que no
puede resolverse después de ver el dato: si el estimando confirmatorio promedia
las dos semillas nuevas o las tres. La semilla 42 es la que disparó la
confirmación, así que incluirla lo sesga hacia arriba por el mismo mecanismo que
V5 ya documentó. Borrador en
`~/nosiesuppressnet-oracle/BORRADOR_preregistro_v7_semillas.md`.

**El desempate de 6.2**, si sobra tiempo: un tercer brazo con `g` constante
aprendida separaría "ayuda a optimizar" de "sabe cuándo replegarse". No es
necesario para escribir V7 con las reservas puestas, pero es lo que convertiría
la afirmación en una sola y no en dos superpuestas.

---

## Archivos

- `models/crn.py` — `CRN(gate=...)`; con `gate=False` es bit-idéntico al original
- `training/config.py` — `CONFIG_V7_GATE`, `CONFIG_V7_CONTROL`, y las cuatro de semillas
- `training/trainer.py` — `--resume-from` y `Trainer._resume`
- `tests/test_resume.py` — reanudación bit-exacta, cada corrida en su proceso
- `analysis/v7_gate_endpoints.py` — calcula E1 a E5 desde el barrido
- `results/v7_epochs/` — 36 evaluaciones (2 brazos × 6 épocas × 3 sellados)
- `results/v7_endpoints.json` — la salida del análisis
- `scripts/run_v7_scratch.sh`, `scripts/resume_v7_scratch.sh`, `scripts/resume_v7_eval.sh`,
  `scripts/run_v7_mls_sweep.sh`, `scripts/run_v7_seeds.sh`
- `docs/preregistro_v7_desde_cero.sha256` — hash del preregistro
- `logs/v7_scratch.log`, `logs/v7_resume.log`, `logs/v7_mls_sweep.log`
