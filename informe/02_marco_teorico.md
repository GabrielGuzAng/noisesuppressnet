# Capítulo 2 — Marco teórico

**Extensión estimada**: 14 pp · **Estado**: redactable
**Fuentes primarias**: `stft.py`, `models/crn.py`, `training/losses.py`,
`tests/test_causality.py`, `tests/test_stft.py`, bibliografía central de `CLAUDE.md`.

Capítulo expositivo. Va todo lo que hace falta para entender los capítulos 4 y 5,
y nada más. El criterio de corte: si un concepto no se usa después, no entra.

---

## 2.1 Representación tiempo-frecuencia: la STFT

- **Afirma**: el sistema opera sobre la magnitud de la STFT y reconstruye
  combinando la magnitud estimada con la fase de la señal ruidosa.
- **Parámetros**: n_fft = 320, hop = 160 (10 ms a 16 kHz), win = 320 (20 ms),
  ventana Hamming, F = 161 bins. Implementación: `torch.stft`, no custom.
- **Fuente**: `stft.py`; `EXPERIMENTS.md` §V0, "Configuración de STFT".
- **Afirma además**: la reconstrucción es round-trip exacta dentro de la
  tolerancia numérica, verificada por `tests/test_stft.py`.
- **Falta**: el valor del error de round-trip. Correr el test y anotarlo.

## 2.2 Causalidad y latencia

- **Afirma**: la causalidad a nivel de frame se logra por truncamiento temporal
  después de cada convolución del encoder y padding asimétrico en el decoder, y
  se verifica bit-exact: perturbar la entrada en el instante t no modifica la
  salida en ningún instante anterior a t.
- **Afirma, y es la parte que importa**: `center=True` en la STFT agrega 10,0 ms
  de lookahead a nivel de señal. **La latencia algorítmica del sistema es de
  10 ms, no cero.** Es aceptable para tiempo real; lo que no es correcto es
  afirmar "sin lookahead".
- **Fuente**: `tests/test_causality.py`; `decisions.md`, "Causalidad temporal —
  detalles de implementación"; `CLAUDE.md`.
- **Falta**: nada. Esta subsección es la que sostiene la corrección D6 del
  capítulo 1 y conviene que tenga una figura del cono de dependencia temporal.

## 2.3 Arquitectura CRN

- **Afirma**: encoder convolucional de 5 capas que reduce la dimensión
  frecuencial, LSTM unidireccional de 2 capas con hidden 1024 que modela la
  dependencia temporal, decoder simétrico con skip connections, salida softplus
  para garantizar magnitudes positivas. 17.579.459 parámetros, de los cuales
  ~16,5 M (94 %) están en el LSTM.
- **Fuente**: `models/crn.py`; Tan & Wang 2018, Fig. 5 e Interspeech.
- **Afirma además**: la unidireccionalidad del LSTM es lo que garantiza la
  causalidad del bottleneck; una BLSTM daría mejor PESQ y rompería la
  restricción operativa.
- **Falta**: nada. Conviene una figura de bloques con las dimensiones anotadas.
- **Nota**: la concentración del 94 % de los parámetros en el LSTM explica el
  costo del determinismo de cuDNN (§4.4) y es un dato reutilizable, no trivia.

## 2.4 Funciones de pérdida

Cuatro subapartados, uno por familia usada en el proyecto.

- **MSE sobre magnitud STFT** — la del paper base. Fuente: `training/losses.py`,
  `mse_magnitude`.
- **SI-SDR** — invariante a escala, definida en el dominio temporal.
  Fuente: `training/losses.py`, `si_sdr_loss`; Le Roux et al. 2019.
- **Combinada MSE + SI-SDR** — 0,7 · MSE + 0,3 · SI-SDR, la de V2.
  Fuente: `training/losses.py`, `mse_plus_sisdr`; Braun & Tashev 2021 §4.3.
- **Perceptual con proxy diferenciable** — PESQNet (Xu et al. 2022) en el plan
  original, TorchAudio-Squim (Kumar et al. 2023) en la implementación.
  **Se expone la teoría acá y se la ejecuta en el capítulo 6, donde falla.**
- **Falta**: la justificación del 0,7/0,3 de V2. Verificar si está en
  `decisions.md` o si fue una elección sin registro; si no está, se dice.

## 2.5 Métricas de evaluación

- **Afirma**: PESQ-NB y PESQ-WB (intrusivas, perceptuales), STOI
  (inteligibilidad), SI-SDR (separación señal/distorsión). Rango, qué mide cada
  una y por qué no son intercambiables.
- **Fuente**: `evaluation/metrics.py`.
- **Afirma, y es necesario para el capítulo 6**: la cabeza de Squim estima
  **WB**-PESQ, acotada en [1, 4,64] por una sigmoide, con MAE nominal de 0,142
  (Kumar 2023, Tabla 2). Toda comparación contra el proxy se hace contra PESQ-WB,
  nunca contra NB.
- **Falta**: decidir DNSMOS y SegSNR (D7). Si entran, este apartado crece.

## 2.6 Transfer learning y olvido catastrófico

- **Afirma**: el fine-tuning de una red pre-entrenada sobre un dominio nuevo
  mejora en el dominio nuevo a costa de degradar el original, y la magnitud del
  intercambio depende de la agresividad del learning rate.
- **Fuente**: Yosinski et al. 2014; McCloskey & Cohen 1989.
- **Afirma además, y es aporte del proyecto**: la media no es el estimando
  correcto del olvido. En este trabajo se mide por **fracción de archivos que se
  rompen**, porque la distribución del cambio en inglés tiene mediana en cero y
  una cola izquierda pesada (skew −2,2 a −3,5).
- **Fuente**: `reanalisis_estadistico.md`, F3.
- **Falta**: nada. Este apartado prepara el capítulo 7 y conviene que sea
  explícito sobre por qué se cambió de estimando.
