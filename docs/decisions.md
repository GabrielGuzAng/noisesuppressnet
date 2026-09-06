## Simplificaciones para el preproyecto

1. STFT: torch.stft estándar en vez de implementación custom con F.relu sobre Hamming.
   Justificación: equivalente numéricamente (Hamming no tiene valores negativos), torch.stft es C++ optimizado.

2. Padding temporal: simétrico (no estrictamente causal).
   Justificación: el repo de referencia (CRN-causal) también usa padding simétrico en la STFT.
   Esta limitación se documenta y se evalúa estrictamente en el informe final.

3. LSTM no agrupada (G=1) en vez del GCRN del repo (G=2).
   Justificación: el paper original usa LSTM estándar; el repo agrega grouping como extensión posterior.


## Causalidad temporal — detalles de implementación

El paper especifica convoluciones causales (kernel 2x3, sin lookahead temporal) 
pero no detalla cómo implementarlas en PyTorch. El repo de referencia 
(JupiterEthan/CRN-causal) revela el patrón:

**Encoder:** después de cada conv2d con padding=(1,0), se trunca el último 
frame temporal con [:, :, :-1, :]. Esto elimina el lookahead que PyTorch 
introduce por su padding simétrico.

**Decoder:** después de cada conv_transpose2d con padding=(1,0), se aplica 
F.pad(x, [0,0,1,0]) que agrega 1 frame de padding solo al pasado.

Esto mantiene la dimensión T constante a lo largo de la red y garantiza 
causalidad estricta a nivel de frame.


# Tests de validación del modelo:
- "Causalidad estricta: diff = 0.00e+00 con input modificado en t≥50"
- "Reconstrucción STFT: error round-trip = 9.54e-07"
- "Pipeline integrado audio→STFT→CRN→audio: funcional con gradientes finitos"


## Test set sellado v1 (26/07/2026)

Se selló el test set v1 con 200 pares en inglés (LibriSpeech + MUSAN/ESC-50).

**Configuración:**
- Seed 42 fija para reproducibilidad
- 4 buckets balanceados de SNR: [-5,0], [0,5], [5,10], [10,15],[15,20]  dB
- 50 pares por bucket
- Script: scripts/seal_test_set_en.py
- Hash: data/test_v1_hash.txt

**Razones del diseño:**
- Buckets balanceados en vez de SNR random garantizan cobertura uniforme
  del rango de dificultad al comparar V1–V5.
- Seed fija + RNGs separados por decisión (archivos/SNR/offsets)
  permiten reproducir el test set exacto si se corrompe.
- Metadata por par (archivo origen, SNR, categoría de ruido) habilita
  el análisis por categoría (matriz de resultados por tipo de ruido)
  mencionado en el Plan de Calidad.

**Uso durante el proyecto:**
- V0, V1, V2, V3, V4, V5 se evalúan sobre este mismo test set.
- El hash prueba que no se tocó entre corridas.
- El test set v2 (100 pares ES adicionales) se sellará por separado


## EDA Common Voice ES v26 y splits de train/dev/test (17/08/2026)

Antes de generar las mixturas ES para V3 se corrió `scripts/analyze_cv26_es.py`
sobre `cv-corpus-26.0-2026-06-12/es` (1.680.810 clips totales en el corpus,
2.278,7 h). El script filtra por calidad, valida los splits oficiales y
escribe manifests limpios en `data/interim/cv26_es/`.

**Hallazgos:**

1. **Splits oficiales disjuntos, sin leakage verificado.** train 320.708
   clips (459,3 h, 5.042 hablantes), dev 13.139 clips (22,1 h, 3.249
   hablantes), test 12.620 clips (21,2 h, 6.099 hablantes). Chequeo cruzado
   confirma 0 hablantes y 0 frases en común entre los tres splits.
2. **Filtros de calidad aplicados sobre cada split:** `up_votes >= 2`,
   `down_votes == 0` (13,1% de los validados tenían `down_votes > 0`),
   duración en [1, 12] s, y exclusión de las frases marcadas en
   `reported.tsv` (motivos principales: grammar-or-spelling, different-language,
   difficult-pronounce).
3. **Sample rate no uniforme.** Verificado sobre muestra de 20 clips:
   32.000 / 44.100 / 48.000 Hz mezclados, mono. No es 48 kHz fijo como se
   asumía antes de correr el EDA.
4. **Distribución de acentos en train:** México 41,1%, sin declarar 24,9%,
   España (Sur + Norte + Centro peninsular) ~21,4%, Andino-Pacífico 3,7%,
   Rioplatense 3,0%, Caribe 2,2%.

**Razones del diseño:**
- Usar los splits `train.tsv`/`dev.tsv`/`test.tsv` oficiales de Common
  Voice en vez de `validated.tsv` porque estos últimos son la unión de los
  tres y no garantizan separación por hablante/frase.
- Tomar duraciones de `clip_durations.tsv` en vez de abrir cada MP3
  individualmente (2.278 h de audio, inviable leer cabecera por cabecera).
- Verificar sample rate real sobre una muestra chica en vez de todo el
  corpus: suficiente para decidir que el resampling es obligatorio, sin
  pagar el costo de recorrer 320k archivos.

**Implicancias:**

- **`data/interim/cv26_es/train_manifest.tsv` y `dev_manifest.tsv` son la
  única fuente válida para generar `data/processed_es/train` y
  `data/processed_es/val`.** `test_manifest.tsv` queda reservado
  exclusivamente para `scripts/seal_test_set_es.py`. Nunca usar
  `validated.tsv` directo para las mixturas: mezclaría clips que después
  podrían terminar sellados en test_v2_es, rompiendo la separación
  train/test que ya se verificó a nivel de hablante y frase.
- El sample rate mixto no requiere cambios de código: `load_resample_mono`
  en `datasets/make_mixtures.py` ya resamplea todo a 16 kHz vía
  `torchaudio.functional.resample`. Se documenta acá para que no se asuma
  erróneamente en el futuro que CV26 es 48 kHz uniforme (asunción que
  llevó al bug de verificación en la primera versión del script de EDA).
- Los mismos filtros de calidad (upvotes, downvotes, duración, reported)
  ya aplicados en el manifest mantienen un estándar de "señal limpia"
  comparable al de LibriSpeech train-clean-100 (pre-filtrado por diseño).
  Esto aísla la variable lingüística en el ablation V1→V3: si V3 pierde
  calidad en algún eje, no debería explicarse por diferencias de curación
  de la fuente de voz.
- Con Rioplatense en apenas 3,0% de train (~9-10 k clips de 320k),
  **filtrar V3 o V5 estrictamente por acento argentino reintroduciría el
  régimen de dataset chico que causó el output collapse en V0** (200
  pares). La decisión tomada es entrenar V3 con todos los acentos y
  evaluar en V5 si un fine-tuning adicional filtrado por Rioplatense
  aporta, sabiendo que el volumen disponible para ese filtro está muy por
  debajo del umbral (~50k pares) que rompió el collapse en V1.


## Generación de mixturas ES y sellado de test v2_es (18/08/2026)

Antes de correr `datasets/make_mixtures.py` sobre Common Voice ES se
resolvieron tres puntos abiertos: dónde vive `data/processed_es`, el
tamaño del test set v2_es, y un efecto de `pad_or_crop` específico de
este dataset.

**1. `data/processed_es` como symlink a `/mnt/Datos`.**
`data/processed` y `data/raw` ya son symlinks a
`/mnt/Datos/noisesuppressnet/data/` (625 GB libres), mientras que el
disco raíz solo tiene 109 GB libres de 218 GB. Se creó
`data/processed_es -> /mnt/Datos/noisesuppressnet/data/processed_es`
para mantener la misma convención antes de generar las ~52.000 mixturas
(~13 GB estimados). Sin esto, `PROCESSED_ES` en `make_mixtures.py`
hubiera escrito directo al disco raíz.

**2. Test set v2_es fijado en 250 pares (5 buckets SNR × 50), no 100.**
La nota de la sección "EDA Common Voice ES v26..." de más arriba en este
mismo archivo (17/08/2026) no menciona tamaño; una nota posterior en
notas de trabajo mencionaba 100 pares, pero quedó desalineada con el
pendiente ya fijado en `CLAUDE.md` ("Sellar test set v2 ES (250 pares
balanceados, hash SHA-256)"). Se confirmó 250 pares para igualar el
poder estadístico de v1_en y poder comparar V1-V5 en ambos idiomas con
la misma granularidad por bucket. `scripts/seal_test_set_es.py` quedó
escrito con ese diseño, usando `test_manifest.tsv` (split disjunto de
train/dev, sin leakage verificado) como única fuente de voz.

- Nota aparte: `scripts/seal_test_set_es.py` corrige las rutas de hash y
  metadata a `seal_test_metadata/test_v2_*` directamente. El script
  original `seal_test_set_en.py` las escribe hardcodeadas en `data/` y
  terminaron movidas a mano a `seal_test_metadata/` después (así es como
  las usa `scripts/verify_test_set.py` hoy). No se corrigió
  retroactivamente `seal_test_set_en.py` para no tocar un script ya
  usado para sellar datos existentes; sí se evitó repetir la
  inconsistencia en la versión ES.

**3. ~23% de los clips de Common Voice ES miden menos de 4s (target de
duración de las mixturas) y activan el camino de `repeat()` en
`pad_or_crop`.** Sobre las 320.708 filas de `train_manifest.tsv`: media
5,16s, mediana 5,06s, pero 22,9% caen por debajo de 4s (mínimo 1,04s).
LibriSpeech (fuente de V1/V2) rara vez dispara ese camino porque sus
clips son sustancialmente más largos. Esto significa que en ~1 de cada 4
pares ES el audio de voz limpio es una repetición looped del clip
original en vez de un recorte aleatorio de un clip más largo — una
diferencia real entre el dataset EN y ES que puede introducir artefactos
de repetición audibles. No se cambió el comportamiento de
`pad_or_crop` (es el mismo mecanismo ya usado y validado en V1/V2); se
documenta acá para tenerlo presente si aparece algo anómalo en las
métricas de V3, y para no confundirlo con un bug nuevo.

**Validación antes de lanzar la corrida completa:** smoke test de 10
pares (semilla 42) confirmó pipeline correcto — 16 kHz, 4s exactos, sin
`NaN`, sin clipping (peak ≤ 0.9) — y un throughput de ~0,09 s/par
(~1,3 h estimadas para 52.000 pares). La generación completa
(`nohup python -m datasets.make_mixtures`) se lanzó en background con
log en `logs/mixtures_es.log`.

**Actualización 19/08/2026 — el proceso murió antes de terminar.**
Solo se generaron 8.135/50.000 pares de train y 0/2.000 de val antes de
que el proceso (PID 11441) dejara de existir a las 23:51:29 del
18/08/2026. `journalctl` muestra que exactamente a esa hora se cerró el
scope `app-org.gnome.Terminal.slice` (13min de CPU, 5,8 GB de memoria
pico) — coincide al segundo con el timestamp del último par escrito. No
hay señal de OOM real (systemd-oomd aparece deteniéndose, no matando por
falta de memoria, y `free -h` post-mortem no muestra presión de
memoria). Diagnóstico: cerrar la ventana de terminal mató el proceso
pese a `nohup`, probablemente porque systemd-logind terminó el cgroup de
sesión completo al cerrarse la terminal. `nohup` protege contra SIGHUP
pero no contra la sesión/cgroup completo siendo terminado.
**Implicancia:** para relanzar, usar `tmux`/`screen` o `setsid` en vez de
`nohup comando &` en una terminal que se puede cerrar.

Aprovechando que este run no llegó a producir un dataset completo (no
hay reproducibilidad que proteger todavía), se detectó además que
`datasets/make_mixtures.py` nunca llama `random.seed()` — usa el módulo
global `random` sin sembrar, a diferencia de `seal_test_set_es.py` que
sí usa `random.Random(SEED)` separados. Viola la restricción dura de
reproducibilidad del proyecto. Se corrige antes de relanzar la
generación ES.


## Diseño de V3: fine-tuning desde V1 con misma loss, sobre ES (19/08/2026)

De las tres alternativas que quedaron abiertas en la nota "EDA Common
Voice ES v26..." (17/08/2026) y reflejadas como pendiente en
`CLAUDE.md` — fine-tuning desde V1 con MSE puro, desde V2 con
MSE+SI-SDR, o from-scratch con EN+ES combinado — se decidió: **V3 =
fine-tuning desde el checkpoint de V1 (`checkpoints/v1/best.pt`), sobre
el dataset ES (`data/processed_es`), con la misma loss que V1** (MSE
pura sobre magnitud STFT, `mse_magnitude`, sin componente SI-SDR).

**Razones del diseño:**
- Responde directamente la pregunta central de V3: ¿el fine-tuning en
  español mejora el desempeño frente a audio en español? Comparar V1
  (EN) contra V3 (EN fine-tuneado a ES) con la loss constante aísla la
  variable idioma/dataset, que es la única que debe cambiar entre V1 y
  V3 según la regla de ablation limpio del proyecto (una sola variable
  por variante).
- Partir de V2 (MSE+SI-SDR) en vez de V1 mezclaría dos variables no
  atribuibles por separado: loss combinada Y fine-tuning en español. Esa
  combinación queda para V5 (V3 + PESQNet/Squim sobre ES), no para V3.
- From-scratch con EN+ES combinado es una pregunta de investigación
  distinta (¿un modelo entrenado conjuntamente en ambos idiomas
  generaliza mejor que uno fine-tuneado?) que no está en el alcance
  fijado para V3 en `docs/EXPERIMENTS.md` ("V1 + fine-tuning sobre
  Common Voice ES").

**Implicancias:**
- `training/trainer.py` necesita soporte para inicializar `CRN()` desde
  un `state_dict` existente en vez de pesos aleatorios, antes de poder
  correr V3.
- `training/config.py` necesita `CONFIG_V3`: mismo `loss` que
  `CONFIG_V1` (`mse_magnitude`, o ausencia de la key ya que
  `get_loss_name` defaultea a eso), `train_dir`/`val_dir` apuntando a
  `data/processed_es`, `checkpoint_dir` → `checkpoints/v3`, y un campo
  nuevo (ej. `init_checkpoint`) apuntando a `checkpoints/v1/best.pt`.
- Hiperparámetros de optimización (lr, batch, scheduler) se mantienen
  iguales a V1 salvo que el fine-tuning amerite ajuste explícito y
  documentado (ej. lr menor); cualquier cambio ahí es una desviación que
  debe registrarse acá si se decide.


## CuDNN determinism: activado por default, desactivado en el sweep de V3b (22/08/2026)

Al preparar el sweep de lr de V3b se agregó `torch.backends.cudnn.deterministic`
+ `np.random.seed()` a `training/trainer.py`, cerrando un gap real: `CLAUDE.md`
afirmaba "CuDNN determinism activado en trainer" pero el código nunca lo
tuvo — V1, V2 y V3 se entrenaron solo con `torch.manual_seed(42)` (sin
`cudnn.deterministic`), que da reproducibilidad de init de pesos y orden
lógico de datos, pero no bit-exactitud del cómputo en GPU (la suma en
punto flotante en paralelo no es asociativa; cuDNN puede autoseleccionar
distinto algoritmo entre corridas). No se corrige retroactivamente V1/V2/V3
— ya están taggeados y reportados — pero queda documentado como limitación
real de esos resultados.

**Costo medido**: con `cudnn.deterministic=True` cada época del sweep pasó
de ~1.886s (V3, sin el flag) a ~3.820s — el doble. Este modelo es 94% LSTM,
y los kernels deterministas de cuDNN para RNN son notablemente más lentos
que los default. El sweep completo hubiera tardado ~26-27h en vez de las
~12h estimadas.

**Decisión**: `cudnn_deterministic` queda como flag de config, default
`True` (la restricción dura se cumple por default para cualquier variante
nueva). Se desactiva explícitamente solo en `CONFIG_V3_SWEEP_BASE`
(`training/config.py`) — es la única config pensada para corridas
exploratorias descartables (comparación relativa entre 5 candidatos de lr,
no un resultado citable). `CONFIG_V3B` (el entrenamiento final, el que se
taggea) no define la clave y por lo tanto queda determinista por default.

**Razón**: en investigación/industria es práctica estándar no pagar el
costo de determinism durante hyperparameter search — las corridas del
sweep se descartan después de elegir el ganador, y si el ruido de punto
flotante alcanzara para cambiar cuál lr gana, esa diferencia ya sería
demasiado débil para ser una conclusión sólida de todos modos. Reservar
el determinism para la corrida que sí se reporta (V3b final) es donde
realmente importa poder decir "esto es reproducible bit-exact".

Bonus de velocidad: cuando `cudnn_deterministic=False`, el trainer activa
`cudnn.benchmark=True` — como todos los segmentos de audio miden
exactamente 64.000 samples (shape fija), cuDNN autotunea una vez el
kernel más rápido para ese tamaño y lo reusa.


## Diseño de V4: MSE + Squim (proxy aprendido) sobre PMSQE (aproximación cerrada), con monitoreo obligatorio de gaming (30/08/2026)

Antes de escribir código para V4 ("V1 + PESQNet loss (EN)" en el roadmap
de `EXPERIMENTS.md`) se evaluaron dos familias de loss perceptual
diferenciable candidatas, con literatura descargada y leída
íntegramente (carpeta `Papers V4`, no citas de segunda mano): **Squim**
(`torchaudio.pipelines.SQUIM_OBJECTIVE`, proxy aprendido — red entrenada
para predecir PESQ/STOI/SI-SDR sin referencia) vs **PMSQE** (Martín-Doñas
et al. 2018, IEEE SPL — aproximación de forma cerrada derivada
matemáticamente del propio algoritmo PESQ, con términos de disturbance
simétrico/asimétrico en escala Bark/sone).

**Decisión: V4 usa una loss combinada `mse_plus_squim` (MSE sobre
magnitud STFT + término Squim, ponderados), no PMSQE.** V4 es rama
paralela a V2 (no fine-tuning de V1, no stack sobre V2): init random,
mismo dataset EN y mismos hiperparámetros que V1, única variable que
cambia es la loss — misma disciplina de ablation limpio que V1→V2.

**Razones del diseño, con evidencia verificada (no intuición):**

1. **PMSQE pierde contra alternativas más simples en un benchmark
   independiente.** López-Espejo, Edraki, Chan, Tan & Jensen (2023,
   *Speech Communication* 150, "On the deficiency of intelligibility
   metrics as proxies for subjective intelligibility") entrenaron PMSQE
   contra SI-SDR, SI-SDR+preénfasis, STOI, ESTOI, STGI y STGI+SI-SDR
   sobre WSJ0, misma arquitectura FCNN. Resultado textual (p. 15):
   *"despite PMSQE being an approximation of PESQ (Martín-Doñas et al.,
   2018), $\mathcal{L}_{PMSQE}$ clearly yields the worst PESQ results
   among all the evaluated loss functions, which is consistent with
   previous findings in Kolbæk et al. (2020)."* Con números: a SNR
   −10dB, PMSQE (PESQ 1,30) casi no mejora sobre noisy (1,31), mientras
   SI-SDR solo ya llega a 1,55–1,90 en ese mismo rango; a 20dB, PMSQE
   queda en 2,99 contra 3,79 de STGI+SI-SDR. La aproximación
   matemáticamente más "fiel" a PESQ da, en la práctica, el peor PESQ de
   las seis losses comparadas.
2. **Optimizar directo contra ese mismo linaje de aproximación cerrada
   es explotable de forma demostrada.** de Oliveira, Welker, Richter &
   Gerkmann (2024, Interspeech, "The PESQetarian: On the Relevance of
   Goodhart's Law for Speech Enhancement") entrenaron un modelo contra
   `torch-pesq`, implementación diferenciable basada en Martín-Doñas
   2018 + Kim et al. 2019 — el mismo linaje que PMSQE. Resultado: PESQ
   3,82 (top-4 en el leaderboard de VB-DMD a la fecha del paper)
   mientras SI-SDR cae de +8,4dB (noisy) a **−19,8dB**, POLQA cae de
   3,11 a 1,46, y en la variante más agresiva (PESQ+SDR) el modelo
   aprende a insertar un click de amplitud extrema al inicio del audio
   que exploda un bug de estimación de nivel en el propio cómputo de
   PESQ (saturación por outlier en la suma de cuadrados + el click cae
   en la muestra 0, multiplicada por 0 por la ventana Hann del STFT
   interno de PESQ — invisible para la métrica). Un "modelo" de un solo
   parámetro (click fijo c=666, sin red neuronal) ya alcanza PESQ 3,46
   por sí solo.
3. **Squim es de otra familia (proxy aprendido, no fórmula cerrada) y ya
   está validada como diferenciable en este repo**
   (`tests/test_squim_differenciable.py`, gradient norm 3.11e-01).
   Quality-Net (Fu et al. 2018) — precedente de la misma familia, red
   BLSTM entrenada para predecir PESQ sin referencia — muestra
   correlación alta con PESQ real (LCC 0,905 en audio noisy/clean), con
   la salvedad de que esa correlación cae a LCC 0,816 específicamente
   sobre audio *procesado por un modelo de enhancement* — el régimen
   que importa acá.
4. **Combinada (MSE + Squim), no Squim puro.** En el propio experimento
   de PESQetarian, el modelo baseline entrenado solo con MSE fue el
   mejor en *todas* las demás métricas (POLQA, SI-SDR, ESTOI, DNSMOS)
   salvo PESQ — el ancla de reconstrucción importa. Quality-Net cita a
   su vez el precedente de Talebi & Milanfar (NIMA / learned perceptual
   image enhancement) de combinar assessment-loss con reconstruction-loss
   en vez de usar el assessment solo. `alpha` alto (MSE dominante, Squim
   como refinamiento) es la lectura conservadora de ambos papers.

**Salvaguardas que pasan de "buena práctica" a obligatorias para V4, con
justificación puntual:**

- `evaluation/monitor_correlation.py` debe medir la correlación
  Squim-vs-PESQ real **sobre la salida del CRN (audio enhanced) en cada
  checkpoint**, no solo sobre los pares noisy/clean del test set
  sellado — justificado por la caída de LCC 0,905→0,816 de Quality-Net
  específicamente en audio procesado. Medir solo sobre noisy/clean mide
  el régimen fácil del proxy, no el relevante.
- Chequeo cruzado por época: si el componente Squim de la loss mejora
  mientras el SI-SDR o el STOI de validación caen (por debajo del batch
  anterior o por debajo de Noisy), es señal de alarma de gaming — no una
  posibilidad remota, es el patrón de falla exacto que documentó
  PESQetarian (PESQ↑ con todo lo demás↓).
- Escucha dirigida de 5-10 archivos del test set al cierre de V4,
  motivada por dos papers independientes (no uno): buscar
  específicamente artefactos agudos/metálicos de alta frecuencia (firma
  reportada del modelo PESQetarian puro) y clicks o saltos abruptos de
  rango dinámico al inicio del archivo (firma del exploit de nivel del
  modelo PESQ-SDR). López-Espejo et al. 2023 aporta una segunda
  confirmación independiente, con metodología estadística real (26
  oyentes, test de Kruskal-Wallis), de que ganancias en la métrica de
  entrenamiento no garantizan mejora perceptual.

**Lo que sigue siendo hipótesis propia, no precedente publicado —
aclarado a propósito para no repetir el error de sobre-atribución
bibliográfica de un análisis previo de otro agente sobre este mismo
tema:** el mini-sweep de calibración de `alpha`/`squim_scale` y la
posibilidad de warm-start con MSE puro antes de introducir el término
Squim. Ningún paper de `Papers V4` prueba esto específicamente — es
diseño razonado a partir del riesgo de inestabilidad en época 1 (CRN con
pesos random alimentando audio-basura a Squim), y se valida
empíricamente con el mismo criterio que el sweep de lr de V3b, no se da
por sentado.

**Implicancias:**
- `training/losses.py` necesita `mse_plus_squim(mag_est, mag_clean,
  audio_est, squim_pesq_hat, alpha, squim_scale)`, mismo patrón que
  `mse_plus_sisdr`.
- `training/trainer.py` necesita cargar `SQUIM_OBJECTIVE` frozen,
  `.eval()`, con `torch.backends.cudnn.flags(enabled=False)` alrededor
  únicamente del forward de Squim (no de todo el training step) — la
  solución (1) validada en `tests/test_squim_differenciable.py`,
  preferida sobre `.train()` con pesos congelados porque evita que
  BatchNorm/estado interno de Squim mute entre corridas.
- `evaluation/monitor_correlation.py` no existe todavía — hay que
  construirlo con el diseño de dos regímenes descrito arriba antes de
  poder cerrar V4 de forma responsable.
- Detalle del roadmap completo, fases y orden de ejecución en
  `docs/PLAN_V4.md`.


## V4: por qué Squim queda fijo, sin warm-start desde V1 ni reentrenamiento
alternado del proxy (30/08/2026)

Al completar Fase 0/1/2 del plan de V4 se releyó Xu, Strake & Fingscheidt
(2022, *Deep Noise Suppression Maximizing Non-Differentiable PESQ
Mediated by a Non-Intrusive PESQNet*, IEEE/ACM TASLP — carpeta `Papers`,
no `Papers V4`) con más detalle, porque es estructuralmente el paper más
parecido a `mse_plus_squim`: su loss (ec. 9) es literalmente
`α·MSE + (1−α)·PESQNet_loss`, misma convención de `α` que la nuestra
(`α` pesa el término MSE).

**Hallazgo que tensiona la decisión del 30/08 de arriba:** Xu barre
`α ∈ {0, 0.5, 1}` en su segunda etapa de fine-tuning y encuentra que
`α=0` (0% MSE, 100% término perceptual) da el mejor resultado — PESQ
3.45 vs. 3.37 del baseline MSE puro en DNS1-dev, mejor DNSMOS en casi
todas las condiciones. `α=1` ("placebo") rinde igual que el baseline
MSE. Esto contradice, en el paper estructuralmente más cercano al
nuestro, la lectura "alpha alto es la opción conservadora" que se venía
aplicando (basada en PESQetarian y Quality-Net/NIMA, ver entrada de
arriba).

**Por qué ese resultado no es transferible sin más — dos diferencias de
protocolo, no de arquitectura:**

1. El modelo de Xu parte de un checkpoint **ya preentrenado con MSE**
   antes del barrido de `α` (fine-tuning de segunda etapa) — no de init
   random como V4.
2. Usan un **protocolo de entrenamiento alternado**: DNS y PESQNet se
   reentrenan turnándose a nivel de época (con gradient accumulation
   para estabilizar), de forma que el PESQNet nunca queda "stale"
   respecto a la distribución de salidas del DNS actual. Citan su propio
   trabajo previo (`FCRN/PESQNet [24]`) donde, sin ese esquema alternado
   — PESQNet **fijo**, igual que nuestro Squim — la mejora de PESQ fue
   "muy limitada... incluso ninguna mejora en datos sintéticos". Ese es
   el régimen que V4 tiene planeado.

**Decisión: NO se implementa el esquema alternado de Xu, ni se
warm-startea V4 desde `checkpoints/v1/best.pt`.** Razones:

- **Squim no está construido para esto.** `SQUIM_OBJECTIVE` es un
  pipeline de inferencia de torchaudio con pesos publicados fijos, sin
  receta de fine-tuning documentada. Replicar el esquema de Xu
  requeriría computar PESQ real (no diferenciable, vía el paquete
  `pesq` ya usado en `evaluation/metrics.py`) sobre la salida del CRN
  periódicamente y reentrenar los pesos de Squim contra eso — no es un
  flag de config, es reimplementar el contenido central de ese paper
  (su título es literalmente "Novel Training Loss/**Protocol**"), fuera
  de alcance para una sola variante de un ablation study de tesis.
- **Rompe la disciplina de ablation del proyecto.** V4 está fijado como
  rama paralela a V2 (init random, única variable: la loss). Un
  warm-start desde V1 o una alternancia de reentrenamiento cambiarían
  el mecanismo de entrenamiento además de la loss — no se podría
  atribuir un resultado a una sola causa, violando la restricción dura
  de `CLAUDE.md` ("una sola variable cambia por vez").
- **Desproporcionado al presupuesto de cómputo/tiempo disponible.** Xu
  corre 25 épocas de fine-tuning alternado sobre un modelo ya
  preentrenado, más una etapa de pretraining separada del propio
  PESQNet — sobre una sola GPU (RTX 4060) y con V5 + comparaciones
  contra RNNoise/DeepFilterNet2 todavía pendientes antes del 29/12/2026.
- **La mitigación elegida es detectar, no prevenir — y ya está
  planeada.** `evaluation/monitor_correlation.py` (Fase 4) + el chequeo
  cruzado por checkpoint (Fase 7) + la escucha dirigida (Fase 8) no
  evitan que el proxy quede stale, pero lo detectan si pasa. Es una
  estrategia proporcional al alcance de un TP de grado: si aparece
  gaming, es un resultado negativo caracterizado y válido como material
  de tesis (ya lo dice el criterio de cierre de `docs/PLAN_V4.md`), no
  hay que perseguir "que funcione" reimplementando el mecanismo
  completo de otro paper.

**Consecuencia práctica para Fase 3:** como V4 no tiene la red de
seguridad que hizo segura la zona perceptual-dominante en Xu, la
calibración de `alpha` se mantiene en zona MSE-dominante — doblemente
justificado ahora (PESQetarian/Quality-Net **y** ausencia del mecanismo
estabilizador de Xu) — pero se amplía la grilla del sweep de
{0.95, 0.9, 0.8} a {0.95, 0.9, 0.8, **0.7**} para poder ver si aparece
una inflexión visible en el chequeo cruzado (SI-SDR/STOI cayendo) antes
de acercarse a esa zona, sin llegar a probar el régimen de Xu sin su
mecanismo estabilizador.

El calentamiento "con/sin warm-start" que ya contemplaba Fase 3 (2-3
épocas de MSE puro antes de activar el término Squim, dentro de la
misma corrida random-init) no es lo mismo que lo descartado acá y se
mantiene sin cambios — es un detalle de curriculum dentro de V4, no un
cambio del punto de partida ni un mecanismo de reentrenamiento del
proxy.


## Paso 1 de V4 acotado a 3 épocas (30/08/2026)

`scripts/squim_scale_sweep.py` (Paso 1 del mini-sweep de Fase 3) estaba
escrito con `N_EPOCHS_SWEEP = 4`, presupuesto medido ~14.8h (control
~1.6h + 4 candidatos × ~3.3h, a razón de ~24.2 min/época sin Squim y
~49.5 min/época con Squim — números de la propia Fase 3). Se acotó a
**3 épocas**, bajando el presupuesto a **~11.1h** (control ~1.2h + 4
candidatos × ~2.47h), para reducir el costo de cómputo de una corrida
exploratoria/descartable sin cambiar el diseño (misma cobertura de
alpha, mismo control, mismo criterio de selección).

**Por qué 3 y no menos:** con 2 épocas el chequeo de inestabilidad en
época 1 (parte del criterio de selección de Paso 1 y del criterio para
decidir si Paso 2/warm-start es prioritario) queda con un solo punto de
comparación posterior — insuficiente para distinguir tendencia real de
ruido entre épocas. 3 épocas conserva ese margen a un costo intermedio
entre las ~14.8h originales y las ~7.4h de un recorte a 2.

**Implicancia en Paso 2 (`scripts/squim_warmstart_check.py`):** estaba
hardcodeado a 2 épocas de warmup (`mse_magnitude`) + 2 épocas de
`mse_plus_squim` = 4 total, exactamente para igualar el presupuesto de
los candidatos de Paso 1 y poder comparar "con warm-start" vs "sin
warm-start" al mismo total de épocas (ver Fase 3 de `PLAN_V4.md`). Bajar
Paso 1 a 3 sin tocar Paso 2 hubiera roto esa paridad. Se ajustó a 1
época de warmup + 2 épocas de `mse_plus_squim` (total 3) — se mantiene
al menos 1 época de warmup porque esa es la mitigación completa del
riesgo que motiva Paso 2 (inestabilidad en época 1 por audio-basura del
CRN random alimentando a Squim), y se prioriza no recortar más las
épocas con Squim activo (2, igual que antes) ya que son las que
efectivamente comparan contra el resultado sin warm-start de Paso 1.


## Cierre de V4: Squim frozen confirma Goodhart's Law con datos propios — no se avanza con `mse_plus_squim` (31/08/2026)

Paso 1 del mini-sweep (5 corridas, 3 épocas, `scripts/squim_scale_sweep_evaluate.py`) dio resultado
inequívoco sobre `test_v1_en` (n=250):

| Corrida | PESQ-NB Δ vs Noisy | STOI Δ | SI-SDR Δ |
|---|---|---|---|
| control_mse (α=1.0, sin Squim) | **+0.081** | +0.027 | +4.15 dB |
| alpha_095 | −0.343 | −0.069 | +2.29 dB |
| alpha_090 | −0.806 | −0.169 | −0.63 dB |
| alpha_080 | −0.943 | −0.252 | −4.00 dB |
| alpha_070 | −0.846 | −0.251 | −4.56 dB |

`control_mse` es la única corrida que mejora sobre Noisy en las cuatro métricas reales. Las cuatro
corridas con Squim degradan, con una relación dosis-respuesta razonable en `alpha` (más peso a Squim,
peor resultado real, salvo un cruce menor entre 070/080). **Ningún candidato pasa el criterio de
selección de Fase 3** (PESQ-NB real por encima del control, sin que SI-SDR/STOI reales caigan por
debajo).

**Diagnóstico, con evidencia propia (no solo cita bibliográfica):** los `history.json` de cada corrida
muestran que `val_squim` (≈ `-pesq_hat` estimado por Squim sobre la salida del propio modelo) llega a
−4.0/−4.4 hacia la época 3 — Squim cree que su propia salida tiene PESQ ≈ 4.0–4.4, mejor que cualquier
variante real entrenada en este proyecto. El PESQ real medido cae a 1.1–1.8, **por debajo de Noisy**
(2.15). El patrón ya está instalado desde época 1 en las cinco corridas (incluidas `alpha_090`/`alpha_095`,
que el chequeo automático marcó "sin señales de inestabilidad") — no es un pico transitorio de pesos
random que warm-start pueda mitigar, es un exploit que se refuerza cada época. Coincide con el patrón
de gaming documentado en de Oliveira et al. 2024 (PESQetarian), ya citado en la entrada de diseño de V4
del 30/08.

**Corrección a una hipótesis explorada y descartada:** se evaluó si el fallo tenía además una causa
distribucional específica — la hipótesis de que `SQUIM_OBJECTIVE` nunca vio, en su propio entrenamiento,
salidas de una red parcialmente entrenada con artefactos propios (ruido musical, huecos espectrales),
y que por eso evalúa "fuera de distribución" desde el primer batch. **Se verificó contra la fuente
primaria (Kumar et al. 2023, sección 4.4.1) y no se sostiene tal como se planteó**: el dataset de
entrenamiento de Squim (DNS Challenge 2020) incluye explícitamente salidas de sistemas de enhancement
basados en arquitectura GCRN "with varying degree of performances due to different configurations" —
es decir, Squim sí vio salidas de enhancement imperfectas como parte de su entrenamiento. No se agrega
esta hipótesis a los hallazgos confirmados del proyecto; el diagnóstico de Goodhart's Law/gaming (arriba)
es el que queda respaldado por evidencia directa.

**Precisión sobre el mecanismo (agregada 31/08/2026 tras releer Xu et al. 2022, sección III):** la
pregunta "¿la distribución de entrenamiento de Squim era demasiado angosta?" resultó ser la pregunta
equivocada, pero por una razón más interesante que la de arriba. Xu, describiendo el trabajo previo de
Fu et al. [31], reporta el mismo fenómeno que observamos: *"the fixed PESQNet was reported to be **fooled**
by the updated DNS (**estimated PESQ scores increase while true PESQ scores decrease**) after training
for several minibatches. In [31], this was mainly caused by the fixed Quality-Net **not having seen the
enhanced speech signal generated by the updated DNS**."* La causa que identifica no es la amplitud del
dataset original del proxy, sino que el proxy queda **stale** respecto de un modelo que se mueve: el
DNS deriva hacia regiones que ningún dataset fijo cubre, por ancho que sea. Nuestro resultado es una
replicación independiente de [31] con otro proxy (Squim en vez de Quality-Net) y otra arquitectura.

**Limitación de esta evaluación (detectada 31/08/2026, aplica retroactivamente):** los cinco checkpoints
de Paso 1 se evaluaron desde `best.pt`, que `Trainer` elige por `val_loss` mínima — y para
`mse_plus_squim` esa `val_loss` **incluye** el término `(1-α)·(-pesq_hat)`. Más gameado = mayor
`pesq_hat` = menor `val_loss` = "mejor" checkpoint. El criterio de selección está contaminado por
exactamente lo que se quería medir: `alpha_070` guardó la época 3 (`val_squim` −4.43, el más inflado) y
`alpha_080` la época 2. No cambia la conclusión — la tendencia era monótona en las tres épocas de las
cuatro corridas — pero los números de la tabla de arriba son, en rigor, los del punto más gameado de
cada corrida. V4b (abajo) evalúa época por época para no repetir el sesgo.

**Decisión:** se cierra la línea `mse_plus_squim` para V4. No se corre Paso 2 (warm-start) — ataca
inestabilidad de época 1, que no es la causa observada. No se sub-barre `squim_scale` — el desbalance
entre `mse_component` y `squim_component` crece *durante* el entrenamiento (autoreforzado), no es una
mala calibración estática que un factor de escala distinto resuelva. Construir un proxy PESQ propio
(entrenado explícitamente sobre salidas de redes parcialmente entrenadas, no solo mezclas ruidosas
naturales) queda como posibilidad de investigación abierta, a intentar solo si sobra tiempo después de
GCRN (ver `docs/PLAN_GCRN.md`) — no es la prioridad actual.

**Próximo paso (actualizado 01/09/2026):** antes de cerrar la línea del proxy se corrió V4b (abajo),
que prueba el protocolo estabilizado de Xu sin reentrenar el proxy. Después de V4b se pasa a GCRN
(Tan & Wang 2020, IEEE/ACM TASLP) — evolución directa del propio paper base del proyecto (Tan & Wang
2018), con complex spectral mapping + GLU + LSTM agrupada. Plan completo en `docs/PLAN_GCRN.md`.


## V4b: el protocolo estabilizado de Xu reduce el daño un tercio, pero no rescata el término perceptual — cierre definitivo de la línea de proxy (01/09/2026)

V4 corrió el régimen máximamente inestable (init random, lr 2e-4, update por minibatch). Xu et al. 2022
(sección IV-B) usa cuatro elementos estabilizadores; **tres no requieren reentrenar el proxy**, y V4 no
tenía ninguno. V4b los prueba. Script: `scripts/v4b_protocol_check.py` + `..._evaluate.py`. 3 corridas ×
3 épocas = 8.02h, arranque desde `checkpoints/v1/best.pt`, dataset EN, `save_every_n_epochs=1`.

**Qué se probó y qué no:**
- ✅ Arranque desde modelo convergido con MSE (`init_checkpoint`, no init random).
- ✅ lr 10x más bajo en la etapa perceptual (2e-5, el valor de Xu, vs. 2e-4 del proyecto).
- ❌ Gradient accumulation a **un update de pesos por época**. Descartado con justificación, no por
  descuido: con 3 épocas serían 3 updates totales partiendo de V1 convergido — no se movería nada y el
  nulo sería trivial. Incompatible con el presupuesto de cómputo.
- ❌ Alternancia ⟨1-1⟩ reentrenando el PESQNet — ya descartada por alcance (entrada del 30/08).

**Diseño 2×2 (tres celdas; la cuarta, init random + lr 2e-5, es inútil porque casi no entrenaría).**
Motivado explícitamente por la lección de V3b (épocas y lr quedaron confundidos y no se pudo atribuir):

| Δ PESQ-NB vs Noisy | lr = 2e-4 | lr = 2e-5 |
|---|---|---|
| **init random** | −0.806 (`alpha_090`, Fase 3) | — |
| **init V1 convergido** | −0.263 (`v4b_lr2e4`) | **−0.059** (`v4b_main`) |

**La tabla contra Noisy engaña, y por eso el placebo de Xu era imprescindible.** −0.059 parece "casi
arreglado", pero el punto de partida cambió: V1 convergido ya daba +0.497. El control correcto es
`v4b_placebo` (α=1, mismo `init_checkpoint`, mismo lr 2e-5, mismas épocas, sin término perceptual):

| corrida (época 3) | ΔPESQ-NB | ΔPESQ-WB | ΔSTOI | ΔSI-SDR | `pesq_hat` | brecha |
|---|---|---|---|---|---|---|
| `v4b_placebo` (α=1) | **+0.539** | +0.486 | +0.054 | +6.10 dB | — | — |
| `v4b_main` (α=0.9, lr 2e-5) | −0.059 | −0.006 | −0.029 | +3.83 dB | 3.936 | +2.36 |
| `v4b_lr2e4` (α=0.9, lr 2e-4) | −0.263 | −0.132 | −0.048 | +3.32 dB | 4.217 | +2.77 |

**Resultado central: el término Squim cuesta −0.598 PESQ-NB** respecto del mismo fine-tuning sin él.
Comparando el daño *atribuible* entre ambos regímenes:

- V4 (init random, lr 2e-4): `control_mse` +0.081 vs `alpha_090` −0.806 → **−0.887**
- V4b (V1 convergido, lr 2e-5): `v4b_placebo` +0.539 vs `v4b_main` −0.059 → **−0.598**

Los dos estabilizadores recuperan ~33% del daño; **dos tercios del gaming sobreviven al protocolo de Xu
menos la alternancia**. Es evidencia directa de que lo que hace funcionar el esquema de Xu es
específicamente el reentrenamiento del proxy — consistente con lo que él mismo afirma y con lo que
reporta de su trabajo previo [24]/[31] para proxy fijo.

**La brecha descarta el confound de descalibración.** Se había marcado que parte de la brecha
`pesq_hat − PESQ-WB real` podía ser error de calibración de Squim y no gaming. No es el caso: el MAE
nominal de Squim para WB-PESQ es **0.142** (Kumar et al. 2023, Tabla 2, "Ours with MTL"), y las brechas
observadas (+2.36 y +2.77) son ~16-20× ese error, creciendo monótonamente época a época en las dos
corridas con Squim (2.04→2.30→2.36 y 2.59→2.70→2.77). Nota metodológica: la comparación se hace contra
PESQ-**WB**, no NB, porque la cabeza de Squim estima WB-PESQ (Kumar et al. 2023: *"the term PESQ will
refer to WB-PESQ throughout this paper"*), acotado por construcción en [1, 4.64] vía sigmoide (ec. 5).
`v4b_lr2e4` llegó a 4.217, a 0.42 puntos del techo estructural.

**Hallazgo lateral con valor propio: V1 no estaba del todo convergido.** `v4b_placebo` en la época 2
(V1 + 2 épocas a lr 2e-5, sin ningún cambio de loss) da PESQ-NB **2.716 / PESQ-WB 2.102 / STOI 0.906 /
SI-SDR 13.945**, mejor que V1 (**2.650 / 2.021 / 0.904 / 13.805**) en las cuatro métricas — pese a que su
`val_loss` MSE es *peor* (0.0836 vs 0.0724). El criterio de selección de checkpoint por mínimo de val MSE
no está alineado con PESQ. Dos implicancias: (a) los números reportados de V1 son levemente pesimistas;
(b) el placebo es un control fuerte, no un modelo degradado, lo que refuerza la atribución de −0.598.

**Decisión: se cierra definitivamente la línea de proxy perceptual (V4/V4b), sin más variantes.** El
espacio de decisión quedó cubierto: régimen inestable (V4, destruye el modelo) y régimen estabilizado
según la propia literatura (V4b, sigue destruyendo dos tercios). Lo único no probado es la alternancia
de Xu, ya descartada por alcance con razones documentadas. Construir un proxy PESQ propio queda como
posibilidad remota, solo si sobra tiempo después de GCRN (`docs/PLAN_GCRN.md`, Fase 10).

**Resultado para el informe:** el término perceptual con proxy fijo no aporta ni bajo el protocolo
estabilizado que prescribe la literatura; la degradación catastrófica del régimen inestable es
atribuible en ~1/3 a init random + lr alto, y en ~2/3 al gaming del proxy en sí. Eso es atribución
causal con controles, no observación bruta.

**Limitaciones de V4b:** `cudnn_deterministic=False` en las tres corridas, elegido a propósito para ser
comparable con `alpha_090` (que se corrió así en el sweep de Fase 3) — ninguna de las dos es bit-exacta.
3 épocas por corrida, no 25 como la segunda etapa de Xu.
