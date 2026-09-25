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


## GCRN descartado como línea de trabajo; el próximo experimento es el control de idioma vs canal (06/09/2026)

Las entradas de V4 (31/08) y V4b (01/09) de más arriba en este mismo archivo cierran la
línea de proxy perceptual y señalan a GCRN (Tan & Wang 2020) como el paso siguiente,
remitiendo a `docs/PLAN_GCRN.md`. **Ese plan se descarta y el archivo ya no está en el
repositorio.** Las referencias de aquellas entradas quedan como estaban: eran ciertas en
su fecha, y reescribirlas hacia atrás falsificaría la bitácora.

**Por qué se descarta.** GCRN es una mejora de arquitectura —complex spectral mapping,
GLU, LSTM agrupada— sobre un eje que no es el aporte del proyecto. Ninguna de las cuatro
afirmaciones diferenciadoras depende de ella: el cambio de signo de la transferencia según
el SNR, el trade-off adaptación/olvido, la validación downstream con emparejamiento de
idioma y el despliegue causal en el hardware objetivo se sostienen o se caen con el CRN
actual. Mejor PESQ absoluto no defiende ninguna, y las horas de GPU compiten directamente
contra los dos experimentos que sí las alimentan, con la defensa fijada al 29/12/2026.

**Qué cambió la prioridad.** El reanálisis estadístico del 05-06/09 movió dos cosas.
Primero, expuso que el contraste EN/ES actual confunde idioma con canal de grabación
—LibriSpeech son audiolibros en FLAC, Common Voice es MP3 crowdsourced— y que ese confound
tiene un paper dedicado en contra: Wang, Lee, Tsao & Wang (Interspeech 2022,
*Disentangling the Impacts of Language and Channel Variability on Speech Separation
Networks*) concluye que el canal domina y el idioma es despreciable. Es la objeción más
fuerte que existe contra el aporte 1 y hoy no tiene respuesta. Segundo, la familia F5
mostró que el olvido en inglés es real y está concentrado en un subconjunto identificable
de archivos, lo que le devuelve sentido al experimento de fine-tuning con capas congeladas
con un endpoint medible. Los dos rinden más que reproducir una arquitectura de 2020.

**Único encuadre en que GCRN valdría la pena, si sobra tiempo.** No como búsqueda de mejor
PESQ sino como chequeo de validez externa: si el patrón de decaimiento por SNR se reproduce
en una arquitectura distinta, el hallazgo deja de ser atribuible a una particularidad del
CRN. Esa es una pregunta que un jurado va a hacer igual, y así GCRN la contesta en vez de
competir. Con ese encuadre habría que rehacer el plan, no recuperarlo.

**Qué se conserva del trabajo hecho.** La lectura del paper y del repo oficial fue real y
verificada: kernel temporal de tamaño 1 (causal sin necesitar el truco de truncamiento que
usa `models/crn.py`), STFT bit-exacta con la convención del proyecto (n_fft=320, hop=160,
Hamming), y GCRN(G=2) con 23.82M MACs/frame contra los 25.27M del CRN actual. Si alguna vez
se revive la línea, ese análisis está en el mensaje de este commit y no hay que rehacerlo
desde cero.

**Qué lo reemplaza.** Sellar `test_v3_mls_es` con voz de Multilingual LibriSpeech español
—LibriVox, el mismo paradigma de grabación que LibriSpeech— reusando verbatim las
condiciones de ruido de `test_v1_en`, y evaluar V1, V2 y V3e sin reentrenar nada. El 2×2
resultante permite separar el efecto de idioma del de canal en las mismas unidades. Las
predicciones y los criterios de decisión están preregistrados fuera del repositorio, con
el hash comprometido en `docs/preregistro_mls_es.sha256` antes de que el dato exista.


## V5: la contingencia de R01 rinde más que la línea que reemplaza (07/09/2026)

**Diseño.** V5 es la "propuesta completa" del anteproyecto, redefinida. Su definición
original era «español + PESQNet»; PESQNet/Squim se descartó con evidencia propia (ver
entradas de V4 y V4b). Era el riesgo R01 de la matriz —el más alto, con VME 3,5 días— y su
contingencia declarada era caer a la loss combinada de V2. Esto es exactamente eso.

**V5 NO es una celda del ablation, y hay que enunciarlo así.** La cadena V1→V2→V3e queda
cerrada y es la que da la atribución causal, una variable por vez. V5 combina lo que
funcionó y su rol es ser el mejor sistema. Dos cambios sobre V3e:

1. **Arranca de `checkpoints/v2/best.pt` y conserva su loss combinada MSE + SI-SDR.** Van
   juntos: fine-tunear con MSE puro desde un checkpoint que ganó +0,20 PESQ gracias a la
   loss combinada desharía la mitad de esa ganancia. V3/V3b/V3e partieron de V1 con MSE puro
   por disciplina de ablation, así que la línea española nunca había heredado la mejora de V2.
2. **Entrena sobre `data/processed_es_wide`**, con SNR ~ uniform(-5, 20). El bucket [15,20]
   de los test sellados quedaba entero fuera de la distribución de entrenamiento, y es donde
   vive el hallazgo principal. Se declaró por adelantado que esto podía ATENUAR el efecto
   observado: es un test de robustez, no un amplificador.

Receta de optimización: la de V3e (lr 1e-4, 25 épocas, decay en la 12).

**Un tercer cambio que se evaluó y se descartó: `center=False`.** Se creía que el lookahead
de 10 ms venía del padding simétrico de la STFT. Se midió y no: el retardo sigue
`n_fft − hop` y es idéntico con padding causal, porque lo impone el overlap-add de la
síntesis, no el análisis. Verificado sobre cinco configuraciones de n_fft/hop en
`tests/test_causality.py`. Como es la configuración STFT que especifica el paper base
(ventana 20 ms, hop 10 ms), el sistema de referencia tiene idéntica latencia: no hay
desviación, había una afirmación mal escrita en `CLAUDE.md`. V5 no lleva el cambio.

**Screening de lr en vez de sweep, con criterio declarado antes de correr.** Tres corridas
de 3 épocas (5e-5, 1e-4, 2e-4). La lección de V3b es que en este modelo un sweep corto elige
por convergencia rápida y se equivoca: su ganador (5e-5) terminó peor que el candidato que a
5 épocas no había convergido. Por eso el criterio escrito fue *descartar* el lr que degrade
o diverja, y si los tres pasan usar 1e-4 por la evidencia de las corridas convergidas, NO
por cuál queda más alto a 3 épocas.

Resultado sobre `test_v2_es`: 5e-5 → 2,568; 1e-4 → 2,567; 2e-4 → 2,572. **Ninguna diferencia
sobrevive un test apareado** (p entre 0,44 y 0,87). Se aplicó la regla y **se eligió 1e-4
aunque 2e-4 quedó nominalmente más alto** — que es exactamente la situación para la que se
escribió. Dato de respaldo: 2e-4 tuvo el `val_mse` de la época 1 en 0,0924 contra ~0,085 de
los otros dos, la firma de un lr demasiado agresivo desde un checkpoint asentado.

**Resultado.** 25 épocas, 13,2 h, `best.pt` en la época 21 (val_loss −0,0792).

| test set | ruidoso | V1 | V2 | V3e | **V5** |
|---|---|---|---|---|---|
| español, Common Voice | 2,161 | 2,330 | 2,451 | 2,551 | **2,686** |
| inglés, LibriSpeech | 2,152 | 2,650 | 2,849 | 2,619 | **2,774** |
| español, audiolibro | 2,195 | 2,602 | — | 2,686 | **2,840** |

En español: **+0,136 sobre V3e** (p = 2e−17) y **+0,235 sobre V2** (p = 1e−29), apareado, y
mejora en las cuatro métricas. **Cruza OP-1 (PESQ ≥ 2,5) en los tres test sets**, objetivo
que hasta ahora solo se cumplía en inglés.

**El olvido cambia de referencia y hay que reportarlo con cuidado.** V3/V3b/V3e partían de V1,
así que su olvido se medía contra V1. V5 parte de V2, así que su referencia es V2:

- contra V2: **−0,075** (p = 0,004), con **16,8 %** de archivos que caen más de 0,2
- contra V1: **+0,125** (p = 1e−15), con 9,2 % de archivos rotos

Las dos son ciertas y dicen cosas distintas. La honesta es la primera —V5 pierde algo
respecto de su propio punto de partida— pero no hay que omitir la segunda: **V5 es mejor que
V1 en inglés**, así que el fine-tuning al español no dejó al modelo peor que el baseline
reproducible del paper. El 16,8 % está más cerca de V3 (17,6 %) que de V3e (11,6 %), lo cual
es coherente: V5 corrió la receta más agresiva durante 25 épocas.

**Limitación por error de configuración, declarada.** `CONFIG_V5` no definió
`save_every_n_epochs`, así que solo quedó `best.pt`. En V4b se vio que el criterio de
selección por mínima `val_loss` no está alineado con PESQ, y en V5 no se va a poder chequear
si alguna otra época daba mejor PESQ que la 21. Las réplicas de semilla sí guardan cada 3
épocas para cubrir ese hueco.


## El control preregistrado idioma/canal: el efecto por SNR es lingüístico, la adaptación es mixta (07/09/2026)

Se evaluaron V1, V3e y V5 sobre `test_v3_mls_es` (español, canal de audiolibro, condiciones
de ruido reusadas verbatim de `test_v1_en`). Las predicciones y los criterios de decisión
estaban escritos antes de que el test set existiera; el hash del preregistro está commiteado
en `docs/preregistro_mls_es.sha256`. Toda la inferencia va **agrupada por hablante**: son ~6
pares por cada uno de los 40 hablantes y los pares no son independientes.

**P1 — la pendiente contra el SNR sigue al idioma, no al canal. Criterio: IDIOMA.**

| V1 evaluado en | idioma | canal | rho |
|---|---|---|---|
| `test_v1_en` | inglés | audiolibro | −0,065 (n.s.) |
| `test_v2_es` | español | crowdsourced | −0,262 |
| `test_v3_mls_es` | español | audiolibro | **−0,212**, IC95 agrupado [−0,309, −0,115] |

El criterio preregistrado pedía rho ≤ −0,15 con el intervalo sin cruzar cero: se cumple.
Cambiar el **canal** manteniendo el idioma mueve la pendiente de −0,262 a −0,212; cambiar el
**idioma** manteniendo el canal la lleva a −0,065, indistinguible de cero. Es la respuesta
directa a Wang et al. 2022 en las unidades del proyecto: en este régimen el idioma no es
despreciable frente al canal.

**P2 — la ganancia del fine-tuning transfiere entre canales, pero atenuada. Criterio: PARCIAL.**

Ganancia de V3e sobre V1: **+0,221** en Common Voice contra **+0,084** en audiolibro
(IC95 agrupado [+0,056, +0,112]), con `prop_improve` 0,748 (IC95 [0,686, 0,806]). El umbral
para "aprendió idioma" era media > +0,10 **y** prop_improve > 0,70: la proporción pasa, la
media no. Cae en la zona intermedia declarada.

Lectura: tres de cada cuatro archivos mejoran y el intervalo no toca cero, así que la
adaptación **sí** es lingüística en parte — pero pierde ~62 % de su magnitud al cambiar de
canal. **Lo que V3e aprendió fine-tuneando sobre Common Voice es en parte español y en parte
ese canal.** Eso no estaba en las afirmaciones del proyecto, que hablan de adaptación al
idioma, y obliga a reformular el aporte 2.

**Hallazgo lateral, y es el más accionable: cuánto se retiene depende fuertemente de la receta.**

> **RETRACTADO el 09/09/2026.** El estimando de esta tabla está mal elegido y el 67 % no se
> sostiene. Ver la corrección al final de este archivo antes de citar cualquier número de acá.

| modelo | ganancia en Common Voice | en audiolibro | retiene |
|---|---|---|---|
| V3e | +0,221 | +0,084 | **38 %** |
| V5 | +0,356 | +0,239 | **67 %** |

V5 conserva casi el doble entrenando sobre un solo dataset, igual que V3e. Lo que cambió fue
el punto de partida y el rango de SNR. Así que la dependencia del canal no es una propiedad
inevitable del fine-tuning: es una propiedad de la receta, y se puede reducir. Queda como
métrica para evaluar cualquier intento futuro de invariancia al canal (por ejemplo
augmentación espectral al estilo de Braun & Tashev 2020, sección 5).

**Limitación estructural, declarada en el preregistro antes de sellar.** 40 hablantes contra
los ~248 de Common Voice, y no es corregible: un corpus de audiolibros son pocos lectores
leyendo mucho, así que el canal que se quiere fijar causa esa estructura. La dirección del
confusor se declaró por adelantado: 40 lectores de audiolibro son un dominio más fácil, así
que **P1 es conservador** —el efecto sobrevive a pesar de la condición favorable— pero **P2
es ambiguo en esa dirección**: parte de la atenuación podría ser diversidad de locutor y no
canal.


## Corrección: el costo del `cudnn_deterministic` no reproduce (07/09/2026)

La entrada del 22/08/2026 afirma que activar `torch.backends.cudnn.deterministic` cuesta ~2x
por época en este modelo, midiendo 1.886 s sin el flag contra 3.820 s con él durante el sweep
de V3b. **Esa medición no se reproduce.** Tiempos por época medidos desde entonces:

| corrida | determinism | loss | min/época |
|---|---|---|---|
| V2 | ON | combinada | 31,5 |
| V3e | ON | MSE | 31,7 |
| screening V5 | OFF | combinada | 32,0 |
| V5 | ON | combinada | **31,0** |

Con el flag activo la época tarda lo mismo que sin él. Los 3.820 s del sweep son el dato
anómalo —probablemente había otra carga en la máquina— y no el costo real del determinism.

Dos implicancias: **V3e corrió efectivamente determinista** (no hay una afirmación de
reproducibilidad falsa sobre una variante taggeada), y **desaparece la justificación para
desactivar el flag en corridas exploratorias**. De acá en más conviene dejarlo activo por
defecto en todo, salvo que una medición nueva vuelva a mostrar un costo real.


## El rendimiento por bucket de SNR es plano en inglés: la U invertida era efecto techo (06/09/2026)

La mejora cruda de PESQ es máxima en SNR medio y cae en los extremos, lo que sugiere que
convendría reforzar los buckets medios en el entrenamiento. **Normalizando por el margen
disponible para mejorar** (techo nominal de PESQ-NB 4,5 menos el PESQ del ruidoso), la
conclusión se invierte:

| bucket | PESQ ruidoso | margen | V1 inglés | V1 español | V3e español |
|---|---|---|---|---|---|
| [5, 10] | 1,97 | 2,53 | **24,7 %** | 14,8 % | 21,4 % |
| [10, 15] | 2,41 | 2,09 | **25,0 %** | 3,4 % | 19,8 % |
| [15, 20] | 2,96 | 1,54 | **22,3 %** | **−11,4 %** | 8,6 % |

**En inglés el rendimiento normalizado es plano** entre los buckets 2 y 4: el modelo recupera
un ~22-25 % del margen disponible en todo el rango. La caída del delta crudo de +0,625 a
+0,343 no es que funcione peor a SNR alto, es que queda mucho menos para ganar. **En español
no se disuelve**: 14,8 % → 3,4 % → −11,4 %.

**Implicancia de diseño: no conviene reforzar los buckets medios en el entrenamiento.** En
inglés no hay nada que corregir ahí, y en español el problema no es distribucional sino
lingüístico. Reforzar el centro acentuaría el pico en vez de llenar los extremos. La palanca
correcta para los extremos es más datos en los extremos, que es lo que hizo el cambio a
uniform(-5, 20).

**Y elimina la explicación alternativa más obvia del hallazgo principal.** Ante la objeción
"la caída a SNR alto en español es porque no queda margen", la respuesta es que con el margen
idéntico el inglés recupera el 22,3 % y el español da −11,4 %. Mismo margen, resultado
opuesto. Es el mismo tipo de refutación condicionada al confusor que F5.d aplicó al nivel
basal en el eje de olvido.

Salvedad a declarar: el techo de 4,5 es el máximo nominal de PESQ-NB y la normalización
asume margen lineal, lo cual es una simplificación. Va como análisis complementario, no como
métrica principal.


## `make_mixtures.py` no tenía guard de sobrescritura (06/09/2026)

`make_pairs` escribía con `mkdir(exist_ok=True)` sin verificar si el destino ya tenía
contenido. Un `--lang es` distraído habría pisado en silencio los 13 GB de
`data/processed_es` —la fuente con la que se entrenaron V3, V3b y V3e— rompiendo la
reproducibilidad de tres variantes ya taggeadas. Es una violación directa de la restricción
dura del proyecto que estuvo latente desde el sprint de agosto.

Ahora aborta si el destino tiene contenido, el rango de SNR pasó de literal enterrado a
constantes con su justificación, y el modo de rango ancho escribe a un directorio separado
(`data/processed_es_wide`) en vez de reusar el existente.

De paso quedó cuantificado que el **0,13 %** de los pares tiene ruido digitalmente nulo
(recortes que caen en tramos de silencio de MUSAN/ESC-50, con SNR efectivo infinito). La
proporción es idéntica en `processed_es`, así que es una propiedad de los corpus de ruido
y no algo que introduzca el rango ancho. A ese nivel es despreciable, pero conviene tenerlo
registrado.

---

## El estimando pasa a ser la trayectoria promediada, no el checkpoint seleccionado (15/09/2026)

V6 midió, sobre seis checkpoints consecutivos de dos ramas que sólo difieren en la compuerta,
un piso de ruido por checkpoint de **sd 0,0099 sobre `test_v2_es` y 0,0166 sobre `test_v1_en`**
en el contraste de PESQ-NB. Los efectos que este proyecto está persiguiendo en sus últimas
variantes son de ese mismo orden.

La consecuencia no es estadística sino de diseño: **ninguna comparación entre dos checkpoints
únicos puede resolver un efecto de este tamaño**, caiga donde caiga la selección. Da igual si la
selección fue por mínimo de `val_loss`, por PESQ sobre validación o por época fija: el número que
se reporta lleva incorporado un término de ruido tan grande como el efecto.

Desde V7 el estimando primario de una intervención arquitectónica es **el contraste promediado
sobre un tramo declarado de épocas** (15 a 20, las últimas seis de veinte), con el tramo escrito
en el preregistro antes de correr. Eso obliga a `save_every_n_epochs=1` en cualquier corrida cuyo
resultado se vaya a reportar.

Dos límites que hay que enunciar cada vez que se usa: los checkpoints de una trayectoria **no son
muestras independientes**, así que el p-valor sobre el promedio es descriptivo y no confirmatorio;
y promediar sobre épocas no sustituye a replicar sobre semillas, que es una fuente de varianza
distinta.

Evidencia acumulada de que la selección por mínimo de `val_loss` no está alineada con PESQ, en
orden: el placebo de V4b (V1 + 2 épocas a lr 2e-5 gana en las cuatro métricas con peor val MSE),
las dos réplicas de semilla de V5 (pico en la época 18, no en la de mínima `val_loss`), y el
barrido por época de V6. `scripts/select_by_val_pesq.py` reemplaza el criterio; el sd de la
diferencia apareada entre checkpoints es 0,059, así que con 300 pares de validación el error
estándar queda en 0,0034.

---

## V7 reentrena su propio control en vez de comparar contra V2 (15/09/2026)

La comparación barata para "¿sirve la compuerta desde cero?" sería correr sólo la rama con
compuerta y contrastarla contra V2, que ya está entrenado con la misma receta. **Está confundida.**
V2 se entrenó antes de que el trainer tuviera `cudnn_deterministic`, así que su trayectoria de
punto flotante es otra. Se verificó que `np.random` no interviene en el camino de datos del
entrenamiento —sólo aparece en `make_mixtures.py`, que genera el dataset offline— y que el
barajado usa el generador de torch, de modo que la única diferencia real es la selección de
kernels de cuDNN. Esa diferencia es del orden de cambiar la semilla: **sd 0,004 a 0,019** medido
en las réplicas de V5, o sea **del mismo tamaño que el efecto buscado**.

Costo de la decisión: 10,6 h de GPU extra. Beneficio: el contraste no arrastra el cambio de
régimen de `cudnn_deterministic`.

**Salvedad encontrada el 19/09, al validar el preregistro de las réplicas.** "Una sola variable"
no es literalmente cierto. Construir la cabeza de compuerta consume draws del generador global de
CPU, del que `RandomSampler` saca su semilla de época, así que **los dos brazos ven el mismo
dataset en distinto orden** — verificado: la inicialización del backbone es bit-idéntica
(`max|diff| = 0`) pero las semillas de sampler difieren en todas las épocas. El ruido que esta
decisión evitaba por un lado está metido por otro, del mismo orden (la sd de orden de datos de V5,
0,004-0,019) y un orden de magnitud por debajo del efecto. No se corrige, porque cambiarlo haría
que las réplicas corran otro protocolo; se declara, y las réplicas lo absorben.

Es la tercera vez que el proyecto paga este precio y la tercera vez que cambia la conclusión.
En V4b, contra el audio sin procesar el término perceptual parecía casi inocuo (−0,059) y contra
el placebo con idéntico protocolo costaba −0,598. En V6, contra V2 la compuerta parecía mejorar
+0,036 en inglés y contra el placebo costaba −0,011. **El control con protocolo idéntico no es
un lujo del diseño: es la diferencia entre la conclusión correcta y la opuesta.**

---

## La reanudación de una corrida cortada tiene que ser bit-exacta o no sirve (16/09/2026)

Dos cortes de energía el 16/09 —uno a las 06:50 durante el entrenamiento de V7control, otro a las
13:08 durante el barrido de evaluación— obligaron a decidir entre reanudar o reentrenar 10,5 h.

**Reanudar "de forma equivalente" no era una opción.** Las épocas perdidas eran 15 a 20, que son
exactamente las que cargan el estimando primario. Una reanudación que reponga los pesos pero no el
resto introduce una perturbación del mismo orden que el efecto buscado —el mismo argumento por el
que V7 reentrena su control en lugar de reusar V2—. O la reanudación es bit-exacta, o el
experimento se cae.

Las tres piezas que hay que reponer, y que `Trainer._resume` repone:

1. **Pesos y momentos de Adam.** Vienen en el checkpoint periódico.
2. **La fase del `StepLR`.** El lr ya viene restaurado dentro de `opt_state`, y `StepLR` es
   multiplicativo sobre el lr corriente —no lo recalcula desde `initial_lr`—, así que replicar los
   `step()` lo decaería una segunda vez. Sólo hay que reponer `last_epoch`. Este es el bug que
   agarró el test.
3. **El RNG global de CPU.** No se guarda en el checkpoint: se **replica**. `RandomSampler.__iter__`
   saca su semilla de época de ahí, y cada iterador de `DataLoader` saca de ahí su `base_seed` de
   workers, del que salen los recortes aleatorios de `NSDataset`. Reconstruir un iterador de train
   y uno de val por época ya hecha avanza el generador exactamente lo que esas épocas consumieron.
   El generador de CUDA no necesita réplica: el modelo no tiene ninguna operación estocástica en
   el dispositivo.

`tests/test_resume.py` verifica que los checkpoints de una corrida reanudada son **bit-idénticos**
a los de la misma corrida sin cortar, y corre cada entrenamiento **en un proceso aparte**: reanudar
después de un corte es necesariamente un proceso nuevo, y probarlo dentro del mismo intérprete
dejaría sin verificar justamente lo que podría no reproducirse entre procesos (elección de kernels
de cuDNN, alineación de memoria en GPU). Lo único que legítimamente difiere es `epoch_time_s`.

Evidencia externa al test, sobre la corrida real: la secuencia de lr de las épocas 15-20 del brazo
reanudado coincide exactamente con la del brazo que corrió sin cortes, la train loss engancha sin
escalón en el empalme (−0,0867 → −0,0886), y la val loss no muestra discontinuidad.

**Operativo:** el patrón de lanzamiento sigue siendo `setsid nohup ... < /dev/null &`, que sobrevive
al cierre de la terminal pero no al corte de energía. Un corte ahora cuesta, como máximo, la época
en vuelo.

---

## Corrección: el "67 % de retención entre canales" no se sostiene (09/09/2026)

La entrada del 07/09 cierra con un hallazgo lateral: que V5 retiene el 67 % de su ganancia al
cambiar de canal contra el 38 % de V3e, y que por lo tanto **"la dependencia del canal no es
una propiedad inevitable del fine-tuning: es una propiedad de la receta"**. Esa frase queda
retractada, y el número con ella.

**El error es de estimando, no de cálculo.** F6 define la ganancia como `variante − V1`, y ese
no es el mismo contraste para las dos recetas. V3e parte de V1, así que `V3e − V1` aísla el
fine-tuning al español. V5 parte de V2, así que `V5 − V1` mete adentro el salto de la loss
combinada — una mejora entrenada **solo en inglés**, que no tiene por qué comportarse como la
adaptación al idioma cuando cambia el canal. Se estaban comparando dos cosas distintas.

Con cada efecto medido desde su propio punto de partida:

| efecto | crowdsourced | audiolibro | retiene |
|---|---|---|---|
| `V2 − V1` (loss combinada, entrenada en inglés) | +0,121 | +0,160 | **132 %** |
| `V3e − V1` (fine-tuning al español desde V1) | +0,221 | +0,084 | **38 %** |
| `V5 − V2` (fine-tuning al español desde V2) | +0,235 | +0,079 | **34 %** |

**V5 retiene menos que V3e, no el doble.** Y lo que reemplaza a la frase retractada tiene mejor
control que ella: una mejora agnóstica al idioma transfiere al 132 % ante el mismo cambio de
canal, mientras que los dos fine-tunings al español transfieren al 34-38 %. O sea que la
dependencia del canal no distingue recetas de fine-tuning: distingue **qué tipo de mejora** es.

**Dónde sigue vivo el número viejo, y qué falta.** Esta corrección se declaró en el mensaje del
commit `cd0f443` y nunca bajó a los documentos, así que el 67 % siguió leyéndose como hallazgo
vigente en dos lugares hasta el 19/09: la entrada del 07/09 de este archivo y la tabla de F6 en
`docs/reanalisis_estadistico.md`. Los dos quedan marcados. **Pendiente y no hecho:**
`retention_by_recipe` en `analysis/reanalysis_stats.py` sigue calculando contra V1, así que
`results/reanalysis_stats.json` sigue publicando el estimando equivocado. Corregirlo toca código
con tests, y por la regla 2 de este proyecto el cambio y sus tests no los escribe el mismo agente.

**Lección de proceso, que es lo que hace que esto merezca una entrada propia:** una retractación
que vive únicamente en un mensaje de commit no existe. Los mensajes de commit no se releen; los
documentos sí. Cualquier número retirado se marca en el lugar donde se publicó, el mismo día.

---

## Qué miden las réplicas de semilla de V5, y qué no habilitan decir (08-09/09/2026)

Se corrieron V5 con semillas 43 y 44 además de la 42. **Las tres parten del mismo
`checkpoint/v2/best.pt` y el CRN no tiene dropout**, así que lo único que la semilla mueve es el
orden en que el fine-tuning ve los datos.

| | español, Common Voice | inglés, LibriSpeech | español, audiolibro |
|---|---|---|---|
| s42 / s43 / s44 | 2,686 / 2,658 / 2,661 | 2,774 / 2,770 / 2,766 | 2,840 / 2,803 / 2,816 |
| sd | 0,015 | 0,004 | 0,019 |

**Tres cosas que esto habilita, y una que no.**

Habilita: la ventaja de V5 sobre V3e en español es de siete a nueve desvíos de semilla, así que
el efecto no es orden de datos. Habilita también fijar la magnitud de ese ruido para diseños
futuros. Y obliga a una regla de reporte: **la semilla 42 es la más alta de las tres en los tres
sellados**, o sea que el titular del proyecto sale del máximo de tres corridas, y eso se dice
cada vez que se cita el número.

**No habilita decir "robusto a la semilla" a secas.** La varianza de inicialización no se midió:
las tres corridas arrancan del mismo lugar. Es una distinción que parece pedante hasta que se
necesita — en V7, donde los dos brazos entrenan **desde cero**, la semilla mueve también la
inicialización, y estos 0,004-0,019 son el denominador equivocado para juzgar aquel efecto. De
ahí sale la necesidad de las réplicas de V7 y no de un argumento genérico sobre n=1.

**`save_every_n_epochs=3` en las réplicas.** V5 sólo había guardado `best.pt`, lo que dejó sin
poder chequear si la época de mínima `val_loss` es también la mejor por PESQ. Cada 3 épocas da 8
checkpoints por corrida (1,6 GB) y cubre la meseta donde cae el mínimo, sin acopiar 5 GB.

**El barrido de época que eso permitió, y la decisión de no usarlo.** En las dos réplicas el pico
está en la época 18 y no en la de mínima `val_loss`: +0,036 (s43) y +0,022 (s44), p < 0,001
apareado. **No se seleccionó nada con esto**, y no se va a seleccionar: está medido sobre el
sellado, y elegir época mirándolo sería selección sobre el test set. V5 se reporta con su
`best.pt` de la época 21. El hallazgo cuenta como evidencia del problema de criterio de
selección, no como criterio (ver la entrada del 15/09).

---

## Un preregistro lo valida alguien que no lo escribió (18/09/2026)

En el preregistro de V6 la predicción P3 quedó escrita **con el signo invertido**: pedía que `g`
creciera con el SNR, contradiciendo la sección del mismo documento donde se explica que la
compuerta sólo puede aprender a replegarse, y replegarse es `g` bajando. El criterio literal
falló, el mecanismo real se cumplió con ρ ≈ −0,4, y el veredicto no se revisó — el preregistro
existe justamente para que no se arregle el enunciado después de ver el dato.

**La causa no fue distracción: fue que nadie más lo leyó.** Es la misma omisión que había dejado
a F6 sin tests, y es la regla 2 del proyecto aplicada a un documento en vez de a código: nunca la
misma persona —o el mismo agente— escribe algo y lo verifica.

**Regla desde acá:** ningún preregistro se hashea sin que lo lea un validador independiente, que
no haya participado en escribirlo. El validador busca, en este orden: contradicciones internas
—con atención especial a los signos—, criterios que no se puedan computar con el código y los
datos que van a existir, y grados de libertad que le dejen al autor acomodar el análisis después.
No busca mejorar la redacción.

Aplicado por primera vez al preregistro de la confirmación de V7 con semillas, el 18/09.

---

## El estimando confirmatorio de V7 excluye la semilla que disparó la confirmación (18/09/2026)

V7 pasó su screening con la semilla 42 y eso activa la confirmación con tres semillas que su
propio preregistro había declarado. Al escribir el preregistro de esa confirmación apareció una
decisión que no admite postergarse: **si el número confirmatorio promedia las tres semillas o
sólo las dos nuevas.**

**Decisión: el estimando confirmatorio es el promedio de las semillas 43 y 44. Las tres se
reportan siempre juntas.** La 42 es la que generó la hipótesis; incluirla en el número que la
confirma lo sesga hacia arriba por el mismo mecanismo por el que la semilla más alta de tres se
convierte en titular, que es el problema que V5 documentó en carne propia. Un dato no puede
generar y confirmar la misma afirmación.

**Consecuencia aceptada:** el confirmatorio tiene n=2. No hay potencia para testear variabilidad
entre semillas, así que se declara como **verificación de replicación y no como test de
hipótesis**, y no se fabrica un p-valor que no corresponde.

**Consecuencia operativa, que es la que fija el presupuesto:** con este estimando hay que correr
semillas nuevas completas, los dos brazos de cada una. Recortar a una sola dejaría el
confirmatorio otra vez en n=1 y haría inútil el gasto.

**Corregido el 19/09 al cerrar el preregistro: van TRES semillas nuevas (43, 44 y 45), no dos.**
Esta entrada se escribió el 18/09 con un plan de dos semillas y ~45 h, y quedó desactualizada.
El documento hasheado declara seis corridas y ~68 h, que además es la lectura estricta de las
"~63 h de la confirmación con tres semillas" que presupuestaba el preregistro de V7. Con n=3 el
piso por semilla de C2 deja de ser un parche sobre un diseño corto y pasa a ser una guarda sobre
uno que ya tiene margen. Vale el documento hasheado, no esta entrada: el hash está en
`docs/preregistro_v7_semillas.sha256`.

El preregistro completo vive fuera del repo por la regla anti-contaminación; acá queda la
decisión y su razón, que es lo que tiene que sobrevivir aunque el documento no se lea. El hash va
a `docs/preregistro_v7_semillas.sha256` antes de lanzar.

Auditoría de números heredados al consolidar EXPERIMENTS.md (20/09/2026)
 
Escribir las secciones faltantes de la bitácora obligó a releer cada número contra su JSON.
Cuatro no daban, todos por transcripción y ninguno por medición:

1. decisions.md:543 — control_mse mejora en tres métricas, no cuatro (PESQ-WB: −0,019).
2. decisions.md:780 — OP-1 lo cruzaba V3e, no lo cruza V5 por primera vez; y es PESQ-NB, no WB.
3. V3b en español es +0,152, no +0,153 (2,4825 − 2,3302). Vivía en informe/, CLAUDE.md y el generador del dashboard.
4. El mensaje del commit 7f7884d atribuye a V3e el val_loss 0,0864, que es el de V3b (época 5). El de V3e es 0,0843 (época 14). Un mensaje de commit ya pusheado no se reescribe: queda anotado acá.

Los cuatro son errores de transcripción entre documentos, no de medición. Ninguna conclusión cambia. Es el modo de falla que ya produjo la retractación del "67 % de retención": un número sobrevive en prosa después de que su fundamento se movió. Mitigación adoptada: al escribir cualquier sección nueva, cada número se recomputa contra su JSON en vez de copiarse.
