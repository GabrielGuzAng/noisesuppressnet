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
