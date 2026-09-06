# CLAUDE.md — Contexto operacional para Claude Code

Este archivo es contexto para trabajar en NoiseSuppressNet con Claude Code.
No es documentación de usuario final. Es briefing operacional para asistencia técnica.

---

## Qué es este proyecto

**NoiseSuppressNet** es el Proyecto Final de Grado de Ingeniería Electrónica en UTN FRBA.

**Autor**: Gabriel Guzmán Anglese
**Materia**: Proyecto Final de Grado
**Universidad**: UTN FRBA (Facultad Regional Buenos Aires)
**Fecha límite defensa**: 29 de diciembre de 2026
**Repositorio**: github.com/GabrielGuzAng/nosiesuppressnet

### Objetivo técnico

Desarrollar un sistema causal de supresión de ruido en habla (speech enhancement) basado en una arquitectura CRN (Convolutional Recurrent Network), con las siguientes propiedades:

- Causal a nivel de frame (verificado bit-exacto). **La STFT usa `center=True`, que agrega
  10,0 ms de lookahead a nivel de señal — medido, no estimado.** La latencia algorítmica es
  10 ms, no cero. Aceptable para tiempo real; lo que no es correcto es afirmar "sin lookahead".
- Sample rate 16 kHz mono
- RTF (Real-Time Factor) < 1.0 en CPU i5-4460 single thread
- Fine-tuning para español rioplatense
- Loss combinada MSE + SI-SDR + PESQ diferenciable (via Squim)

### Aporte diferenciador del proyecto

Tres ejes de contribución:

1. **Ablation study sistemático** (V0-V5) del efecto de cada mejora arquitectural
2. **Fine-tuning para español** desde baseline en inglés (diferenciador vs literatura mayoritariamente EN)
3. **KPI propio**: correlación Pearson entre PESQ real y Squim (proxy diferenciable) como métrica de confiabilidad del loss perceptual

---

## Contexto del autor y del entorno laboral

Gabriel trabaja como Tech Lead en **CyT Comunicaciones**, desarrollando **VoxHub** (sistema de voice AI para contact centers que integra OpenAI Realtime API con Orion PBX). NoiseSuppressNet nace del interés en tener un modelo propio de speech enhancement (sin depender de APIs externas) para eventual integración con VoxHub como componente de preprocesamiento.

**Hardware disponible**:
- MSI H81M-E33 + Intel i5-4460 (4 cores, 3.2 GHz, sin AVX2)
- 12 GB DDR3
- MSI RTX 4060 Gaming X 8 GB VRAM
- Ubuntu 24.04 LTS
- CUDA 13.2, PyTorch 2.5.1+cu121

**Ubicación**: Ramos Mejía, Buenos Aires, Argentina
**Idioma preferido**: español informal argentino en comunicación, inglés en docstrings

---

## Restricciones duras (NO VIOLAR)

### 1. Reproducibilidad blindada

- Seed fijo (`42`) en TODAS las operaciones aleatorias
- CuDNN determinism activado en trainer
- Datasets versionados con hash SHA-256
- Test set sellado (v1 EN y v2 ES) NO se regenera bajo ninguna circunstancia

### 2. Fidelidad al paper base

- Arquitectura CRN sigue Tan & Wang 2018 (Interspeech)
- Modificaciones documentadas explícitamente con justificación
- STFT: n_fft=320, hop=160, Hamming window (bit-exact al paper)
- Causal estricto: verificable con test `test_causality.py`

### 3. Ablation study limpio

- Entre variantes V1→V2→V3→V4→V5, **una sola variable cambia por vez**
- Misma seed (`42`) en todas las variantes
- Mismos hiperparámetros de optimización a menos que la variante sea sobre fine-tuning
- Deviaciones documentadas en `docs/decisions.md`

### 4. No entrenar con outputs propios del modelo

- Retraining futuro NUNCA usa audio "denoiseado" como training
- Solo audio raw con consentimiento explícito
- Evita feedback loops perversos documentados en literatura

### 5. Separación con proyectos hermanos

- `tp-dataset-analysis` es un TP de otra materia, proyecto SEPARADO
- VoxHub es proyecto laboral en CyT, proyecto SEPARADO
- No mezclar código o dependencias

---

## Estructura del repositorio
nosiesuppressnet/
├── CLAUDE.md # Este archivo (contexto operacional)
├── ROLES.md # Prompts especializados por rol
├── AGENTS_GUIDE.md # Guía de agentes y workflows
├── README.md # Documentación de usuario
├── CHANGELOG.md # Historial de versiones
├── LICENSE # MIT
├── .gitignore
├── requirements.txt
│
├── models/
│ └── crn.py # Arquitectura CRN 17.58M params
│
├── stft.py # STFT helper (torch.stft con Hamming)
│
├── datasets/
│ ├── init.py
│ ├── dataset.py # NSDataset (Dataset de PyTorch)
│ └── make_mixtures.py # Generación de pares (noisy, clean)
│
├── training/
│ ├── init.py
│ ├── trainer.py # Loop de entrenamiento
│ ├── losses.py # mse_magnitude, si_sdr_loss, mse_plus_sisdr
│ └── config.py # CONFIG_V1, CONFIG_V2, CONFIG_V3, ...
│
├── inference/
│ ├── init.py
│ └── infer.py # Inferencia sobre archivo o batch
│
├── evaluation/
│ ├── init.py
│ ├── metrics.py # PESQ-NB, PESQ-WB, STOI, SI-SDR
│ ├── evaluate_variant.py # Evalúa una variante sobre test sealed
│ └── monitor_correlation.py # KPI: correlación PESQ real vs Squim
│
├── baselines/
│ └── butterworth.py # Baseline no-DL (filtro pasabajo)
│
├── benchmarks/
│ └── measure_rtf.py # RTF sobre CPU objetivo
│
├── analysis/
│ ├── training_cost_report.py # Reporte tiempo/energía por variante
│ └── plot_psd_comparison.py # PSD comparison (V0 análisis)
│
├── scripts/
│ ├── seal_test_set_en.py # Sellado test set v1 EN (una sola vez)
│ ├── seal_test_set_es.py # Sellado test set v2 ES (una sola vez)
│ ├── verify_test_set.py # Verificación de integridad por hash
│ ├── generate_dashboard.py # Genera docs/dashboard.html desde results/*.json
│ └── generate_listening_samples.py # Audios noisy/clean/enhanced para escuchar a mano
│
├── tests/
│ ├── test_causality.py # Verifica causalidad bit-exact
│ ├── test_stft.py # Verifica round-trip STFT
│ ├── test_pipeline.py # Test end-to-end del pipeline
│ ├── test_pipeline_real.py # Test con archivos reales
│ ├── test_dataloader.py # Test del DataLoader
│ └── test_squim_differentiable.py # Verifica que Squim es diferenciable
│
├── docs/
│ ├── EXPERIMENTS.md # Bitácora de experimentos V0-V5
│ ├── decisions.md # Log de decisiones metodológicas
│ ├── v1_training_cost.md # Reporte costo V1
│ ├── v2_training_cost.md # Reporte costo V2
│ ├── dashboard.html # Panel visual de resultados (generado, no editar a mano)
│ └── ...
│
├── seal_test_metadata/ # Hashes y metadata de test sets
│ ├── test_v1_hash.txt
│ ├── test_v1_metadata.json
│ ├── test_v2_hash.txt # (cuando exista)
│ └── test_v2_metadata.json
│
├── checkpoints/ # Modelos entrenados (gitignored)
│ ├── v0/
│ ├── v1/
│ ├── v2/
│ ├── v3/ # cuando exista
│ ├── v4/ # cuando exista
│ └── v5/ # cuando exista
│
├── data/ # Datasets (gitignored)
│ ├── raw/
│ │ ├── clean_speech/
│ │ │ ├── en/librispeech/ # LibriSpeech train-clean-100
│ │ │ └── es/common_voice/ # Common Voice ES v26 (pendiente descarga)
│ │ └── noise/
│ │ ├── musan/
│ │ └── esc50/
│ ├── processed/ # Mixtures EN (50k train + 2k val)
│ ├── processed_es/ # Mixtures ES (cuando exista)
│ └── test_sealed/
│ ├── v1_en/ # 250 pares sellados en inglés
│ └── v2_es/ # 250 pares sellados en español (cuando exista)
│
├── results/ # Reportes de evaluación
│ ├── summary_metrics.csv # Tabla resumen todas las variantes
│ ├── v1_test_sealed.json
│ ├── v2_test_sealed.json
│ ├── v1_training_cost.json
│ └── v2_training_cost.json
│
└── logs/ # Logs de entrenamiento (gitignored)
├── v1_training.log
├── v2_training.log
└── mixture_v1.log


---

## Estado actual del proyecto (actualizar en cada sesión)

**Última actualización**: 6 de septiembre de 2026

### Completado

**Sprint inicial (junio 2026)**:
- [x] Preproyecto defendido exitosamente el 30 de junio de 2026
- [x] V0 entrenado (baseline con 200 pares, 10 épocas)
- [x] V0 evaluado: PESQ 1.42, STOI 0.79, SI-SDR 3.11 dB
- [x] Output collapse identificado y documentado en análisis PSD
- [x] Baseline Butterworth LP para comparación
- [x] Tests bit-exact de causalidad y STFT
- [x] RTF medido: 0.34 p95 en CPU i5-4460

**Sprint V1-V2 (julio-agosto 2026)**:
- [x] Test set v1 EN sellado (250 pares, 5 buckets SNR balanceados, hash SHA-256)
- [x] Dataset EN escalado (50k train + 2k val)
- [x] V1 entrenado (30 épocas, best época 19, val_loss 0.0724)
- [x] V1 evaluado: PESQ-NB 2.65 (+0.50 vs noisy), PESQ-WB 2.02, STOI 0.90, SI-SDR 13.80 dB
- [x] V2 entrenado (20 épocas, best época 19, loss combinada MSE 0.7 + SI-SDR 0.3)
- [x] V2 evaluado: PESQ-NB 2.85 (+0.20 vs V1), PESQ-WB 2.21, STOI 0.91, SI-SDR 14.44 dB
- [x] Squim verificado diferenciable (gradient norm 3.11e-01 sobre audio ruido)
- [x] Reportes de costo V1 y V2 generados

**Sprint reanálisis estadístico (5-6 de septiembre de 2026)**:
- [x] `analysis/reanalysis_stats.py` — familias F1 a F5 con estimandos declarados, Holm por
      familia, IC bootstrap BCa. Validado por 358 tests en `tests/test_reanalysis_stats.py`.
- [x] Salidas: `results/reanalysis_stats.json` y `docs/reanalisis_estadistico.md`.
- [x] `analysis/hueco_bucket_checks.py` — los dos recomputos que no salen del JSON.
- [x] Secciones 04/05/06/08 de `docs/donde_esta_el_hueco.html` reescritas sobre esos números.
- [x] Proceso de delegación documentado en `~/nosiesuppressnet-oracle/PROMPTS_REANALISIS.md` (fuera del repo) (roles, contrato,
      prohibiciones anti-sesgo, bitácora de 7 compuertas).
- [x] Preregistro de `test_v3_mls_es` escrito **antes** de que exista el dato.

**Infraestructura**:
- [x] Repo con tags v0.1.0, v1.0.0, v2.0.0
- [x] Trainer con soporte multi-variante via config.py
- [x] Sistema de evaluación reutilizable (`evaluate_variant.py`)
- [x] Reporte de costo automatizado (`training_cost_report.py`)
- [x] Panel de resultados local (`scripts/generate_dashboard.py` → `docs/dashboard.html`) — 23-24/08/2026, commiteado en `7f7884d`. Un solo archivo HTML autocontenido (sin servidor, sin dependencias externas), lee todos los `results/*.json` automáticamente. Tabs Inglés/Español con gráficos de barras (con escala numérica de referencia, no solo el valor puntual de cada barra: PESQ cada 0.5, STOI cada 0.2, SI-SDR paso automático según el rango real) comparando PESQ-NB/PESQ-WB/STOI/SI-SDR entre Noisy y cada variante entrenada, línea de ΔPESQ-NB por bucket de SNR (con eje Y etiquetado), tabla completa con deltas, línea de tiempo del proyecto (V0→V3e) y sección wiki explicando cada métrica con referencia bibliográfica. Paleta categórica validada para daltonismo (skill `dataviz`), tema claro/oscuro automático. Pensado para mostrar avances en reuniones/defensa sin depender de internet. Regenerar con `python -m scripts.generate_dashboard` después de evaluar una variante nueva.
- [x] `scripts/generate_listening_samples.py` extendido con `--lang es/en` — 24/08/2026. Genera audios noisy/clean/enhanced de un puñado de pares (uno por bucket de SNR) para escuchar a mano cualquier combinación de variantes sobre cualquiera de los dos test sets sellados, sin correr la evaluación completa de 250 pares.

**Sprint dataset ES (agosto 2026)**:
- [x] Descarga Common Voice ES v26 (1.680.810 clips, 2.278,7 h)
- [x] EDA y generación de manifests limpios por split (`scripts/analyze_cv26_es.py`): train 320.708 / dev 13.139 / test 12.620 clips, 0 leakage de hablante/frase entre splits
- [x] `data/processed_es` symlinkeado a `/mnt/Datos` (misma convención que `data/processed`/`data/raw`)
- [x] `datasets/make_mixtures.py` adaptado para MP3 y manifests de Common Voice (balanceo de género, splits train/dev disjuntos)
- [x] `scripts/seal_test_set_es.py` escrito (250 pares, 5 buckets SNR × 50, mismo diseño que v1_en)
- [x] Dataset ES generado por completo (50.000 train + 2.000 val) — 20/08/2026, corrido en su propia sesión (`setsid`) tras el crash del intento anterior. Log termina en "Listo.", conteos verificados 50000/50000 y 2000/2000.
- [x] Bug de reproducibilidad corregido en `datasets/make_mixtures.py`: `random.seed(42)` fijado al inicio y `collect_files` ahora devuelve `sorted()` (antes dependía del orden no determinista de `rglob()`)
- [x] `CONFIG_V3` agregado a `training/config.py`, soporte de `init_checkpoint` agregado a `training/trainer.py`, V3 registrado en el `__main__` del trainer

**Sprint V3/V3b/V3e — fine-tuning EN→ES (agosto 2026)**:
- [x] Test set v2 ES sellado (250 pares, hash SHA-256) — 20/08/2026, tag `test_set_v2`
- [x] V3 entrenado y evaluado — full fine-tuning agresivo desde V1 (mismo lr 2e-4, 30 épocas). PESQ-NB: +0.230 vs V1 en español, -0.079 en inglés (forgetting moderado). Tag `v3.0.0`.
- [x] Bug de reproducibilidad cerrado: `torch.backends.cudnn.deterministic` + `np.random.seed()` agregados a `training/trainer.py` (antes solo había `torch.manual_seed`, pese a que `CLAUDE.md` ya afirmaba lo contrario). Costo medido: ~2x tiempo/época en este modelo (94% LSTM) — flag `cudnn_deterministic` en config, default `True`, desactivable para corridas exploratorias descartables (sweeps).
- [x] Sweep de lr para V3b (5 candidatos, 5 épocas c/u, `scripts/lr_sweep_v3b.py` + `lr_sweep_v3b_evaluate.py`) — ganador lr=5e-5, score +0.121.
- [x] V3b entrenado y evaluado — fine-tuning conservador (lr=5e-5, 10 épocas). PESQ-NB: +0.153 en español, -0.029 en inglés. Score +0.124 — **peor balance que V3**, hipótesis de partida no se sostuvo (ver hallazgos técnicos).
- [x] V3e entrenado y evaluado — explota un candidato del sweep (lr=1e-4) que no había convergido a 5 épocas, extendido a 25 épocas con decay tardío. PESQ-NB: +0.221 en español, **-0.031** en inglés (el -0.032 que se reportó era un redondeo
  incorrecto). **El "score compuesto" que sumaba ambos ejes quedó retirado por el reanálisis
  de septiembre: mezclaba dos estimandos de naturaleza distinta. Ver F2/F3.** Tag `v3e.0.0` — checkpoint recomendado para V5.
- [x] Gap cerrado: V2 evaluado sobre test_v2_es (nunca se había hecho) — mismo patrón que V1 en SNR alto (degrada la señal).
- [x] Commits: `49235c4` (V3b, sin tag), `d1903e2` (config V3e), `7f7884d` (V3e + resultados + dashboard, tag `v3e.0.0`)

### En curso

- Nada corriendo. Reanálisis estadístico cerrado y validado (05-06/09). Siguiente: control
  idioma/canal con `test_v3_mls_es` (MLS español), preregistrado antes de bajar el dato.
- **GCRN descartado** como línea de trabajo: no alimenta ninguna de las cuatro afirmaciones
  del aporte y compite por las mismas horas. El plan escrito queda fuera del repo.

### Pendiente

**Corto plazo (agosto-septiembre 2026)**:
- [ ] Escribir sección V3 consolidada en `docs/EXPERIMENTS.md` con la narrativa completa de las tres iteraciones (V3→V3b→V3e), tabla comparativa y discusión honesta de por qué V3b no superó a V3 y qué hizo funcionar a V3e. Documentado en el mensaje del commit `7f7884d`, pero no en la bitácora formal todavía.
- [x] V4 (MSE + Squim, EN): Fase 3 Paso 1 corrido y evaluado (31/08/2026) — **resultado negativo caracterizado**. Squim frozen se gamea desde época 1 en las 4 corridas; control MSE puro gana en las 4 métricas reales. Detalle en `docs/decisions.md`, "Cierre de V4" (31/08).
- [x] V4b (protocolo estabilizado de Xu, sin reentrenar el proxy) corrido y evaluado (01/09/2026, 8.02h) — **cierra definitivamente la línea de proxy**. Diseño 2×2 + placebo. Resultado: el arranque desde V1 convergido + lr 2e-5 recuperan ~33% del daño, pero **el término Squim sigue costando −0.598 PESQ-NB** contra el placebo con el mismo protocolo. Dos tercios del gaming sobreviven. Detalle completo con tablas en `docs/decisions.md`, entrada "V4b" (01/09).
- [ ] **Próximo experimento: control de idioma vs canal** — sellar `test_v3_mls_es` con voz de
  Multilingual LibriSpeech español (mismo paradigma de grabación que LibriSpeech) reusando
  verbatim las condiciones de ruido de `test_v1_en`. Sin reentrenar nada: evaluar V1, V2 y V3e.
  Responde a Wang et al. 2022 (Interspeech, *Disentangling the Impacts of Language and Channel
  Variability*), que es la objeción más fuerte contra el aporte 1. Predicciones preregistradas
  antes de bajar el dato; hash comprometido en `docs/preregistro_mls_es.sha256`.
- [ ] **GCRN descartado.** Se evaluó (Tan & Wang 2020, paper y repo oficial leídos) y no alimenta
  ninguna de las cuatro afirmaciones del aporte: es una mejora de arquitectura, no del eje
  cross-lingual, y compite por las mismas horas contra el control de canal y el downstream ASR.
  Único encuadre en que valdría la pena: como chequeo de validez externa del hallazgo por SNR
  sobre otra arquitectura. V5 (Squim sobre ES) también descartado — dependía de que Squim
  aportara, y no aportó.
- [ ] Revisar si conviene re-reportar V1 con el checkpoint del placebo de V4b: V1 + 2 épocas a lr 2e-5 da PESQ-NB 2.716 vs 2.650, mejor en las 4 métricas (ver hallazgo lateral en `decisions.md` 01/09). Implica que el criterio de selección por mínimo de val MSE no está alineado con PESQ. Decisión pendiente: los números de V1 ya están taggeados y reportados, cambiarlos ahora afectaría todas las comparaciones V1→V2/V3/V3e.

**Mediano plazo (octubre-noviembre 2026)**:
- [ ] Ablation opcional V1b (100k pares, si el profesor lo requiere)
- [ ] Análisis comparativo completo V0-V5 con visualizaciones
- [ ] Auditoría auditiva subjetiva sobre muestra representativa
- [ ] Benchmark contra RNNoise y DeepFilterNet2 (baselines externos)

**Cierre (diciembre 2026)**:
- [ ] Redacción del informe final (formato UTN)
- [ ] Preparación presentación de defensa
- [ ] Defensa oral 29/12/2026

### Bloqueadores actuales

- Ninguno técnico crítico. Espacio en disco para Common Voice ES resuelto: dataset y mixturas ES viven en `/mnt/Datos` (625 GB libres), no en el disco raíz. Línea V3/V3b/V3e cerrada; el siguiente trabajo real (V4) todavía no tiene config escrita.

**Nota operativa para futuros jobs largos en background:** el 18/08/2026 un `nohup python ... &` lanzado desde una gnome-terminal murió al cerrar la ventana (systemd-logind terminó el cgroup de la sesión pese a `nohup`). Desde el 19/08 los jobs largos se lanzan con `setsid nohup comando > log 2>&1 < /dev/null &`, que reparenta el proceso a su propia sesión (`SID == PGID == PID`, sin `tty` asociada) y sobrevive al cierre de la terminal. Usado consistentemente para V3, el sweep de V3b, V3b final y V3e — seguir usándolo para V4/V5.

**Gotcha del PID capturado con `$!` en este patrón (30/08/2026):** al lanzar `setsid nohup comando &` desde una shell interactiva, `setsid` ya es líder de su propio grupo de procesos (por job control de bash) y el syscall `setsid()` falla con `EPERM` sobre sí mismo. El binario `setsid` de util-linux resuelve esto haciendo un fork interno: el proceso original (el que devuelve `$!`) termina enseguida, y el **hijo** (con un PID distinto, nunca capturado) es el que de verdad llama a `setsid()` y ejecuta el comando. Consecuencia: `echo $!` guarda el PID equivocado — parece "el proceso murió" (`ps -p $!` no encuentra nada, bash reporta `Done` en el próximo prompt) cuando en realidad el entrenamiento real sigue corriendo bajo otro PID. Pasó el 30/08 con el sweep de V4 (Paso 1) y llevó a perseguir un OOM que nunca existió. **Verificación correcta: `pgrep -af <nombre_del_módulo>` + `nvidia-smi` (buscar el proceso `python` en la tabla de Processes), nunca confiar en el PID guardado en `logs/*.pid` para este patrón de lanzamiento.**

---

## Hallazgos técnicos clave (para no perder contexto entre sesiones)

### Del análisis V0 (sprint inicial)

- **Output collapse severo**: modelo aprendió a predecir "silencio uniforme en HF"
- **Causa raíz identificada**: dataset insuficiente (200 pares) + underfitting extremo
- **PSD promedio**: caída de 20 dB entre 0-8 kHz (natural en audio de 16 kHz)

### Del salto V0→V1

- **Ganancia**: +1.23 puntos PESQ (extraordinariamente alta)
- **Causa dual**: (a) mayor volumen de datos, (b) ruptura de output collapse
- **NO es extrapolable linealmente** — literatura muestra saturación logarítmica
- Papers de referencia: Reddy 2020, Kolbæk 2020, Schröter 2022

### Del salto V1→V2

- **Ganancia**: +0.20 puntos PESQ, +0.64 dB SI-SDR
- **Interpretación**: loss combinada MSE + SI-SDR aporta información complementaria
- **Ganancia mayor en SNR alto** ([10-20] dB): SI-SDR ayuda al "procesamiento quirúrgico"
- Confirmación empírica de Braun & Tashev 2021 (sección 4.3)

### De V4/V4b — el riesgo de gaming se materializó (31/08-01/09/2026)

Lo que era "riesgo identificado" antes de correr nada quedó **confirmado empíricamente con datos
propios**. Resumen; detalle completo con tablas en `docs/decisions.md` (entradas "Cierre de V4" y "V4b").

- **Squim frozen se gamea desde la época 1**, en las 4 corridas de V4 y en las 2 de V4b con término
  perceptual. Firma: `pesq_hat` estimado por Squim trepa hacia su techo estructural (4.64) mientras el
  PESQ real medido cae por debajo de Noisy. Es el patrón exacto que Xu 2022 reporta citando a Fu et al.
  [31]: *"estimated PESQ scores increase while true PESQ scores decrease"*.
- **El protocolo estabilizado de Xu (menos la alternancia) recupera solo ~1/3 del daño.** V4b probó
  arranque desde V1 convergido + lr 2e-5. Daño atribuible al término Squim: −0.887 PESQ-NB en el
  régimen inestable, −0.598 en el estabilizado. **Dos tercios del gaming sobreviven** → lo que hace
  funcionar el esquema de Xu es específicamente reentrenar el proxy, no el arranque ni el lr.
- **El placebo (α=1) es imprescindible para no leer mal el resultado.** Contra Noisy, V4b-main da
  −0.059 y parece "casi arreglado"; contra el placebo con idéntico protocolo (+0.539), el término
  cuesta −0.598. Sin ese control la conclusión hubiera sido la opuesta.
- **La brecha `pesq_hat − PESQ-WB real` es el diagnóstico barato de gaming.** MAE nominal de Squim para
  WB-PESQ: 0.142 (Kumar 2023, Tabla 2). Brechas observadas: +2.36 y +2.77, ~16-20× ese error. Comparar
  siempre contra PESQ-**WB**, no NB: la cabeza de Squim estima WB-PESQ, acotada en [1, 4.64] por sigmoide.
- **Sesgo de selección de `best.pt` con losses que incluyen el proxy**: `Trainer` elige por `val_loss`
  mínima, que para `mse_plus_squim` incluye `-pesq_hat` → más gameado = "mejor" checkpoint. Con esas
  losses hay que usar `save_every_n_epochs=1` y evaluar época por época.
- **La línea de proxy perceptual está cerrada.** No se corren más variantes; el espacio de decisión
  quedó cubierto (inestable + estabilizado). Un proxy propio queda como Fase 10 opcional de
  GCRN, solo si sobra tiempo.
- **Hallazgo lateral**: V1 no estaba del todo convergido — 2 épocas más a lr 2e-5 dan PESQ-NB 2.716 vs
  2.650, mejor en las 4 métricas, pese a peor val MSE. El criterio de selección por val MSE no está
  alineado con PESQ.

### Del reanálisis estadístico (05-06/09/2026)

- **El aporte 1 se enuncia como interacción, no como cruce de signo.** La ganancia de V1 no
  depende del SNR en inglés (ρ=−0,065, n.s.) y decae en español (ρ=−0,262, p_Holm=8,3e−05);
  la interacción sobrevive Holm (ρ=+0,173, p_Holm=0,012). El negativo puntual del bucket
  [15,20] **no** es sostenible por sí solo: n=50 por bucket, test de signo 24/50 (p=0,89).
- **Adaptación difusa vs olvido selectivo.** En español mejora el 85-90% de los archivos
  (test de signo p≈1e−30 a 1e−41). En inglés la mediana no se mueve y el signo es una moneda,
  pero la media cae por una cola izquierda pesada (skew −2,2 a −3,5). El estimando correcto
  del olvido es **la fracción de archivos que se rompen**, no la media: 17,6% (V3) / 11,6%
  (V3e) / 6,8% (V3b), monótono con la agresividad del lr.
- **La cola no es ruido de medición.** Contra controles del mismo idioma (V2, y el placebo de
  V4b) la cola es 0,0-0,4%. Los archivos rotos se repiten entre variantes 4-7× por encima del
  azar (p<1e−6). Solo 54 archivos de 250 se rompen en alguna variante.
- **Hay un confusor real y medido**: los archivos que se rompen tienen PESQ basal más alto
  bajo V1 (+0,276, IC95 [+0,077, +0,480]). No lo descarta un control marginal —el primer
  intento con n=10 no tenía potencia y dio un falso negativo— sino la estratificación: dentro
  del cuartil basal más alto los controles siguen en 0,0% y los tratamientos en 9,5-28,6%.
  El perfil por cuartil **no es monótono**, así que el efecto techo tampoco explica la forma.
- **El control que carga el peso es V2, no el placebo** (el placebo es casi idéntico a V1 y
  por eso no es un sorteo independiente).
- **Este reanálisis es exploratorio**: las hipótesis salieron de estos mismos datos. La
  confirmación fuera de muestra es `test_v3_mls_es`, con predicciones ya preregistradas.

### Del dataset Common Voice ES (previo a V3)

- **~23% de los clips CV ES miden menos de 4s** (target de duración de
  las mixturas): media 5,16s, mediana 5,06s, mínimo 1,04s sobre 320.708
  clips de train. En esos casos `pad_or_crop` loopea el clip (`repeat()`)
  en vez de recortar un segmento aleatorio, algo que en LibriSpeech
  (V1/V2) casi no ocurría por tener clips más largos. Posible fuente de
  artefactos de repetición audible en ~1 de cada 4 pares ES — no es un
  bug, pero es una diferencia real entre datasets que conviene tener
  presente al analizar métricas de V3. Detalle completo en
  `docs/decisions.md` (18/08/2026).
- `Trainer` (en `training/trainer.py`) no tiene mecanismo de carga de
  checkpoint pre-entrenado — siempre instancia `CRN()` con pesos
  aleatorios. Hay que agregarlo antes de poder entrenar V3, sea cual sea
  el diseño elegido (fine-tuning desde V1 o desde V2).

### De V3/V3b/V3e (fine-tuning EN→ES)

- **Full fine-tuning agresivo (V3) funciona pero con forgetting real**: +0,230
  PESQ-NB en español, -0,079 en inglés. Consistente con la literatura de
  transfer learning (Yosinski 2014, McCloskey & Cohen 1989 sobre
  catastrophic interference) para fine-tuning a lr alto sin capas
  congeladas.
- **La hipótesis "lr conservador mejora el balance" (V3b) no se sostuvo**:
  redujo el forgetting (-0,029) pero también la ganancia en español
  (+0,153) — score compuesto peor que V3 (+0,124 vs +0,151). Causa
  identificada: V3b entrenó 10 épocas contra las 30 de V3 — épocas
  totales y agresividad de lr quedaron confundidas (confounded) en el
  diseño, no se pudo aislar cuál pesaba más.
- **V3e resolvió el balance**: mismo lr "prometedor" del sweep (1e-4,
  el que tenía mayor ganancia en ES sin converger a 5 épocas) pero con
  25 épocas y decay tardío (época 12, no época 4 como V3b). Resultado:
  +0,221 en español (casi igual a V3) con -0,031 en inglés (casi igual
  a V3b) — score +0,189, el mejor de las tres. **Checkpoint recomendado
  para V5 de acá en más: `checkpoints/v3e/best.pt`, no V3.**
- **V1 y V2 (entrenados solo en inglés) degradan activamente audio limpio
  en español**: en el bucket de SNR alto [15,20] dB de `test_v2_es`, V1
  tiene PESQ Δ -0,167 y SI-SDR Δ -3,49 dB; V2 tiene PESQ Δ -0,087 y
  SI-SDR Δ -3,58 dB. Ambos negativos — el modelo introduce artefactos
  sobre voz limpia en un idioma no visto durante entrenamiento. Es
  transversal a la loss (pasa con MSE puro y con MSE+SI-SDR), no un
  artefacto puntual de V1. Dato fuerte a favor de la hipótesis central
  del proyecto (el fine-tuning por idioma no es opcional). V3/V3b/V3e
  corrigen esto casi por completo en ese mismo bucket.
- **CuDNN determinism tiene costo real de performance en este modelo**:
  ~2x tiempo/época (94% LSTM, kernels deterministas de cuDNN para RNN
  notablemente más lentos). V1/V2/V3 se entrenaron sin este flag (solo
  `torch.manual_seed`, no bit-exact) — no se corrige retroactivamente,
  pero es una limitación real de esos resultados. Desde V3b en
  adelante, flag `cudnn_deterministic` en config (default `True`,
  desactivado explícitamente solo en corridas exploratorias
  descartables como sweeps).

### Sobre el sample rate y filtrado HF

- Sugerencia de profesor durante defensa: aplicar LP donde PSD cae 20 dB
- Análisis reveló que DeepFilterNet2 NO filtra input (procesa HF con arquitectura liviana)
- Braun & Tashev 2021 documenta que LP degrada PESQ-WB 0.15-0.30 puntos
- **Decisión**: mantener banda completa, documentar como trabajo futuro

---

## Convenciones del código

### Python

- **Versión**: Python 3.10+
- **Style**: PEP 8, docstrings en formato Google
- **Type hints**: en funciones públicas
- **Pathlib** para paths (nunca `os.path`)
- **Logging** con `logging` module (no `print` para debug)

### Naming

- Archivos: `snake_case.py`
- Clases: `PascalCase`
- Funciones y variables: `snake_case`
- Constantes: `UPPER_SNAKE_CASE`
- Configs: `CONFIG_V<n>` (ej. `CONFIG_V3`)
- Checkpoints: `checkpoints/v<n>/best.pt`

### Estructura de archivos

- Un módulo Python por archivo, propósito único
- Docstring de módulo al inicio explicando qué hace
- Imports agrupados: stdlib, terceros, propios

### Documentación

- **Español** en `docs/` (audiencia académica argentina)
- **Inglés** en docstrings (convención internacional)
- Referencias bibliográficas explícitas cuando se aplica una técnica de paper

---

## Comandos frecuentes

### Setup

```bash
cd ~/Desktop/nosiesuppressnet
source .venv/bin/activate
```

### Entrenamiento de una variante

```bash
# Ver configuraciones disponibles
python -c "from training.config import CONFIG_V1, CONFIG_V2; print(list(globals().keys()))"

# Entrenar (nombre de config)
python -m training.trainer --config V1

# En background
nohup python -m training.trainer --config V3 > logs/v3_training.log 2>&1 &
echo $! > logs/v3_training.pid
```

### Evaluación

```bash
# Sobre test set sellado v1 EN (default)
python -m evaluation.evaluate_variant --variant v1

# Sobre otro test set
python -m evaluation.evaluate_variant --variant v3 \
    --test_dir data/test_sealed/v2_es \
    --metadata seal_test_metadata/test_v2_metadata.json

# Con audios guardados
python -m evaluation.evaluate_variant --variant v2 --save_audio
```

### Reporte de costo

```bash
python -m analysis.training_cost_report --variant v2 --output docs/v2_training_cost.md
```

### Verificación de tests

```bash
# Verificar causalidad
python -m tests.test_causality

# Verificar test set no fue modificado
python -m scripts.verify_test_set
```

### Git workflow

```bash
git status
git tag -l  # ver tags actuales

# Después de entrenar y evaluar una variante:
git add [archivos específicos]
git commit -m "vN.0.0: descripción concreta"
git tag -a vN.0.0 -m "Descripción del tag"
git push origin main
git push origin vN.0.0
```

---

## Referencias bibliográficas centrales

Cuando se documenten decisiones o se explique el razonamiento, usar estas referencias:

1. **Tan & Wang 2018** — "A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement", Interspeech
   - Paper base de la arquitectura CRN

2. **Xu et al. 2022** — "PESQ-Optimized Reference-Free Neural Loss for Speech Enhancement"
   - Referencia para PESQNet y loop alternado

3. **Kumar et al. 2023** — "TorchAudio-Squim: Reference-less speech quality and intelligibility measures", ICASSP
   - Referencia para Squim como sustituto de PESQNet

4. **Braun & Tashev 2021** — "A Consolidated View of Loss Functions for Supervised Deep Learning-based Speech Enhancement"
   - Referencia para loss combinadas MSE + SI-SDR

5. **Reddy et al. 2020** — "The INTERSPEECH 2020 Deep Noise Suppression Challenge"
   - Referencia para SNR ranges y test set methodology

6. **Schröter et al. 2022** — "DeepFilterNet2"
   - Referencia para arquitectura de dos etapas y procesamiento por banda

7. **Panayotov et al. 2015** — "Librispeech: An ASR corpus based on public domain audio books", ICASSP
   - Paper del dataset LibriSpeech

8. **Snyder et al. 2015** — "MUSAN: A Music, Speech, and Noise Corpus"
   - Paper del dataset MUSAN

9. **Piczak 2015** — "ESC: Dataset for Environmental Sound Classification"
   - Paper del dataset ESC-50

10. **Le Roux et al. 2019** — "SDR — Half-baked or Well Done?", ICASSP
    - Referencia para SI-SDR como métrica y loss

---

## Documentación complementaria

En la carpeta `docs/entregas/` se encuentran los documentos ya entregados durante el proyecto:

- Preproyecto defendido (30/06/2026)
- Anteproyecto (formato cátedra)
- Plan de Gestión de Calidad
- EDT v3 (65 paquetes de trabajo)
- Cronograma v3 con camino crítico
- Matriz de riesgos
- Presentación de defensa preproyecto

**Antes de crear nueva documentación, revisá si algo similar ya existe en `docs/entregas/`.**

---

## Cómo interactuar con Gabriel

### Estilo de comunicación esperado

- **Directo y honesto**: sin adulación ni suavizados innecesarios
- **Técnico pero accesible**: analogías cuando ayudan
- **Español informal argentino**: "vos", "querés", "acordate"
- **Justificaciones técnicas** con literatura cuando aplica
- **Trade-offs explícitos** cuando hay múltiples opciones
- **Predicciones honestas** (probabilísticas, no dogmáticas)

### Qué evitar

- Adulación o pleitesía innecesaria
- Sobrevender resultados ("¡Excelente!" seguido de nada sustantivo)
- Emojis (excepto casos muy puntuales)
- Bullets excesivos cuando prosa alcanza
- Respuestas largas cuando corta es mejor
- Asumir que Gabriel es principiante (es senior técnico)

### Cuando surge duda

**Preguntar antes de asumir**. Si algo puede interpretarse de dos formas, preguntar cuál. Es mejor que dar por sentado y equivocarse.

---

## Trabajo delegado a agentes

`~/nosiesuppressnet-oracle/PROMPTS_REANALISIS.md` (fuera del repo) tiene el proceso completo: roles, contrato de API, compuertas y
prohibiciones anti-sesgo. Dos reglas que se ganaron con incidentes reales:

1. **Los valores esperados viven FUERA del repo**, en `~/nosiesuppressnet-oracle/`. Dos
   agentes se contaminaron por un `grep -r` rutinario sobre `docs/` cuando estaban adentro.
   Prohibir la lectura no alcanza si el archivo es barrible.
2. **Nunca el mismo agente escribe el análisis y sus tests.** El modo de falla dominante no
   es que el código se rompa: es que pase. Los tres bugs de contrato que aparecieron salieron
   de auditar las "ambigüedades resueltas" que declaran los agentes, no de los tests.

## Anti-patrones a evitar en el código

### 1. Reinventar la rueda

Si existe función en `torchaudio` para algo (STFT, resample, PESQ), usarla. No implementar desde cero salvo justificación técnica.

### 2. Sobre-ingeniería

- NO crear clases si funciones alcanzan
- NO usar patrones de diseño (Factory, Singleton) sin necesidad clara
- NO abstraer prematuramente

### 3. Cambios sin ablation

**NUNCA modificar dos variables entre variantes**. Si querés cambiar loss Y arquitectura, hay que hacerlo en variantes separadas para poder atribuir causalidad.

### 4. Regenerar test set

**El test set sellado NO se regenera**. Ni siquiera para "mejorar" algo. Si se necesita otro test set, se sella uno nuevo (v3, v4) y se mantienen todos.

### 5. Hardcoded paths

Malo:
```python
audio = torchaudio.load("/home/gabriel/data/train/audio.wav")
```

Bueno:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
audio = torchaudio.load(PROJECT_ROOT / "data" / "train" / "audio.wav")
```

### 6. Try/except silencioso

Malo:
```python
try:
    load_audio(path)
except:
    pass
```

Bueno:
```python
try:
    load_audio(path)
except (RuntimeError, FileNotFoundError) as e:
    logger.warning(f"Failed to load {path}: {e}")
    errors.append(str(path))
```

---

## Sobre cómo mantener este archivo actualizado

Al final de cada sesión productiva con Claude Code, revisar:

1. ¿Cambió el "Estado actual"? Actualizar checkboxes.
2. ¿Apareció un hallazgo técnico importante? Agregarlo a "Hallazgos técnicos clave".
3. ¿Cambió alguna convención? Reflejarlo en "Convenciones".
4. ¿Se creó un nuevo archivo importante? Actualizar "Estructura del repositorio".

Este archivo se degrada rápido si no se mantiene. Un CLAUDE.md desactualizado genera más problemas que ayuda.

