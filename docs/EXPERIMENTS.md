# Experimentos — NoiseSuppressNet

Bitácora de experimentos del proyecto. Cada entrada documenta configuración,
resultados cuantitativos, diagnóstico y decisiones tomadas. Se ordena
cronológicamente. Cada entrada corresponde a un tag git.

---

## V0 — Baseline CRN, primer entrenamiento end-to-end

**Tag git:** `v0.1.0`
**Fecha:** 22 junio 2026
**Duración total del sprint:** ~2 semanas (16 al 30 de junio)
**Checkpoint:** `checkpoints/v0/best.pt` (disponible como asset del release v0.1.0)
**Referencia:** Tan, K. & Wang, D.L. (2018). *A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement*. Interspeech 2018.

### Objetivo del experimento

Validar el pipeline end-to-end del proyecto reproduciendo el paper de Tan & Wang 2018
con un dataset reducido. **No** se busca aún calidad perceptual — se busca demostrar
que la arquitectura está correctamente implementada, que el modelo aprende (loss
decrece), y que se cumplen las restricciones operativas (causalidad + RTF < 1.0).

### Configuración del modelo

| Parámetro | Valor | Justificación |
|-----------|-------|----------------|
| Arquitectura | CRN (Tan & Wang 2018) | Sweet spot causal/datos/reproducibilidad |
| Parámetros totales | 17,58 M | Coincide bit-exact con Figura 5 del paper |
| Encoder | 5 conv2d (1→16→32→64→128→256) | Kernel 2×3, stride (1,2), padding (1,0) |
| Bottleneck | LSTM 2 capas, hidden=1024, unidirectional | Causalidad garantizada por unidirectional |
| Decoder | 5 deconv2d simétricas con skip connections | `output_padding=(0,1)` en dec2 (F=39→80) |
| Activación de salida | Softplus | Garantiza magnitudes positivas |
| Loss | MSE sobre magnitud STFT | Fidelidad al paper (no SI-SDR) |
| Total parámetros LSTM | ~16,5 M (94% del modelo) | Concentración típica en RNN puros |

### Configuración de STFT

| Parámetro | Valor |
|-----------|-------|
| n_fft | 320 |
| hop_length | 160 (10 ms a 16 kHz) |
| win_length | 320 (20 ms) |
| Window | Hamming |
| Freq bins (F) | 161 |
| Implementación | `torch.stft` estándar (no custom) |

### Configuración de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 2e-4 (según paper original) |
| amsgrad | False |
| Scheduler | StepLR |
| LR decay | γ=0.98 cada 2 épocas |
| Batch size | 4 |
| Épocas | 10 |
| Seeds | `torch.manual_seed(42)`, `np.random.seed(42)` |
| Hardware | RTX 4060 8 GB, RAM 12 GB, i5-4460 |
| Tiempo total | 87 segundos |
| VRAM peak | ~1,5 GB (margen amplio) |

### Configuración del dataset

| Aspecto | Valor |
|---------|-------|
| Fuente voz | LibriSpeech train-clean-100 (28.539 clips) |
| Fuente ruido | MUSAN (2.016) + ESC-50 (2.000) |
| Sample rate | 16 kHz mono |
| Duración por clip | 4 segundos (64.000 samples) |
| SNR mixture | Continuo uniforme en [0, 15] dB (DNS Challenge style) |
| Normalización etapa 1 | RMS a unidad de potencia |
| Normalización etapa 2 | Peak a 0.9 (evita clipping WAV) |
| Total pares | 250 (200 train + 50 val) |

### Resultados de entrenamiento

**Curvas de loss (MSE sobre magnitud STFT):**

| Época | train_loss | val_loss | time (s) | lr |
|-------|------------|----------|----------|-----|
| 1  | 1,5267 | 1,2547 | 8,9 | 2.00e-04 |
| 2  | 1,2150 | 0,9486 | 8,7 | 1.96e-04 |
| 3  | 0,9530 | 0,6739 | 8,6 | 1.96e-04 |
| 4  | 0,7511 | 0,5318 | 8,8 | 1.92e-04 |
| 5  | 0,6520 | 0,5412 | 8,7 | 1.92e-04 |
| 6  | 0,6115 | 0,4920 | 8,9 | 1.88e-04 |
| 7  | 0,5892 | 0,4719 | 8,8 | 1.88e-04 |
| 8  | 0,5691 | **0,4427** ★ | 8,6 | 1.84e-04 |
| 9  | 0,5472 | 0,4601 | 8,8 | 1.84e-04 |
| 10 | 0,5188 | 0,4457 | 8,6 | 1.81e-04 |

★ Mejor checkpoint: epoch 8, val_loss = 0,4427. Guardado como `checkpoints/v0/best.pt`.

**Observaciones sobre el entrenamiento:**

- val_loss desciende monotónicamente en 8 de 10 épocas
- val < train de manera consistente → sin overfitting
- Tiempo por época estable (~8,7s) → DataLoader no es cuello de botella
- Reducción del 65% en val_loss (1,25 → 0,45) sin signos de estancamiento

### Resultados sobre val set (50 clips)

**Métricas perceptuales por condición:**

| Condición | PESQ-NB | STOI | SI-SDR (dB) |
|-----------|---------|------|-------------|
| Audio ruidoso (referencia) | 2,05 ± 0,69 | 0,842 ± 0,116 | 5,13 ± 6,65 |
| Butterworth LP 4 kHz | 2,05 ± 0,69 | 0,841 ± 0,117 | 4,21 ± 6,69 |
| **CRN v0** | **1,42 ± 0,16** | **0,794 ± 0,095** | **3,11 ± 3,95** |

**Interpretación honesta:** el modelo V0 no supera al audio sin procesar en
ninguna de las tres métricas. PESQ cae 0,63 puntos, STOI cae 0,05, SI-SDR cae
2 dB. **Este resultado es esperado en el régimen de datos utilizado y responde
a un modo de fallo específico documentado (ver Diagnóstico).**

### Benchmarks de eficiencia

**RTF (Real-Time Factor) sobre CPU Intel i5-4460, 1 thread:**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| RTF median | 0,331 | Procesa 3× más rápido que el audio |
| RTF p95 | 0,340 | Margen 3× sobre límite de tiempo real |
| Warm-up runs | 5 | Estabilización de cache/JIT |
| Total runs medidos | 20 | Estadística confiable |

**RTF sobre GPU (RTX 4060) — referencia:**

| Métrica | Valor |
|---------|-------|
| RTF median | 0,0148 |
| RTF p95 | 0,0151 |

### Validaciones técnicas ejecutadas

Cuatro validaciones automatizadas ejecutadas antes de entrenar. Todas reproducibles
con `python -m tests.<nombre>`.

| Validación | Método | Resultado | Umbral |
|------------|--------|-----------|--------|
| Causalidad estricta | Modificar input futuro, verificar salida pasada | diff = **0.00e+00** | < 1e-5 |
| Reconstrucción STFT | Round-trip `istft(stft(x))` sobre señal aleatoria | error = **9.54e-07** | < 1e-5 |
| Pipeline integrado | audio → STFT → CRN → STFT⁻¹ → audio con gradientes | grad_norm = **1.93** (finito) | 0.1 < x < 100 |
| Cantidad de parámetros | `sum(p.numel())` sobre model | **17.582.977** | ≈ 17.58 M (paper) |

### Diagnóstico: output collapse identificado

**Análisis de PSD promediada sobre los 50 clips del val set** (script:
`analysis/plot_psd.py`) reveló el modo de fallo del modelo.

**Firma cuantitativa del output collapse:**

| Métrica | Clean target | CRN v0 output | Interpretación |
|---------|--------------|----------------|----------------|
| PSD media en 4-8 kHz | Cae de -10 a -19 dB | Se mantiene plana en -4 dB | Modelo no discrimina |
| Std entre clips | ±10 dB | ±3 dB | Colapso a la media |
| Ratio std_out / std_target | 1.0 (referencia) | 0.30 | Firma confirmada |

**Interpretación:** El modelo no aprendió a discriminar voz de ruido en alta
frecuencia. Produce una salida casi constante para todos los inputs, cerca
del promedio del dataset. Este es un modo de fallo bien documentado.

**Causas identificadas (contribuciones convergentes):**

1. **Softplus + datos insuficientes → colapso a media del dataset.** Con solo 200 pares, la solución que minimiza MSE es aproximar la magnitud promedio de cada bin frecuencial.

2. **Magnitud sin fase no discrimina voz vs ruido en HF.** Sin información de fase, las fricativas de la voz son estadísticamente similares al ruido blanco en 4-8 kHz.

3. **MSE ponderada por energía → alta frecuencia recibe poco peso.** Como las magnitudes son chicas en HF, sus errores contribuyen poco a la loss total. El optimizer se concentra en LF y deja HF sin mejorar.

**Bibliografía de respaldo del diagnóstico:**

| Referencia | Aporte |
|-------------|--------|
| Reddy et al. (2021) DNS Challenge | Identifica std_out < 0.5 × std_target como firma de collapse |
| Tan & Wang (2018) sección 3.2 + Figura 4 | Reportan mejora perceptual recién en época 20 sobre 320.000 mezclas |
| Williamson et al. (2016) IEEE TASLP | Magnitude-only tiene techo por inconsistencia mag/fase |
| Braun & Tashev (2021) TSD | MSE tiene desbalance frecuencial documentado |
| Xu et al. (2022) IEEE TASLP | Motivación explícita de PESQNet: romper este modo de fallo |

**Cuantificación del gap:** 200 pares × 10 épocas está **1.500× por debajo** de lo
usado en el paper original (320.000 mezclas × 20 épocas). El colapso no invalida
la arquitectura — invalida el régimen de datos.

### Baseline Butterworth como sanity check

Se implementó un filtro pasabajos Butterworth como baseline trivial
(`baselines/butterworth.py`). Resultado: **el Butterworth no aporta**
a este SNR (PESQ +0.002 sobre noisy, STOI −0.001). Esto es informativo:

- Con SNR [0, 15] dB (medio ~5 dB), el ruido no está concentrado en HF
- Un filtro lineal simple no puede resolver el problema
- Confirma que la tarea requiere aprendizaje real, no filtrado

### Decisiones técnicas tomadas y documentadas

Registradas en `docs/decisions.md`:

- **MSE enmascarada sobre magnitud** en vez de SI-SDR (fidelidad al paper)
- **STFT torch.stft** en vez de implementación custom con Hamming+ReLU
- **Padding STFT simétrico** (limitación documentada, coincide con repo Wang)
- **Optimizer Adam lr=2e-4** (paper) en vez de 0.001 (default repo Wang)
- **CRN puro** (no GCRN variant como en el repo JupiterEthan)
- **`output_padding=(0,1)` en dec2** para recuperar F=39→80
- **Truncamiento `[:,:,:-1,:]` en encoder + `F.pad([0,0,1,0])` en decoder**
  para causalidad estricta bit-exact
- **RMS + peak normalization**: RMS a unidad de potencia + peak a 0.9
- **Seed torch + numpy** para reproducibilidad

### Validación externa

Consulta al Dr. DeLiang Wang (co-autor del paper original), 12 junio 2026:

- **Enviado:** 6 preguntas técnicas sobre reproducibilidad
- **Respondido:** el mismo día, en menos de 4 horas
- **Confirmado:** repositorio oficial `JupiterEthan/CRN-causal`
- **Derivación:** consultas técnicas más finas a Tan Ke (primer autor, ahora en Meta)

Correspondencia archivada en `docs/correspondence/wang_2026-06-12.md`.

### Auditoría automatizada del repositorio de referencia

Se ejecutó un agente Anthropic Claude (`noisesuppressnet-ai/scripts/analyze_crn.py`)
para comparar el paper Tan & Wang 2018 vs el código del repo JupiterEthan/CRN-causal.

**Hallazgos que impactaron el diseño:**

- STFT del repo usa Hamming + F.relu (aplicado sobre magnitud) — no necesario para reproducir
- Padding F.conv1d=160 en STFT del repo es simétrico, no estrictamente causal (documentado)
- Repo implementa GCRN (G=2 LSTM grouping) — mi V0 implementa CRN puro (más fiel al paper)
- Repo default lr=0.001, paper lr=0.0002 — usé el del paper
- Repo default batch=16, mi V0 usa batch=4 (limitado por VRAM 8GB)

### Repositorio y reproducibilidad

- **URL:** github.com/GabrielGuzAng/nosiesuppressnet
- **Tag reproducibilidad:** `v0.1.0`
- **Reproducción:** `git checkout v0.1.0 && python -m training.trainer`
- **Tiempo esperado:** ~90 segundos en RTX 4060, ~40 minutos en CPU i5-4460
- **Val_loss esperada final:** 0.4457 ± 0.005 (por variabilidad de PyTorch entre versiones)

### Lecciones aprendidas del sprint

**Lo que funcionó:**

- Empezar con dataset chico (200 pares) permitió detectar bugs con feedback de 90s en vez de horas
- Los tests automáticos de causalidad y STFT reveleron bugs antes del entrenamiento (fue crítico el `output_padding=(0,1)`)
- Consultar al autor original del paper dio validación externa que fortalece la defensa académica
- El análisis PSD promediado sobre 50 clips convirtió el "no funciona" en "modo de fallo identificado"

**Lo que aprendí:**

- val_loss bajando no implica calidad perceptual mejorando
- Cambiar SNR range de [0, 20] a [0, 15] dB (DNS style) mejoró la separación clean/noisy en la PSD
- El clipping en la peak normalization de mixtura era un bug silencioso (todos los rangos daban ±1.0) — solucionado escalando por 0.9
- La `output_padding=(0,1)` en dec2 es un detalle no documentado en el paper que solo aparece en el repo oficial

**Lo que haría distinto:**

- Definir el rango SNR desde el inicio siguiendo DNS Challenge (–5, 15) en vez de [0, 20]
- Escuchar audios generados con `torchaudio.save` **antes** de entrenar (para detectar clipping en fase de mixtura)
- Documentar decisiones técnicas en tiempo real (`docs/decisions.md`), no al final del sprint

### Estado al cierre del sprint (30/06/2026)

- ✅ Pipeline end-to-end funcional
- ✅ Modelo baseline entrenado (17,58 M params, causal bit-exact)
- ✅ Métricas objetivas medidas sobre val set (50 clips)
- ✅ Diagnóstico cuantitativo del modo de fallo (output collapse)
- ✅ RTF validado (0,34 en CPU modesto, 3× margen)
- ✅ Preproyecto defendido con éxito
- ⏳ V1 planificado con dataset escalado y más épocas

---

## Próximos experimentos planificados

Ver `docs/EDT_v3.docx` y `docs/cronograma_v3.docx` para el plan completo.

| Variante | Descripción | Hipótesis a validar |
|----------|-------------|---------------------|
| V1 | CRN + MSE + dataset 50k pares + 30 épocas | ¿Basta con más datos para romper el collapse? |
| V2 | V1 + loss combinada MSE + SI-SDR | ¿La combinación de losses aporta? |
| V3 | V1 + fine-tuning sobre Common Voice ES | ¿El español necesita adaptación específica? |
| V4 | V1 + PESQNet loss (EN) | ¿La loss perceptual rompe el output collapse? |
| **V5** ★ | V3 + PESQNet loss (ES) | **Hipótesis central del proyecto** |

Cada variante deberá agregarse a este documento cuando se complete su corrida,
siguiendo el mismo formato que V0: configuración, resultados, diagnóstico,
decisiones tomadas.

**Nota de septiembre 2026:** esta tabla es el plan de junio y se dejó como estaba. Lo que se
corrió difiere: V4 usó Squim en lugar de PESQNet y dio negativo, y V5 quedó redefinido por la
contingencia del riesgo R01 —español sobre la loss combinada de V2, sin proxy perceptual—. El
índice de variantes más abajo lista lo que efectivamente existe.


---

## V1 — CRN + MSE + dataset escalado (baseline reproducible)

**Tag git:** v1.0.0
**Fecha:** 27 julio 2026
**Checkpoint:** checkpoints/v1/best.pt (época 19)
**Referencia:** Baseline según Tan & Wang 2018, escalado a 50k pares.

### Configuración
- Dataset: 50.000 train + 2.000 val (LibriSpeech + MUSAN + ESC-50)
- Épocas: 30 (mejor época: 19 por val_loss)
- Batch size: 4, lr 2e-4, StepLR γ=0.98 cada 2 épocas
- Tiempo total entrenamiento: 15h 51min sobre RTX 4060
- Test set: sealed v1 (250 pares balanceados en 5 buckets SNR)

### Curva de entrenamiento
- Val_loss inicial: 0.153 (época 1)
- Val_loss mínimo: 0.0724 (época 19)
- Val_loss final: 0.087 (época 30, overfitting parcial)
- Convergencia efectiva en época ~5-9

### Resultados globales sobre test set sealed (n=250)
| Métrica | Noisy | V1 | Δ |
|---------|-------|------|---|
| PESQ-NB | 2.152 | 2.650 | +0.497 |
| PESQ-WB | 1.583 | 2.021 | +0.437 |
| STOI    | 0.852 | 0.904 | +0.051 |
| SI-SDR  | 7.86  | 13.80 | +5.95 dB |

**V1 supera a Noisy en las 4 métricas.**

### Análisis por bucket SNR
- Mejora máxima en PESQ: SNR [5,10] dB (+0.625)
- Mejora máxima en SI-SDR: SNR [-5,0] dB (+8.36 dB)
- Mejora mínima en STOI: SNR [15,20] dB (+0.011)

### Análisis por categoría de ruido
- MUSAN (n=137): ΔPESQ +0.500, ΔSI-SDR +6.07 dB
- ESC-50 (n=113): ΔPESQ +0.494, ΔSI-SDR +5.80 dB
- Sin overfitting a categoría específica.

### Conclusión
**El output collapse observado en V0 se rompió con dataset escalado.**
Escalar el dataset de 200 → 50.000 pares fue suficiente para lograr
mejora perceptual medible en todas las condiciones.

Implicancia para V2-V5: PESQNet como refinamiento sobre V1 (no como
solución al problema del collapse).


---

## V2 — CRN + loss combinada (MSE + SI-SDR)

**Tag git:** v2.0.0
**Fecha:** 2 agosto 2026
**Checkpoint:** checkpoints/v2/best.pt (época 19)
**Referencia:** Braun & Tashev 2021, "A Consolidated View of Loss Functions
for Supervised Deep Learning-based Speech Enhancement"

### Objetivo del experimento
Verificar si combinar loss espectral (MSE-magnitud) con loss temporal (SI-SDR)
mejora las métricas perceptuales sobre V1 (MSE puro).

### Configuración
- Dataset: mismo que V1 (50k train + 2k val)
- Épocas: 20 (menor que V1 porque ya conocíamos convergencia ~época 19)
- Loss: 0.7 * MSE_magnitud + 0.3 * SI-SDR (con escala 0.03)
- Batch size: 4, lr 2e-4, StepLR γ=0.98
- Seed: 42 (misma que V1 para ablation limpio)
- Tiempo total: 10.5 h sobre RTX 4060

### Curva de entrenamiento
- MSE_component final: 0.043 (vs V1 final 0.036)
- SI-SDR train final: -14.08 dB (mejora consistente desde -7.16)
- Val_loss mínimo en época 19 (mismo patrón que V1)

### Resultados sobre test set sealed (n=250)
| Métrica | Noisy | V1 | V2 | Δ V2-V1 |
|---------|-------|-----|-----|---------|
| PESQ-NB | 2.15 | 2.65 | 2.85 | +0.20 |
| PESQ-WB | 1.58 | 2.02 | 2.21 | +0.19 |
| STOI    | 0.85 | 0.90 | 0.91 | +0.01 |
| SI-SDR  | 7.86 | 13.80| 14.44| +0.64 dB |

**V2 supera a V1 en las 4 métricas.**

### Análisis por bucket SNR
Contraintuitivamente, V2 mejora **más en SNR altos** que en SNR bajos:
- Bucket [15, 20] dB: V2 Δ = +0.60 vs V1 Δ = +0.34 (Δ V2-V1 = +0.26)
- Bucket [-5, 0] dB: V2 Δ = +0.58 vs V1 Δ = +0.44 (Δ V2-V1 = +0.14)

Hipótesis: la componente SI-SDR temporal ayuda a preservar la voz cuando
el ruido es poco dominante (procesamiento "quirúrgico" en SNR alto).

### Análisis por categoría de ruido
- MUSAN: Δ PESQ +0.69 (n=137)
- ESC-50: Δ PESQ +0.71 (n=113)
Balance mantenido, sin overfitting a categoría.

### Conclusión
Combinar loss espectral y temporal aporta mejora perceptual medible
sobre MSE puro (+0.20 puntos PESQ). Braun & Tashev 2021 predice esto
teóricamente; V2 lo verifica empíricamente sobre este dataset.

Implicancia para V4/V5: el aporte de PESQNet debe medirse sobre V1
(baseline) Y sobre V2 (referencia superior). Si V4/V5 aportan sobre V2,
demuestran valor perceptual adicional al SI-SDR.


---

## V3 → V3b → V3e — Fine-tuning de V1 sobre Common Voice ES

**Tags git:** v3.0.0 (`82a2748`) y v3e.0.0 (`7f7884d`); V3b quedó sin tag (`49235c4`)
**Fechas:** 20 agosto 2026 (V3); commits de V3b y V3e, 23 y 24 de agosto
**Checkpoints:** `checkpoints/v3/best.pt` (época 21), `checkpoints/v3b/best.pt` (época 5),
`checkpoints/v3e/best.pt` (época 14) — **el recomendado es V3e**, y es de donde sale V5.
**Referencia:** Fine-tuning desde checkpoints/v1/best.pt. Ver docs/decisions.md
19/08/2026 (diseño) y 21/08/2026 (discusión metodológica de fine-tuning).

Las tres iteraciones son una sola línea de trabajo sobre la misma pregunta —cuánto cuesta en
inglés adaptar al español, y con qué receta—, así que van en una sola sección. Lo que sigue
hasta la conclusión de V3 es el registro original de agosto; después vienen V3b y V3e.

### Objetivo del experimento
Responder la pregunta central de V3: ¿el fine-tuning sobre español mejora el
desempeño del modelo frente a audio en español, comparado con V1 (entrenado
solo en inglés)? Para aislar esa única variable (idioma/dataset), V3 usa la
**misma loss, mismo lr (2e-4) y mismas 30 épocas que V1** — sin cambiar
ningún otro hiperparámetro de optimización.

### Configuración
- Dataset: Common Voice ES v26, 50.000 train + 2.000 val (`data/processed_es`),
  balanceado por género, splits train/dev oficiales sin leakage de hablante/frase
  (ver docs/decisions.md 17-18/08/2026)
- Init: checkpoints/v1/best.pt (pesos completos, sin capas congeladas)
- Loss: mse_magnitude — misma que V1, sin componente SI-SDR
- Épocas: 30 (mejor época: 21 por val_loss), batch size 4, lr 2e-4,
  StepLR γ=0.98 cada 2 épocas — idéntico a V1
- Seed: 42
- Tiempo total entrenamiento: ~15h 45min sobre RTX 4060
- Test sets: sealed v1_en (250 pares, EN) y v2_es (250 pares, ES, sellado
  20/08/2026 — ver docs/decisions.md)

### Metodología de fine-tuning — nota importante
Este es **full fine-tuning agresivo**: los 17,58 M parámetros quedan
entrenables (`requires_grad=True` en todos, sin excepción — verificado
contra el código), con el mismo lr y presupuesto de épocas que el
entrenamiento desde cero de V1. La bibliografía de transfer learning
(Yosinski et al. 2014, NeurIPS; Howard & Ruder 2018 ULMFiT, ACL) predice
que este régimen favorece la adaptación al nuevo dominio a costa de
"catastrophic forgetting" (McCloskey & Cohen 1989) del dominio original —
exactamente lo que se observa más abajo. V3b, planificado como siguiente
paso, explora fine-tuning conservador con lr determinado por barrido
empírico para caracterizar ese trade-off (ver docs/PLAN_V3B.md).

### Curva de entrenamiento
- Val_loss inicial: 0,1029 (época 1)
- Val_loss mínimo: 0,0821 (época 21)
- Val_loss final: 0,0854 (época 30)
- Train_loss desciende monótonamente: 0,0976 → 0,0367

### Resultados globales — V1 vs V3 sobre ambos test sets (n=250 c/u)

| | PESQ-NB | PESQ-WB | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| Noisy (test_v1_en) | 2,152 | 1,583 | 0,852 | 7,86 |
| V1 sobre test_v1_en | 2,650 (+0,497) | 2,021 (+0,437) | 0,904 (+0,051) | 13,80 (+5,95) |
| V3 sobre test_v1_en | 2,571 (+0,418) | 1,967 (+0,384) | 0,886 (+0,034) | 12,25 (+4,39) |
| Noisy (test_v2_es) | 2,161 | 1,641 | 0,850 | 8,41 |
| V1 sobre test_v2_es | 2,330 (+0,170) | 1,749 (+0,108) | 0,866 (+0,016) | 11,25 (+2,84) |
| V3 sobre test_v2_es | 2,560 (+0,399) | 1,980 (+0,338) | 0,890 (+0,040) | 13,34 (+4,92) |

**V3 vs V1, ambos sobre test_v2_es (español): +0,230 PESQ-NB, +0,231
PESQ-WB, +0,024 STOI, +2,08 dB SI-SDR.** El fine-tuning en español mejora
el desempeño sobre audio en español, en las 4 métricas.

**Costo en inglés: V3 vs V1 sobre test_v1_en: -0,079 PESQ-NB, -0,054
PESQ-WB, -0,018 STOI, -1,55 dB SI-SDR.** Forgetting real pero moderado —
V3 sigue mejorando sustancialmente sobre el audio ruidoso en inglés
(+0,418 PESQ-NB), no se degrada a nivel de V0.

### Análisis por bucket de SNR (test_v2_es)

| Bucket | V1 ΔPESQ-NB | V1 ΔSTOI | V1 ΔSI-SDR | V3 ΔPESQ-NB | V3 ΔSTOI | V3 ΔSI-SDR |
|---|---|---|---|---|---|---|
| [-5, 0] dB | +0,307 | +0,060 | +6,25 | +0,407 | +0,071 | +7,10 |
| [0, 5] dB | +0,267 | +0,040 | +5,79 | +0,481 | +0,077 | +7,86 |
| [5, 10] dB | +0,368 | +0,032 | +5,29 | +0,538 | +0,045 | +6,50 |
| [10, 15] dB | +0,073 | -0,018 | +0,36 | +0,424 | +0,018 | +3,84 |
| [15, 20] dB | **-0,167** | **-0,035** | **-3,49** | +0,147 | -0,010 | -0,68 |

**Hallazgo clave**: en el bucket de SNR alto [15,20] dB, **V1 degrada
activamente la señal en español** (PESQ y SI-SDR negativos) — el modelo
entrenado solo en inglés introduce artefactos sobre voz limpia en español
que un filtro nulo no introduciría. V3 corrige esto casi por completo
(PESQ vuelve a ser positivo; SI-SDR queda apenas negativo). Es el dato
más fuerte a favor de la hipótesis central del proyecto: el idioma
importa, y el fine-tuning específico lo remedia donde más se nota.

### Análisis por categoría de ruido

| | V1 sobre ES (n) | V1 ΔPESQ-NB | V3 sobre ES (n) | V3 ΔPESQ-NB |
|---|---|---|---|---|
| ESC-50 | 111 | +0,170 | 111 | +0,404 |
| MUSAN | 139 | +0,169 | 139 | +0,395 |

Sin overfitting a categoría específica en ninguno de los dos casos —
mejora pareja entre ESC-50 y MUSAN.

### Conclusión
**El fine-tuning en español responde que sí a la pregunta central de V3**:
mejora medible y consistente en las 4 métricas sobre `test_v2_es`, con
mayor impacto justamente donde V1 fallaba (SNR alto). El costo es un
catastrophic forgetting moderado en inglés, consistente con lo que
predice la bibliografía de transfer learning para full fine-tuning a lr
alto sin capas congeladas.

Implicancia para V3b: vale la pena explorar si un régimen de
fine-tuning más conservador (lr reducido, determinado empíricamente)
reduce el forgetting en inglés sin sacrificar la ganancia en español.
Eso es lo que se corrió a continuación.

### V3b — fine-tuning conservador: la hipótesis que no se sostuvo

**Commit:** `49235c4` (sin tag). **Checkpoint:** `checkpoints/v3b/best.pt` (época 5).

La hipótesis era directa: si el forgetting de V3 viene de un lr alto sin capas congeladas,
bajar el lr debería reducirlo sin perder demasiada ganancia en español. El lr no se eligió a
ojo — se barrió con `scripts/lr_sweep_v3b.py` + `..._evaluate.py`: 5 candidatos × 5 épocas,
13,07 h de GPU, `cudnn_deterministic=False` por ser corridas exploratorias descartables.

| lr | ΔPESQ-NB ES | ΔPESQ-NB EN | score compuesto |
|---|---|---|---|
| 5e-6 | +0,091 | −0,040 | +0,051 |
| 2e-5 | +0,132 | −0,031 | +0,101 |
| **5e-5** | **+0,154** | **−0,033** | **+0,121** |
| 1e-4 | +0,162 | −0,062 | +0,100 |
| 2e-4 | +0,139 | −0,109 | +0,030 |

V3b final corrió con el ganador: lr 5e-5, 10 épocas, StepLR γ=0,5 con decay en la época 4,
5,37 h, mejor checkpoint en la época 5. Sobre `test_v2_es` da PESQ-NB 2,483 (**+0,152** sobre
V1) y sobre `test_v1_en` 2,621 (**−0,028**). El forgetting bajó como se esperaba; la ganancia
en español también, y más de lo que se esperaba.

**Por qué no superó a V3, dicho sin maquillaje: el diseño confundió dos variables.** V3b
entrenó 10 épocas contra las 30 de V3, así que agresividad de lr y presupuesto de épocas
quedaron confundidos (*confounded*) y no se puede atribuir cuál de los dos pesa. Es una
violación de la regla del proyecto —una sola variable por vez— cometida en el propio diseño, y
el control que la desharía (V3c-b: lr conservador con 30 épocas) quedó pospuesto por tiempo y
nunca se corrió. El plan que guiaba esta iteración, `docs/PLAN_V3B.md`, nunca se versionó; lo
que sobrevive del barrido es `results/v3_sweep/` y la entrada del 22/08 en `decisions.md`.

### V3e — explotar el candidato que no había convergido

**Tag git:** v3e.0.0 (`7f7884d`). **Checkpoint:** `checkpoints/v3e/best.pt` (época 14,
`val_loss` 0,0843).

El sweep de V3b guardaba un dato que su propio criterio de selección tapaba: el candidato
lr=1e-4 tenía **la mayor ganancia en español de los cinco** (+0,162 a 5 épocas) y todavía no
había convergido; perdió por el término de forgetting del score. V3e retoma ese candidato y lo
corre 25 épocas con decay tardío (época 12, no la 4 de V3b), para darle recorrido al lr alto
antes de bajarlo. 13,19 h, `cudnn_deterministic=True`.

Resultado: PESQ-NB 2,551 sobre `test_v2_es` (**+0,221** sobre V1) y 2,619 sobre `test_v1_en`
(**−0,031**). Es decir, adapta casi como V3 y olvida casi como V3b. En el bucket [15,20] dB
del español —donde V1 degrada con −0,167— V3e llega a +0,126.

Dos precisiones sobre lo que se reportó en su momento: el forgetting en inglés es **−0,031**,
no el −0,032 que salió en el commit (redondeo mal hecho); y V3e corrió efectivamente
determinista, porque el "costo 2× del `cudnn_deterministic`" que justificaba desactivarlo no
reprodujo en ninguna medición posterior (`decisions.md`, 07/09/2026).

### Las tres iteraciones, una al lado de la otra

| | V3 | V3b | V3e |
|---|---|---|---|
| init | `v1/best.pt` | `v1/best.pt` | `v1/best.pt` |
| lr | 2e-4 | 5e-5 | 1e-4 |
| épocas (decay) | 30 (cada 2, γ 0,98) | 10 (ép. 4, γ 0,5) | 25 (ép. 12, γ 0,5) |
| mejor época | 21 | 5 | 14 |
| horas de GPU | 15,73 | 5,37 | 13,19 |
| Δ PESQ-NB en ES | **+0,230** | +0,152 | +0,221 |
| archivos que mejoran en ES | 89,2 % | 84,8 % | 90,0 % |
| Δ PESQ-NB en EN (media) | −0,079 | **−0,028** | −0,031 |
| archivos rotos en EN (< −0,2) | 17,6 % | **6,8 %** | 11,6 % |

La diferencia en español entre V3 y V3e es de 0,009 puntos; no se corrió un test apareado entre
las dos, así que no hay base para decir cuál adapta más. Lo que sí las separa es la cola en
inglés: V3e rompe 11,6 % de los archivos contra 17,6 % de V3, por esos 0,009 de diferencia en
español.

### Por qué el "score compuesto" quedó retirado

Las tres iteraciones se compararon en su momento con `score = ΔES + ΔEN` (V3 +0,151, V3b +0,124,
V3e +0,189). **El reanálisis estadístico de septiembre lo retiró**: sumaba dos estimandos de
naturaleza distinta. En español la media es representativa —mejora el 85-90 % de los archivos,
test de signo p ≈ 1e−30 a 1e−41 (familia F2)—, pero en inglés la mediana casi no se mueve y el
signo es una moneda (F3: prop_improve 0,47-0,53, p_Holm = 1): la media negativa la arrastra una
cola izquierda pesada, con skew de −2,2 a −3,5. Promediar una media representativa con una
arrastrada por su cola da un número que no mide nada.

**El estimando correcto del olvido es la fracción de archivos que se rompen**, y esa fracción es
monótona con la agresividad del lr: 17,6 % (V3) / 11,6 % (V3e) / 6,8 % (V3b). Con ese estimando
V3e sigue siendo la elección —adapta a la par de V3 y rompe 6 pp menos—, pero la comparación
deja de ser un escalar: V3b rompe menos que los otros dos y adapta menos que los otros dos, y
cuál conviene depende de cuánto pese el inglés en el uso final.

### Conclusión de la línea V3 → V3b → V3e

La línea queda cerrada con **V3e como checkpoint recomendado**, y es de donde sale la receta de
V5 (lr 1e-4, 25 épocas, decay tardío). Lo que deja, más allá del checkpoint:

1. **La hipótesis de partida de V3b era razonable y falló**: bajar el lr reduce el forgetting,
   pero no gratis, y el diseño con el que se probó no permite atribuir cuánto de la pérdida en
   español fue el lr y cuánto las 20 épocas menos. La lección se aplicó en V4b, donde el 2×2 se
   diseñó explícitamente para no repetirla.
2. **Lo que hizo funcionar a V3e no fue un lr nuevo sino un criterio de selección distinto**:
   mirar qué candidato del sweep tenía más recorrido en vez de cuál quedaba mejor a 5 épocas. En
   este modelo los sweeps cortos eligen por velocidad de convergencia. Ese mismo razonamiento es
   el que se escribió **antes** de correr el screening de lr de V5.
3. **Un aviso sobre la comparabilidad**: V1, V2 y V3 se entrenaron sin `cudnn_deterministic` y
   V3b/V3e con él. Ninguna de las tres primeras es bit-exacta, y eso no se corrige hacia atrás.

---

## Índice de variantes y dónde está documentada cada una

Todas las variantes, de V0 a V7, tienen sección propia en este archivo. Lo que vive afuera es
el detalle largo: los documentos por experimento, las decisiones metodológicas y el reanálisis
estadístico. Los resultados negativos (V4, V4b, V6) están al mismo nivel de detalle que los
positivos — es deliberado, y es lo que le da crédito al resto.

| variante | qué es | documentación |
|---|---|---|
| V0 | baseline con 200 pares | este archivo |
| V1 | dataset escalado a 50k | este archivo |
| V2 | loss combinada MSE + SI-SDR | este archivo |
| V3 | fine-tuning agresivo a español | este archivo |
| V3b, V3e | fine-tuning conservador y su corrección | este archivo (dentro de la sección V3) + `decisions.md` |
| V4, V4b | proxy perceptual Squim; negativo | este archivo + `decisions.md` ("Cierre de V4", "V4b") + `docs/PLAN_V4.md` |
| V5 | propuesta completa + réplicas de semilla | este archivo + `decisions.md` (07-09/09), commit `cd0f443` |
| V6 | compuerta atornillada a V2; negativo | `docs/v6_compuerta.md` + sección acá |
| V7 | compuerta desde cero; contraste confirmado con tres semillas, atribución abierta | `docs/v7_compuerta_desde_cero.md` + sección acá + `decisions.md` (26/09) |

---

## V4 y V4b — Término perceptual con proxy fijo (Squim)

**Tags git:** v4.0.0 (`494cd28`) y v4b.0.0 (`afe896c`)
**Fechas:** 31 agosto 2026 (V4) y 1 septiembre 2026 (V4b)
**Checkpoints:** `checkpoints/v4_sweep/*` y `checkpoints/v4b/*`. **Ninguno se promueve a
variante**: no existe `checkpoints/v4/best.pt` y no va a existir.
**Plan y decisiones:** `docs/PLAN_V4.md`; `docs/decisions.md`, entradas "Diseño de V4"
(30/08), "V4: por qué Squim queda fijo" (30/08), "Cierre de V4" (31/08) y "V4b" (01/09).
**Referencias:** Xu, Strake & Fingscheidt 2022 (IEEE/ACM TASLP); Kumar et al. 2023
(ICASSP, Squim); de Oliveira et al. 2024 (Interspeech, PESQetarian); López-Espejo et al.
2023 (Speech Communication); Fu et al. [31] citado por Xu.

**V4 y V4b son un mismo experimento en dos protocolos, y por eso van en una sola sección.**
La pregunta es una: ¿sirve un proxy perceptual **fijo** como término de loss? V4b no es una
variante nueva, es la respuesta a la objeción que el resultado de V4 dejó abierta.

### Objetivo del experimento

V4 mide si un término perceptual diferenciable aporta sobre MSE puro. Es rama paralela a V2
—no fine-tuning, no stack sobre V2—: init aleatoria, mismo dataset EN y la receta de V1, con la
loss como única variable (el Paso 1 recorta épocas y scheduler para abaratar el barrido; ver
Configuración). La loss es
`mse_plus_squim = α·MSE_magnitud + (1−α)·(−PESQ̂)·squim_scale`, con la misma convención de `α`
que la ecuación 9 de Xu 2022 (`α` pesa el MSE).

Squim se eligió sobre PMSQE con evidencia leída: López-Espejo et al. 2023 entrenó PMSQE contra
otras cinco losses sobre WSJ0 con la misma arquitectura y PMSQE dio **el peor PESQ de las
seis**, pese a ser la aproximación cerrada derivada del propio algoritmo PESQ; y de Oliveira
et al. 2024 muestra que optimizar contra ese linaje es explotable hasta el absurdo (un "modelo"
de un solo parámetro que inserta un click alcanza PESQ 3,46 sin red neuronal).

El riesgo de gaming estaba declarado en el plan **antes** de correr nada, con monitoreo
obligatorio. V4 lo materializó; V4b pregunta de quién es la culpa: del proxy fijo o del régimen
de entrenamiento. Xu 2022 usa cuatro estabilizadores en su segunda etapa y V4 no tenía ninguno
— tres de los cuatro no requieren reentrenar el proxy.

### Configuración

**V4** — Fase 3, Paso 1 del plan: 5 corridas × 3 épocas, 14,60 h de GPU. Un control
(`control_mse`, α=1, sin término Squim) y cuatro candidatos con α ∈ {0,95; 0,90; 0,80; 0,70},
todos con init aleatoria, lr 2e-4 y sin StepLR para aislar el efecto de `α`. Dataset EN
(50k train + 2k val), batch 4, semilla 42, `squim_scale = 1,0`, `cudnn_deterministic=False`
por ser corridas exploratorias descartables. Scripts: `scripts/squim_scale_sweep.py` +
`scripts/squim_scale_sweep_evaluate.py`.

**V4b** — diseño 2×2 con placebo, 3 corridas × 3 épocas, 8,02 h, α=0,9. Las tres arrancan de
`checkpoints/v1/best.pt` —el estabilizador de Xu que sí se podía aplicar— y guardan
`save_every_n_epochs=1`: `v4b_main` (lr 2e-5, el valor de Xu), `v4b_lr2e4` (lr 2e-4, el
histórico del proyecto) y `v4b_placebo` (α=1, mismo `init_checkpoint`, mismo lr 2e-5, mismas
épocas, **sin término perceptual**). La cuarta celda del 2×2 —init aleatoria + lr 2e-5— es
inútil: casi no entrenaría. Scripts: `scripts/v4b_protocol_check.py` + `..._evaluate.py`.

Qué **no** se probó, con razones escritas: gradient accumulation a un update por época (con 3
épocas serían 3 updates desde V1 convergido: el nulo sería trivial) y la alternancia ⟨1-1⟩
reentrenando el PESQNet (es el contenido central del paper de Xu, fuera de alcance acá).

### Curva de entrenamiento: la firma del gaming

En las cinco corridas de V4 los `history.json` muestran `val_squim` (≈ `−PESQ̂` sobre la salida
del propio modelo) llegando a **−4,0/−4,4** hacia la época 3: Squim cree que su propia salida
tiene PESQ 4,0-4,4, mejor que cualquier variante real de este proyecto. El PESQ real medido en
esas mismas corridas cae a 1,1-1,8, **por debajo de Noisy** (2,15). El patrón está instalado
**desde la época 1** en las cinco, incluidas `alpha_090` y `alpha_095`, que el chequeo
automático de inestabilidad marcó como "sin señales" — no es un pico transitorio de pesos
aleatorios que un warm-start pueda mitigar, es un exploit que se refuerza cada época.

En V4b la brecha `pesq_hat − PESQ-WB real` crece monótona época a época: 2,04 → 2,30 → 2,36
(`v4b_main`) y 2,59 → 2,70 → 2,77 (`v4b_lr2e4`).

### Resultados de V4 — sobre `test_v1_en` (n=250)

| corrida | PESQ-NB | Δ vs Noisy | Δ STOI | Δ SI-SDR |
|---|---|---|---|---|
| Noisy | 2,152 | — | — | — |
| `control_mse` (α=1) | 2,233 | **+0,081** | +0,027 | +4,15 dB |
| `alpha_095` | 1,810 | −0,343 | −0,069 | +2,29 dB |
| `alpha_090` | 1,346 | −0,806 | −0,169 | −0,63 dB |
| `alpha_080` | 1,210 | −0,943 | −0,252 | −4,00 dB |
| `alpha_070` | 1,307 | −0,846 | −0,251 | −4,56 dB |

El control es la única corrida que mejora sobre Noisy, y lo hace en PESQ-NB, STOI y SI-SDR;
en PESQ-WB queda en −0,019, que es lo esperable con 3 épocas desde init aleatoria (los cuatro
candidatos con Squim están entre −0,22 y −0,51 en esa métrica). Hay relación dosis-respuesta en
`α` —más peso al proxy, peor resultado real— salvo un cruce menor entre 070 y 080. **`validos_por_criterio` quedó vacío**: ningún candidato pasa el criterio de
selección de Fase 3, escrito antes de correr.

Una hipótesis alternativa se exploró y **se descartó contra la fuente primaria**: que Squim
evaluara fuera de distribución por no haber visto salidas de una red parcialmente entrenada.
Kumar et al. 2023 (sección 4.4.1) dice que su dataset incluye salidas de sistemas GCRN "with
varying degree of performances" — Squim sí las vio. Lo que falla no es la amplitud del dataset
del proxy sino que queda **stale** respecto de un modelo que se mueve, que es lo que Xu reporta
de Fu et al. [31]: *"estimated PESQ scores increase while true PESQ scores decrease"*.

### Resultados de V4b — el placebo cambia la conclusión

El 2×2, en Δ PESQ-NB contra Noisy, con la celda de V4 como esquina:

| | lr = 2e-4 | lr = 2e-5 |
|---|---|---|
| **init aleatoria** | −0,806 (`alpha_090`, de V4) | — |
| **init V1 convergido** | −0,263 (`v4b_lr2e4`) | **−0,059** (`v4b_main`) |

Y las tres corridas de V4b en detalle, con el placebo al lado:

| corrida (época 3) | ΔPESQ-NB | ΔPESQ-WB | ΔSTOI | ΔSI-SDR | `pesq_hat` | brecha |
|---|---|---|---|---|---|---|
| `v4b_placebo` (α=1) | **+0,539** | +0,486 | +0,054 | +6,10 dB | — | — |
| `v4b_main` (α=0,9, lr 2e-5) | −0,059 | −0,006 | −0,029 | +3,83 dB | 3,936 | +2,36 |
| `v4b_lr2e4` (α=0,9, lr 2e-4) | −0,263 | −0,132 | −0,048 | +3,32 dB | 4,217 | +2,77 |

Contra Noisy, `v4b_main` da −0,059 y parece "casi arreglado". Contra el placebo, que corrió el
protocolo idéntico sin el término perceptual, **el término Squim cuesta −0,598 PESQ-NB**. Sin
ese control la conclusión hubiera sido la opuesta.

Daño atribuible al término, por régimen:

- V4 (init aleatoria, lr 2e-4): +0,081 → −0,806 = **−0,887**
- V4b (V1 convergido, lr 2e-5): +0,539 → −0,059 = **−0,598**

Los dos estabilizadores recuperan **~33 %** del daño; **dos tercios del gaming sobreviven al
protocolo de Xu menos la alternancia**. Es evidencia directa de que lo que hace funcionar ese
esquema es específicamente reentrenar el proxy, no el arranque ni el lr.

La brecha también descarta el confound de descalibración: el MAE nominal de Squim para WB-PESQ
es **0,142** (Kumar et al. 2023, Tabla 2), y las brechas observadas son ~16-20× ese error, con
`v4b_lr2e4` a 0,42 puntos del techo estructural de la sigmoide (4,64). La comparación va contra
PESQ-**WB**, no NB, porque la cabeza de Squim estima WB-PESQ.

### Sesgo de selección de `best.pt` (aplica retroactivamente a V4)

Los cinco checkpoints de V4 se evaluaron desde `best.pt`, y `Trainer` lo elige por `val_loss`
mínima — que para `mse_plus_squim` **incluye** el término `(1−α)·(−PESQ̂)`. Más gameado = menor
`val_loss` = "mejor" checkpoint: el criterio estaba contaminado por lo mismo que se quería
medir. No cambia la conclusión (la tendencia es monótona en las tres épocas de las cuatro
corridas) pero los números de la tabla de V4 son los del punto más gameado de cada corrida. De
ahí la regla, aplicada ya en V4b: **con losses que incluyen el proxy, `save_every_n_epochs=1` y
evaluación época por época, nunca `best.pt`.**

### Hallazgo lateral: V1 no estaba del todo convergido

`v4b_placebo` en la época 2 —V1 más 2 épocas a lr 2e-5, sin ningún cambio de loss— da PESQ-NB
**2,716** / PESQ-WB 2,102 / STOI 0,906 / SI-SDR 13,945, mejor que V1 (2,650 / 2,021 / 0,904 /
13,805) en las cuatro métricas, **pese a tener peor `val_loss` MSE** (0,0836 contra 0,0724). El
criterio de selección por mínimo de val MSE no está alineado con PESQ. Dos implicancias: los
números reportados de V1 son levemente pesimistas, y el placebo es un control fuerte —no un
modelo degradado—, lo que refuerza la atribución de −0,598. Re-reportar V1 con ese checkpoint
quedó como decisión abierta: cambiaría todas las comparaciones V1→V2/V3/V3e ya taggeadas.

### Conclusión

**La línea de proxy perceptual queda cerrada, sin más variantes.** El espacio de decisión está
cubierto por los dos extremos: régimen inestable (V4, destruye el modelo) y régimen estabilizado
según la propia literatura (V4b, sigue destruyendo dos tercios). Lo único no probado es la
alternancia con reentrenamiento del proxy, descartada por alcance con razones documentadas.

Es un resultado negativo, y es el mejor caracterizado del proyecto. Lo que deja:

1. **Replicación independiente de Fu et al. [31]** con otro proxy (Squim en vez de Quality-Net)
   y otra arquitectura, sobre datos propios.
2. **Atribución causal con controles, no observación bruta**: ~1/3 del daño es init aleatoria +
   lr alto, ~2/3 es el gaming del proxy en sí.
3. **Un diagnóstico barato y reusable**: la brecha `pesq_hat − PESQ-WB` contra el MAE nominal
   de 0,142, que detecta gaming sin esperar a la evaluación completa.
4. **La confirmación de que el placebo no es opcional** cuando el protocolo cambia junto con la
   variable de interés. Es la lección que V6 y V7 heredan.

Efecto sobre el plan: la línea de PESQNet era el riesgo R01 de la matriz —el más alto, VME
3,5 días— y su contingencia declarada era caer a la loss combinada de V2. Eso es lo que hace V5.

**Limitaciones declaradas:** `cudnn_deterministic=False` en las ocho corridas (ninguna es
bit-exacta); 3 épocas por corrida contra las 25 de la segunda etapa de Xu; y la Fase 8 del plan
(escucha dirigida) no se corrió porque las métricas ya habían decidido el resultado.

---

## V5 — La propuesta completa: V2 fine-tuneado a español con la receta de V3e

**Tag git:** v5.0.0 (`cd0f443`)
**Fechas:** 7 septiembre 2026 (corrida principal), 8-9 septiembre 2026 (réplicas 43 y 44)
**Checkpoint:** `checkpoints/v5/best.pt` (época 21, `val_loss` −0,0792)
**Decisiones:** `docs/decisions.md`, entradas "V5: la contingencia de R01 rinde más que la
línea que reemplaza" (07/09), "Qué miden las réplicas de semilla de V5" (08-09/09) y
"Corrección: el 67 % de retención entre canales no se sostiene" (09/09).

**V5 no es una celda del ablation, y hay que enunciarlo así.** La cadena V1→V2→V3e está cerrada
y es la que da atribución causal, una variable por vez. V5 combina lo que funcionó y su rol es
ser el mejor sistema, no aislar un efecto: cambia dos cosas a la vez respecto de V3e, y un
jurado va a objetarlo si el informe no lo dice primero.

### Objetivo del experimento

El anteproyecto define V5 como "español + PESQNet". PESQNet/Squim quedó descartado con
evidencia propia (V4 y V4b): era el riesgo R01 de la matriz —el más alto, VME 3,5 días— y su
contingencia declarada era caer a la loss combinada de V2. V5 es esa contingencia, y rinde más
que la línea que reemplaza. Dos cambios sobre V3e, los dos deliberados:

1. **Arranca de `checkpoints/v2/best.pt` y conserva su loss combinada MSE + SI-SDR.** Van
   juntos: fine-tunear con MSE puro desde un checkpoint que ganó +0,20 PESQ por la loss
   combinada desharía la mitad de esa ganancia. V3/V3b/V3e partieron de V1 con MSE puro por
   disciplina de ablation, y la línea española nunca había heredado la mejora de V2.
2. **Entrena sobre `data/processed_es_wide`**, con SNR ~ uniform(−5, 20) en vez de
   uniform(−5, 15). El bucket [15,20] de los test sellados quedaba entero fuera de la
   distribución de entrenamiento, y es justo donde vive el hallazgo principal. Se declaró por
   adelantado que meterlo en distribución podía **atenuar** el efecto observado: es un test de
   robustez, no un amplificador.

Un tercer cambio se evaluó y se descartó: `center=False`. Se creía que el lookahead de 10 ms
venía del padding simétrico de la STFT y se midió que no — el retardo sigue `n_fft − hop` con
padding causal, porque lo impone el overlap-add de la síntesis (cinco configuraciones
verificadas en `tests/test_causality.py`). V5 no lleva el cambio.

### Configuración

- Init `checkpoints/v2/best.pt`; loss `mse_plus_sisdr` α 0,7, `sisdr_scale` 0,03
- Dataset `data/processed_es_wide` (50k train + 2k val, SNR uniform(−5, 20)); batch 4, semilla 42
- Receta de V3e: lr 1e-4, 25 épocas, StepLR γ=0,5 con decay tardío en la época 12
- `cudnn_deterministic=True` (corrida reportable); los tres sellados como test

**Screening de lr en vez de sweep, con criterio escrito antes de correr.** Tres corridas de 3
épocas (5e-5, 1e-4, 2e-4). La lección de V3b es que en este modelo un sweep corto elige por
convergencia rápida y se equivoca, así que el criterio fue *descartar* el lr que degrade o
diverja y, si los tres pasaban, usar 1e-4 por la evidencia de las corridas convergidas. Sobre
`test_v2_es` dieron 2,568 / 2,567 / 2,572 y ninguna diferencia sobrevive un test apareado
(p entre 0,44 y 0,87): se aplicó la regla y se eligió 1e-4 **aunque 2e-4 quedó nominalmente más
alto**, que es la situación exacta para la que se escribió.

### Curva de entrenamiento

25 épocas, 13,25 h sobre RTX 4060 (1.908 s/época), mejor checkpoint en la época 21 con
`val_loss` −0,0792. Eficiencia 83,5 % (tiempo hasta el mejor checkpoint sobre tiempo total).

### Resultados sobre los tres sellados (n=250 c/u)

PESQ-NB, corrida principal (semilla 42):

| test set | ruidoso | V1 | V2 | V3e | **V5** |
|---|---|---|---|---|---|
| español, Common Voice (`test_v2_es`) | 2,161 | 2,330 | 2,451 | 2,551 | **2,686** |
| inglés, LibriSpeech (`test_v1_en`) | 2,152 | 2,650 | 2,849 | 2,619 | **2,774** |
| español, audiolibro (`test_v3_mls_es`) | 2,195 | 2,602 | 2,762 | 2,686 | **2,840** |

Las cuatro métricas de V5, con delta contra el ruidoso:

| test set | PESQ-NB | PESQ-WB | STOI | SI-SDR (dB) |
|---|---|---|---|---|
| `test_v2_es` | 2,686 (+0,526) | 2,103 (+0,461) | 0,896 (+0,046) | 14,27 (+5,85) |
| `test_v1_en` | 2,774 (+0,622) | 2,167 (+0,584) | 0,898 (+0,046) | 13,42 (+5,56) |
| `test_v3_mls_es` | 2,840 (+0,645) | 2,196 (+0,560) | 0,906 (+0,055) | 13,64 (+5,61) |

En español: **+0,136 sobre V3e** (p = 2e−17) y **+0,235 sobre V2** (p = 1e−29), apareado, con
mejora en las cuatro métricas. **Cruza OP-1 (PESQ-NB ≥ 2,5) en los tres sellados** y con margen
—2,686 / 2,774 / 2,840—; V3e también lo cruzaba en los tres, pero al filo en español
crowdsourced (2,551). En PESQ-WB ninguna variante del proyecto llega a 2,5.

En el bucket [15,20] dB de `test_v2_es` —donde V1 (−0,167) y V2 (−0,087) degradan activamente
la señal— V5 llega a **+0,321** de ΔPESQ-NB y +0,82 dB de ΔSI-SDR. Es el bucket que el rango
ancho de SNR metió en distribución, así que la mejora no es sólo del idioma.

**El olvido cambia de referencia.** V3/V3b/V3e partían de V1 y su olvido se medía contra V1;
V5 parte de V2, así que su referencia es V2: **−0,075** (p = 0,004) con **16,8 %** de archivos
que caen más de 0,2, contra **+0,125** (p = 1e−15) y 9,2 % si se lo mide contra V1. Las dos son
ciertas y dicen cosas distintas: la honesta es la primera —V5 pierde algo respecto de su propio
punto de partida— pero omitir la segunda también engaña, porque **V5 es mejor que V1 en
inglés**, en PESQ-NB y PESQ-WB (en STOI y SI-SDR queda apenas por debajo: −0,006 y −0,38 dB).
El 16,8 % está más cerca de V3 (17,6 %) que de V3e (11,6 %), coherente con haber corrido la
receta más agresiva durante 25 épocas.

### Réplicas de semilla, y qué no habilitan decir

Se corrieron las semillas 43 y 44 además de la 42. **Las tres parten del mismo
`checkpoints/v2/best.pt` y el CRN no tiene dropout**, así que lo único que la semilla mueve es
el orden en que el fine-tuning ve los datos: miden la varianza del **fine-tuning**, no la de
inicialización.

| sellado | métrica | s42 | s43 | s44 | media | sd | s42 − media |
|---|---|---|---|---|---|---|---|
| `test_v1_en` | PESQ-NB | 2,7744 | 2,7697 | 2,7660 | 2,7700 | 0,0043 | +0,0044 |
| `test_v1_en` | PESQ-WB | 2,1667 | 2,1270 | 2,1428 | 2,1455 | 0,0200 | +0,0212 |
| `test_v1_en` | STOI | 0,8980 | 0,8992 | 0,9001 | 0,8991 | 0,0011 | −0,0011 |
| `test_v1_en` | SI-SDR | 13,4207 | 13,3631 | 13,5076 | 13,4305 | 0,0727 | −0,0097 |
| `test_v2_es` | PESQ-NB | 2,6863 | 2,6582 | 2,6614 | 2,6686 | 0,0154 | +0,0177 |
| `test_v2_es` | PESQ-WB | 2,1026 | 2,0436 | 2,0597 | 2,0687 | 0,0305 | +0,0340 |
| `test_v2_es` | STOI | 0,8960 | 0,8947 | 0,8971 | 0,8959 | 0,0012 | +0,0001 |
| `test_v2_es` | SI-SDR | 14,2657 | 14,0792 | 14,1542 | 14,1664 | 0,0938 | +0,0993 |
| `test_v3_mls_es` | PESQ-NB | 2,8404 | 2,8030 | 2,8159 | 2,8198 | 0,0190 | +0,0206 |
| `test_v3_mls_es` | PESQ-WB | 2,1961 | 2,1411 | 2,1489 | 2,1620 | 0,0298 | +0,0341 |
| `test_v3_mls_es` | STOI | 0,9055 | 0,9051 | 0,9047 | 0,9051 | 0,0004 | +0,0004 |
| `test_v3_mls_es` | SI-SDR | 13,6442 | 13,5600 | 13,5763 | 13,5935 | 0,0447 | +0,0507 |

**La advertencia de reporte, con la precisión que corresponde: la semilla 42 es la más alta de
las tres en PESQ-NB y PESQ-WB sobre los tres sellados.** En STOI y SI-SDR no: a veces es la
más alta, a veces la del medio, y en STOI sobre `test_v1_en` es **la más baja**. O sea que el
titular del proyecto —los PESQ de V5— sale del máximo de tres corridas, y eso se dice cada vez
que se cita el número. No es "la más alta en todo", y escribirlo así sería falso.

Qué habilitan: la ventaja de V5 sobre V3e en español es de siete a nueve desvíos de semilla, o
sea que el efecto no es orden de datos, y fijan la magnitud de ese ruido para diseños futuros.
**Qué no habilitan: decir "robusto a la semilla" a secas**, porque la varianza de
inicialización no se midió. La distinción parece pedante hasta que se necesita: en V7 los dos
brazos entrenan desde cero y estos 0,004-0,019 son el denominador equivocado para aquel efecto.

### Dos convenciones de reporte, porque el mismo nombre carga dos números

**En el ablation y en toda comparación interna, V5 es el checkpoint de la semilla 42**, tal
como está reportado en las tablas de arriba: cambiarlo retroactivamente rompería la
comparabilidad con V1, V2 y V3e, que no tienen réplicas y son números ya taggeados.
**En la comparación contra RNNoise y DeepFilterNet2 y en la de WER, el estimando es la media
de las tres semillas**, y así quedó fijado en el preregistro correspondiente.

No es una redefinición de V5: el modelo es el modelo y el estimando depende de contra qué se
compara, igual que en V6/V7 el estimando es la trayectoria promediada sin que eso redefina el
modelo. Las dos convenciones van escritas porque si el mismo nombre carga dos números sin
etiqueta, esa es exactamente la deriva que produjo la retractación del "67 % de retención".

### Barrido de época, y la decisión de no usarlo

`CONFIG_V5` no definió `save_every_n_epochs`, así que de la corrida principal sólo quedó
`best.pt` — limitación declarada, producto de un error de configuración. Las réplicas sí guardan
cada 3 épocas, y en las dos el pico está en la **época 18**, no en la de mínima `val_loss`:
+0,036 (s43) y +0,022 (s44), p < 0,001 apareado. Es la cuarta evidencia del mismo problema de
criterio de selección, después del placebo de V4b. **No se seleccionó nada con esto y no se va a
seleccionar**: está medido sobre el sellado, y elegir época mirándolo sería selección sobre el
test set. V5 se reporta con su `best.pt` de la época 21.

### Retención entre canales: el "67 %" quedó retractado

Con V2 evaluado sobre `test_v3_mls_es` se puede medir cada efecto desde su propio punto de
partida, que es lo que la primera versión del análisis no hacía:

| efecto | crowdsourced | audiolibro | retiene |
|---|---|---|---|
| `V2 − V1` (loss combinada, entrenada en inglés) | +0,121 | +0,160 | **132 %** |
| `V3e − V1` (fine-tuning al español desde V1) | +0,221 | +0,084 | **38 %** |
| `V5 − V2` (fine-tuning al español desde V2) | +0,235 | +0,079 | **34 %** |

**V5 retiene menos que V3e, no el doble**: el error era de estimando —`V5 − V1` mete adentro el
salto de la loss combinada, entrenada sólo en inglés— y la dependencia del canal no distingue
recetas de fine-tuning, distingue qué tipo de mejora es. Pendiente declarado:
`retention_by_recipe` en `analysis/reanalysis_stats.py` sigue calculando contra V1, así que
`results/reanalysis_stats.json` todavía publica el estimando equivocado.

### Conclusión

V5 es el mejor sistema del proyecto: cruza OP-1 en los tres sellados, mejora en las cuatro
métricas sobre V3e en español, y en inglés queda por encima de V1 en PESQ y por debajo de V2.
La contingencia de R01 rindió más que la línea que reemplazaba.

Tres reservas van al lado del resultado, no en un apéndice: **el titular sale del máximo de tres
corridas** (semilla 42, en PESQ); **V5 cambia dos variables a la vez** respecto de V3e, así que
no da atribución causal y no entra en el ablation; y **pierde 0,075 PESQ-NB contra su propio
punto de partida en inglés**, con 16,8 % de archivos rotos.

---

## V6 — Compuerta causal de paso directo (fine-tuning desde V2)

**Documento completo: `docs/v6_compuerta.md`.** Acá va el resumen de bitácora.

### Objetivo del experimento

El CRN de Tan & Wang 2018 hace mapeo espectral directo: la salida sale de
`softplus(bn1_t(conv1_t(d2)))` y no toca nunca la magnitud de entrada. Para dejar
un bin como está hay que reconstruirlo exactamente desde el cuello de botella
LSTM — no existe un camino barato para expresar "acá no hagas nada". La hipótesis
es que esa ausencia es el mecanismo por el cual el modelo degrada voz limpia fuera
de dominio, que es lo que V1 y V2 hacen en el bucket [15,20] dB del español.

Se verificó leyendo el código oficial de Tan & Wang 2020 (GCRN) que el sucesor
publicado del paper base **tampoco** tiene camino de identidad: sus compuertas son
GLU entre capas, gating de features internas, no un camino a la entrada.

### Configuración

M_out = g · M̂ + (1 − g) · M_noisy, con g aprendida, causal y por banda.
193 parámetros sobre 17.579.459 (+0,0011 %). Inicialización preservadora del nulo
(sesgo +3,0, g₀ = 0,953): el tratamiento arranca casi igual al placebo y sólo
puede aprender a replegarse.

| | arranca de | arquitectura | épocas | lr |
|---|---|---|---|---|
| placebo | `checkpoints/v2/best.pt` | CRN sin compuerta | 6 | 2e-5 |
| tratamiento | `checkpoints/v2/best.pt` | CRN con compuerta | 6 | 2e-5 backbone / 1e-3 compuerta |

Semilla 42, `cudnn_deterministic=True`, `save_every_n_epochs=1`, loss
`mse_plus_sisdr` α 0,7, datos en inglés. El estimando es tratamiento − placebo,
apareado archivo por archivo. Preregistro con hash en
`docs/preregistro_compuerta.sha256`.

### Verificaciones previas

- `CRN(gate=False)` es **bit-idéntico** al CRN original: cargando el mismo
  checkpoint en las dos clases, `max|diff| = 0,000e+00`.
- Causalidad de frame con la compuerta activa: `tests/test_causality.py`
  parametrizado con y sin compuerta, 24 tests.

### Resultados

| endpoint | criterio | resultado |
|---|---|---|
| P1 (primaria) | margen recuperado en [15,20] dB ES ≥ +3 pp, p < 0,05 | **falla**: +2,5 pp, p = 0,181 |
| P2 (costo acotado) | — | pasa |
| P3 (mecanismo) | *escrito con el signo invertido* | el mecanismo se cumple con ρ ≈ −0,4, p < 1e−8 |
| P4 (degeneración) | media de g en rango | pasa: 0,57 a 0,65 |

PESQ-NB global, checkpoint seleccionado:

| test set | V2 | placebo | compuerta | comp − plac |
|---|---|---|---|---|
| inglés, LibriSpeech | 2,8493 | 2,8963 | 2,8856 | −0,0106 (p=5e−03) |
| español, Common Voice | 2,4511 | 2,4635 | 2,4775 | +0,0141 (p=0,72) |
| español, audiolibro | 2,7615 | 2,8089 | 2,8055 | −0,0034 (p=0,21) |

### Conclusión

Por la regla de decisión preregistrada, **la hipótesis no queda sostenida**. Tres
cosas sobreviven al veredicto:

1. **El mecanismo existe y es fuerte**: ρ(g, SNR) ≈ −0,4 con p < 1e−8 en los tres
   sellados. La compuerta aprendió a replegarse sobre entrada limpia, que es
   exactamente lo que se buscaba.
2. **El placebo cambió la conclusión.** Contra V2 el tratamiento parecía mejorar
   +0,036 en inglés; contra el placebo con idéntico protocolo cuesta −0,011. Las
   seis épocas extra a lr 2e-5 valen +0,047 por sí solas.
3. **El piso de ruido por checkpoint** (sd 0,0099 ES / 0,0166 EN) es del mismo
   tamaño que el efecto buscado, lo que invalida el **diseño** de cualquier
   comparación de un solo checkpoint. Ver `decisions.md` (15/09/2026).

El costo en inglés que se reportó la primera mañana quedó **retractado**: sobre la
trayectoria de seis épocas el contraste es +0,004 con 3 de 6 positivas, centrado
en cero.

Diagnóstico que abre V7: la compuerta aprendió a detectar *"la entrada está
limpia"* —una señal de SNR disponible en el entrenamiento inglés— y no *"estoy
fuera de mi dominio"*, que no existe en datos de un solo idioma. Y la compuerta
estaba atornillada a un backbone que ya había convergido 20 épocas sin ella.

---

## V7 — La misma compuerta, entrenada desde cero

**Documento completo: `docs/v7_compuerta_desde_cero.md`.** Acá va el resumen de
bitácora.

### Objetivo del experimento

V6 dejó una ambigüedad que no se podía resolver con sus propios datos: su
endpoint falló, pero la compuerta estaba atornillada a un backbone que ya había
convergido 20 épocas sin ella. **V7 pregunta si con el camino de identidad
disponible desde la inicialización la red aprende una división del trabajo
distinta** — el decoder especializado en corrección residual en vez de síntesis
completa.

Es screening declarado: umbral +0,050, unas tres veces la sd por checkpoint que
midió V6, de modo que sólo puede detectar un efecto grande.

### Configuración

| | arquitectura | init | épocas | lr | loss | datos |
|---|---|---|---|---|---|---|
| control | CRN sin compuerta | aleatoria | 20 | 2e-4, StepLR(2 · 0,98) | mse_plus_sisdr α 0,7 | `data/processed` (EN) |
| tratamiento | CRN con compuerta | aleatoria | 20 | idéntico | idéntica | idéntico |

Semilla 42, `cudnn_deterministic=True`, `save_every_n_epochs=1`.
**El control se reentrena y no se reusa V2** (ver `decisions.md`, 15/09/2026).
Estimando: contraste global de PESQ-NB, compuerta − control, apareado archivo por
archivo, **promediado sobre las épocas 15 a 20**.

Preregistro con hash en `docs/preregistro_v7_desde_cero.sha256`, escrito antes de
crear las configs.

### Curva de entrenamiento

31,8 min/época en los dos brazos: 10,60 h el tratamiento y 10,58 h el control,
más ~1,5 h de barrido de evaluación. Total ~22,7 h de GPU.

Dos cortes de energía el 16/09 (06:50 y 13:08) mataron el control a mitad de la
época 15 y después el barrido. Se reanudó **bit-exacto** desde `epoch_14.pt`;
`tests/test_resume.py` verifica que los checkpoints de una corrida reanudada son
idénticos a los de la misma corrida sin cortar. Importaba: las épocas perdidas
eran justamente las que cargan el estimando primario.

### Resultados

| endpoint | criterio preregistrado | resultado | |
|---|---|---|---|
| E1 contraste global ES, media ép. 15-20 | ≥ +0,050 | +0,0795 | pasa |
| E2 costo en inglés | ≥ −0,020 | +0,0620 | pasa |
| E3 ρ(g,SNR), tres sellados | negativo, \|ρ\| ≥ 0,15 | −0,350 / −0,415 / −0,399 | pasa |
| E4 degeneración | media de g en (0,05 · 0,99) | 0,42 a 0,51 | pasa |
| E5 desde cero vs. tarde | contra el +0,0187 de V6 | 4,3× | desde cero rinde |

Contraste sobre los tres sellados, promediado sobre las épocas 15-20, con las seis
épocas positivas en los tres: **+0,0795** (español Common Voice) · **+0,0753**
(español audiolibro) · **+0,0620** (inglés LibriSpeech).

### Análisis por bucket de SNR

Δ PESQ-NB contra noisy en el bucket [15,20] dB, donde V1 y V2 degradan:

| | inglés / audiolibro | español / audiolibro | español / crowdsourced |
|---|---|---|---|
| V1 | +0,343 | +0,104 | −0,167 |
| V2 | +0,595 | +0,330 | −0,087 |
| V7 control | +0,561 | +0,318 | −0,084 |
| V7 compuerta | +0,647 | +0,412 | **+0,006** |

El control desde cero **replica la patología** que V1 y V2 muestran en español
crowdsourced, lo que la convierte en una propiedad de la arquitectura y la receta
y no de una variante puntual. La compuerta la cruza a cero.

### Conclusión

Por la tabla de desenlaces del preregistro el resultado cae en *"E1 ≥ +0,050 y E3
se cumple"*: la compuerta desde cero es un aporte arquitectónico real **y
corresponde confirmarlo con tres semillas antes de escribirlo como tal**.

Tres reservas van escritas al lado del resultado, no en un apéndice:

1. **Una sola semilla por brazo.** Las réplicas de V5 midieron sd entre semillas
   de 0,004 a 0,019; E1 es de 4 a 20 veces eso, improbable como ruido de semilla
   pero no descartado. Acá la semilla mueve además la inicialización, porque los
   dos brazos entrenan desde cero.
2. **El mecanismo propuesto no es la única explicación.** El contraste es parejo
   entre buckets y entre sellados, y el brazo con compuerta ajusta mejor en
   validación (`val_mse` 0,0899 contra 0,1026). Parte del efecto es que un camino
   de identidad facilita la optimización, como cualquier conexión residual.
3. **La sd entre épocas (0,0340 en español) es más ancha que la que midió V6**
   (0,0099), con la que se fijó el umbral. E1 lo pasa, pero por unos dos errores
   estándar, y los seis checkpoints no son muestras independientes.

Efecto sobre V6: queda **acotado, no retractado**. Su endpoint falló y sigue
fallando, pero falla para la compuerta agregada tarde, no para la compuerta.

### Confirmación con tres semillas nuevas

Corridas del 19 al 22/09/2026; endpoint adjudicado el 26/09/2026. La "Conclusión" de arriba es la
del screening y queda como registro. Lo que sigue la cierra.

**Diseño.** Semillas 43, 44 y 45, los dos brazos completos en cada una: seis corridas de 20
épocas con la receta del screening sin cambios. Estimando por semilla idéntico al del screening,
calculado por el mismo código congelado. **Confirmatorio: media no ponderada de las tres semillas
nuevas. La 42 queda excluida por diseño, porque fue la que disparó la confirmación, y se reporta
al lado.** Preregistro con hash en `docs/preregistro_v7_semillas.sha256` (`0424c74`), cerrado
antes de que existiera cualquier resultado sobre los sellados.

| sellado | confirmatorio | s43 | s44 | s45 | sd entre semillas | s42 (excluida) |
|---|---|---|---|---|---|---|
| `test_v2_es` (primario) | **+0,061370** | +0,073098 | +0,080308 | +0,030703 | **0,026802** | +0,079493 |
| `test_v1_en` | +0,044470 | +0,059170 | +0,038858 | +0,035381 | 0,012849 | +0,061970 |
| `test_v3_mls_es` | +0,044083 | +0,055806 | +0,036702 | +0,039740 | 0,010266 | +0,075347 |

| criterio | umbral | resultado | |
|---|---|---|---|
| C1 dirección, las tres semillas en ES | > 0 | la menor, +0,030703 | pasa |
| C2a magnitud media en ES | ≥ +0,050 | +0,061370 (margen +0,011370) | pasa |
| C2b piso por semilla en ES | ≥ +0,025 | la mínima, +0,030703 (margen +0,005703) | pasa |
| C3 costo en inglés | ≥ −0,020 | +0,044470 | pasa |
| C4 ρ(g,SNR) < 0 con \|ρ\| ≥ 0,15 | las 54 épocas (3 semillas × 3 sellados × 6) | 54 de 54; \|ρ\| mínimo 0,315119 | pasa |
| C5 media de g en (0,05 ; 0,99) | las 54 épocas | 54 de 54; g entre 0,385669 y 0,566935 | pasa |

250 pares en todas las épocas de las tres semillas, cero exclusiones.

**Nivel: preregistrado y confirmado.** Es fuera de muestra en la semilla, que es la variable que
se confirma. Los sellados son los mismos del screening.

**Se confirma el contraste compuerta − control, en signo y con +0,061370 PESQ-NB en el sellado
primario. No se confirma que la mejora provenga de la compuerta**: el preregistro excluye esa
atribución, y separarla pide un brazo con `g` constante aprendida que este experimento no tiene.
La reserva 2 del screening sigue en pie sin cambios.

**Fragilidad, al lado del número:** la sd entre semillas en español (0,026802) es mayor que el
margen de C2a sobre su umbral (+0,011370). La semilla 45 carga esa dispersión: +0,030703, con
dos de sus seis épocas negativas (`+ − + + − +`, sd entre épocas 0,042923). Lo que decidió el
resultado fue el piso por semilla: con la 45 0,005703 más abajo, el desenlace habría sido otro.

**Lo que corrige del screening.** La magnitud que se reporta es +0,061370, no +0,0795. La 42 es
la más alta de las cuatro en inglés y en audiolibro; en el sellado primario la supera la 44. El
orden entre sellados que el screening mostraba (Common Voice > audiolibro > inglés) no se
reproduce: audiolibro e inglés quedan a la par. El diferencial español − inglés (+0,016900) no
habilita afirmar que el efecto sea específico de estar fuera de dominio. El descriptivo del
bucket [15,20] dB sigue siendo de la semilla 42 sola.

Detalle y discusión: `docs/v7_compuerta_desde_cero.md` §9 y `decisions.md` (26/09/2026).
