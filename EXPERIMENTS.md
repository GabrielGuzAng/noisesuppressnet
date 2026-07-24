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
