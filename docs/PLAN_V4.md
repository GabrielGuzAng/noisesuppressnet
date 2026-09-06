# Plan V4 — CRN + MSE + Squim (proxy perceptual de PESQ), EN

**Estado:** planificado, no iniciado.
**Fecha del plan:** 30 de agosto de 2026.
**Decisión de diseño de fondo:** `docs/decisions.md`, entrada "Diseño de V4:
MSE + Squim (proxy aprendido) sobre PMSQE (aproximación cerrada), con
monitoreo obligatorio de gaming" (30/08/2026). Este documento no repite
esa justificación — la asume y la traduce en pasos ejecutables.

Este plan está escrito para que la próxima sesión arranque directo en
Fase 0 sin tener que reconstruir el razonamiento. Cada fase tiene
archivos concretos a tocar y un criterio de salida antes de pasar a la
siguiente — no avances de fase sin cumplir el criterio.

---

## Objetivo del experimento

Responder si un término perceptual (Squim, proxy diferenciable de PESQ)
combinado con MSE aporta sobre V1 (MSE puro) y, en la evaluación, sobre
V2 (MSE + SI-SDR) — sin que el modelo explote el proxy en vez de mejorar
calidad real. V4 es rama paralela a V2 (mismo punto de partida que V1,
no fine-tuning ni stack), única variable: la loss.

**Hipótesis:** `mse_plus_squim` mejora PESQ-NB/WB sobre V1 en magnitud
comparable o mejor que V2, sin degradar SI-SDR/STOI por debajo de V1 ni
por debajo de Noisy, y sin las firmas de gaming documentadas en
PESQetarian (de Oliveira et al. 2024).

**Criterio de cierre de V4 (para no quedar dando vueltas):** V4 se da por
cerrado cuando pasó Fase 7 (chequeo cruzado sin señales de alarma) y
Fase 8 (escucha dirigida sin las firmas de gaming conocidas), documentado
en `EXPERIMENTS.md`. Si alguna fase falla, el resultado (incluyendo el
fallo) se documenta igual — un negative result caracterizado es
material válido de tesis, no hay que perseguir "que funcione" a
cualquier costo.

---

## Fase 0 — Verificaciones rápidas antes de escribir código nuevo

No tocar `losses.py`/`trainer.py` todavía. Objetivo: confirmar que los
supuestos técnicos de los que depende todo lo demás siguen siendo
ciertos en el entorno actual.

- [x] Re-correr `python -m tests.test_squim_differenciable` (nota: el
      archivo tiene ese nombre con la falta de ortografía, no
      "differentiable") y confirmar que ambas soluciones siguen dando
      gradiente finito y no nulo. Si algo cambió (versión de
      torchaudio, etc.), documentarlo antes de seguir.
- [x] Verificar el sample rate esperado por `SQUIM_OBJECTIVE` — confirmar
      que es 16kHz (debería coincidir con el proyecto, pero no está
      verificado explícitamente en el test actual, que no lo declara).
- [x] **Pregunta abierta real, no resuelta:** `test_squim_differenciable.py`
      prueba con `torch.randn(1, 64000)` — ruido blanco sin normalizar a
      nivel de voz típico. El audio del proyecto está normalizado RMS a
      potencia unitaria + peak a 0.9 (`datasets/make_mixtures.py`). Antes
      de integrar a `trainer.py`, correr un smoke test manual: tomar un
      batch real de `NSDataset`, reconstruir audio con
      `stft.from_spec`, y pasarlo por Squim frozen para confirmar que no
      hay mismatch de escala (predicciones fuera de rango, gradiente
      nulo o explosivo). Si hay mismatch, puede hacer falta un
      re-escalado antes de alimentar a Squim (no antes del resto del
      pipeline).
- [x] Confirmar que `SQUIM_OBJECTIVE` acepta segmentos de 64.000 samples
      (4s, el `segment_samples` fijo del proyecto) sin requerir una
      duración mínima distinta.

**Resultados (30/08/2026):**

- `test_squim_differenciable.py`: PASS en las dos soluciones (cuDNN
  desactivado y `.train()` con pesos congelados) — `grad_norm` 6.01e-02 y
  6.15e-02 respectivamente, ambas finitas y no nulas. Corrido contra un
  tensor de 64.000 samples en las dos, lo que de paso confirma el cuarto
  punto (Squim no reclamó una duración mínima distinta).
- `SQUIM_OBJECTIVE.sample_rate == 16000` — coincide con el proyecto.
- Smoke test manual (`noisy`/`clean` real de `data/processed/train`,
  reconstruido con `stft.from_spec`, pasado por Squim frozen): rms=0.118,
  peak=0.900 preservados en el round-trip STFT. Predicciones Squim
  STOI=0.6447, PESQ=1.1355, SI-SDR=-5.1437 — todas dentro de rango físico
  razonable para audio ruidoso (no saturadas, no en el límite de escala).
  `grad_norm` = 3.66e-01, finito y no nulo. **Sin mismatch de escala** —
  no hace falta re-escalar antes de alimentar a Squim.

**Criterio de salida: cumplido.** Los cuatro puntos confirmados sin
señales de alarma. Pasa a Fase 1.

---

## Fase 1 — `training/losses.py`

Agregar, seguido de `mse_plus_sisdr` y con el mismo patrón:

```python
def mse_plus_squim(mag_est, mag_clean, pesq_hat, alpha=0.9, squim_scale=None):
    """
    alpha * MSE_magnitud + (1-alpha) * (-pesq_hat) * squim_scale

    pesq_hat: salida de SQUIM_OBJECTIVE sobre audio_est, ya computada
    por el caller (trainer.py posee el modelo Squim y el contexto
    cudnn.flags(enabled=False) — losses.py se mantiene stateless,
    igual que el resto del archivo).
    """
```

- `squim_scale` **no** se fija a priori — sale calibrado de Fase 3. Dejar
  un valor placeholder documentado como "sin calibrar" si hace falta
  correr algo antes de tener el número final.
- Actualizar `LOSS_REGISTRY` y `get_loss_name` si corresponde (mismo
  patrón que `mse_plus_sisdr`).
- Sanity test en el `if __name__ == "__main__":` del archivo, mismo
  estilo que los existentes (tensores dummy, verificar `.backward()` no
  falla y grad_norm > 0) — pero acá con `pesq_hat` dummy (no hace falta
  levantar Squim real para este test unitario).

**Criterio de salida:** `python -m training.losses` corre sin error y
confirma gradiente finito para `mse_plus_squim` con tensores dummy.

---

## Fase 2 — `training/trainer.py`

1. En `__init__`: si `self.loss_name == "mse_plus_squim"`, cargar
   `SQUIM_OBJECTIVE.get_model().to(self.device)`, `.eval()`,
   `requires_grad_(False)` en todos los parámetros. Guardar como
   `self.squim_model`.
2. En `_compute_loss`, agregar una rama nueva junto a `mse_magnitude` y
   `mse_plus_sisdr`:
   - Reconstruir `audio_est` con `self.stft.from_spec(...)` (ya existe,
     reutilizar tal cual como en la rama `mse_plus_sisdr`).
   - Computar `pesq_hat` así:
     ```python
     with torch.backends.cudnn.flags(enabled=False):
         _, pesq_hat, _ = self.squim_model(audio_est)
     ```
     Envolver **solo esta llamada**, no el resto del training step — así
     no interfiere con `cudnn_deterministic`/`cudnn.benchmark` que ya
     configura el LSTM del propio CRN.
   - Llamar `losses.mse_plus_squim(mag_est, mag_clean, pesq_hat, alpha=self.squim_alpha, squim_scale=self.squim_scale)`.
3. Agregar `self.squim_alpha = config.get("squim_alpha", 0.9)` y
   `self.squim_scale = config.get("squim_scale")` en `__init__`, mismo
   patrón que `loss_alpha`/`sisdr_scale`.
4. **Logging de componentes por época** (necesario para Fase 6/7, no
   opcional): extender `train_epoch`/`validate`/`fit` para trackear el
   componente Squim igual que ya se trackea `mse`/`sisdr` en
   `self.history`. Sin esto, no hay forma barata de ver la trayectoria
   del término perceptual época a época sin correr PESQ real sobre todo
   el val set en cada una.
5. **Checkpoints periódicos para V4** (cambio de comportamiento respecto
   a V1/V2/V3, justificado): agregar un flag de config
   `save_every_n_epochs` (ej. cada 5) que además del `best.pt` habitual
   guarde `checkpoints/v4/epoch_XX.pt`. Sin esto, `monitor_correlation.py`
   (Fase 4) solo puede correr contra el checkpoint final, y el chequeo
   cruzado de gaming (Fase 7) pierde toda su utilidad — la gracia es ver
   la trayectoria, no un solo punto al final.

**Criterio de salida:** correr el trainer 1-2 épocas sobre un subset
chico (o el sweep de Fase 3 ya cumple esto) sin errores de shape/device,
con `history.json` mostrando el componente Squim por época y al menos un
`epoch_XX.pt` guardado si `save_every_n_epochs` está seteado.

---

## Fase 3 — Mini-sweep de calibración (antes de comprometer 30 épocas)

Mismo criterio disciplinado que el sweep de lr de V3b
(`scripts/lr_sweep_v3b.py` como referencia de estilo). **Escalonado en
dos pasos, no un grid único** — decisión del 30/08/2026 tras medir el
overhead real de Squim (ver `decisions.md`): +105% por época, ~49.5
min/época extrapolado a escala completa (50k train + 2k val) vs. ~24.2
min/época sin Squim. Un grid completo de 4 alphas × warm-start (8
corridas × 4 épocas) sale ~26.4h; escalonado sale ~20h con la misma
cobertura real.

**Paso 1 — barrido de `alpha`, sin warm-start.** Script:
`scripts/squim_scale_sweep.py`.

- 5 corridas, dataset completo, `cudnn_deterministic: False` (corridas
  descartables, igual que `CONFIG_V3_SWEEP_BASE`), lr fijo sin
  scheduler, 3 épocas cada una (acotado desde 4 el 30/08/2026 para bajar
  el presupuesto de ~14.8h a ~11.1h, ver `decisions.md`), Squim activo
  desde época 1 (sin warm-start):
  - **Control**: `loss: mse_magnitude` (equivalente a `alpha=1.0`, sin
    término Squim — es la "corrida equivalente sin término Squim" del
    criterio de selección de abajo, entrenada al mismo presupuesto de 3
    épocas que los candidatos, no el V1 completo de 30 épocas, para que
    la comparación sea al mismo punto de convergencia).
  - **4 candidatos**: `alpha` ∈ {0.95, 0.9, 0.8, 0.7} (MSE dominante en
    todos los casos — ampliado desde {0.95, 0.9, 0.8} el 30/08/2026 tras
    leer Xu et al. 2022, ver discusión completa en `decisions.md`. El
    punto 0.7 busca ver si aparece una inflexión visible en el chequeo
    cruzado antes de acercarse a zona perceptual-dominante, no
    reproducir el `α=0` de Xu — ese resultado dependía de un mecanismo
    estabilizador que V4 no tiene, ver `decisions.md`).
- Checkpoints en `checkpoints/v4_sweep/<nombre>/`.
- Evaluación: `scripts/squim_scale_sweep_evaluate.py` — corre
  `evaluation.evaluate_variant` (vía subprocess, mismo patrón que
  `lr_sweep_v3b_evaluate.py`) sobre `test_v1_en` para cada uno de los 5
  checkpoints, arma tabla comparativa y aplica el criterio de selección.
- **Criterio de selección del ganador — no es "el que da mejor PESQ
  estimado por Squim solo"** (eso sería medir el régimen que se supone
  hay que vigilar). Es: el que mejora PESQ-NB real (evaluado, no
  estimado por Squim) **sin** que el SI-SDR o el STOI reales caigan por
  debajo del control. Ese es el chequeo cruzado de Fase 2/7, aplicado ya
  en esta etapa exploratoria.

**Paso 2 — warm-start, solo sobre el ganador de Paso 1** (o el top-2 si
quedan cerca en score). Script: `scripts/squim_warmstart_check.py`.

- El warm-start existe para mitigar un riesgo específico: inestabilidad
  en época 1 por audio-basura del CRN random alimentando a Squim. Si el
  ganador de Paso 1 no muestra esa inestabilidad en su propia curva de
  época 1 (`history.json`), correr Paso 2 igual pero documentar en
  `decisions.md` que se descartó por evidencia directa, no se salteó.
- Implementación: reutiliza `init_checkpoint` (ya existe en
  `trainer.py`, usado desde V3) — 1 época de `mse_magnitude` (config de
  warmup corta) seguida de 2 épocas de `mse_plus_squim` con el `alpha`
  ganador, partiendo del checkpoint del warmup. Mismo presupuesto total
  (3 épocas, acotado el 30/08/2026 — ver `decisions.md`) que los
  candidatos de Paso 1, para comparar manzanas con manzanas.
- Comparar contra el resultado **sin** warm-start del mismo `alpha` (ya
  disponible de Paso 1, no hay que re-correrlo).
- Registrar resultado de ambos pasos en `decisions.md` (mismo formato
  que el sweep de V3b), con el `alpha`/`squim_scale`/decisión de
  warm-start ganadores y el porqué.

**Criterio de salida:** un `alpha`/`squim_scale`/decisión de warm-start
elegidos y documentados, sin señales de inestabilidad (NaN, divergencia)
ni del patrón de gaming en ninguna corrida de ninguno de los dos pasos.

---

## Fase 4 — `evaluation/monitor_correlation.py` (no existe, hay que crearlo)

Diseño ya fijado en `decisions.md` — dos regímenes de medición, no uno:

1. **Baseline de referencia:** correlación (Pearson y Spearman) entre
   PESQ real (`evaluation/metrics.py::pesq`, ya existe) y el PESQ
   estimado por Squim, sobre los pares noisy/clean del test set sellado.
   Esperable ~0,90 LCC si Squim se comporta como Quality-Net.
2. **Régimen real de uso:** la misma correlación pero sobre la salida
   del CRN (`audio_est`) en cada checkpoint disponible
   (`checkpoints/v4/epoch_XX.pt` + `best.pt` de Fase 2). Esperable más
   bajo que (1) — eso es lo que hay que trackear por época, no un
   número aislado.
- Interfaz: standalone, reusable contra cualquier checkpoint (no
  hardcodeado a V4 — V5 lo va a necesitar también).
- Guardar resultado en `results/v4_correlation.json`, siguiendo la
  convención existente de `results/*.json`.
- No hace falta visualización todavía — un JSON con los números por
  checkpoint alcanza para Fase 7. Si después hace falta un gráfico,
  agregar al dashboard existente (`scripts/generate_dashboard.py`) en
  vez de crear un visualizador aparte.

**Criterio de salida:** el script corre contra un checkpoint cualquiera
(puede probarse contra `checkpoints/v1/best.pt` como smoke test, aunque
V1 no tiene componente Squim — la correlación ahí es solo real-PESQ
consigo mismo évidentemente trivial, sirve para confirmar que el script
no rompe) y produce el JSON esperado.

---

## Fase 5 — `training/config.py` y `trainer.py` `__main__`

```python
CONFIG_V4 = {
    "train_dir": PROJECT_ROOT / "data" / "processed" / "train",
    "val_dir": PROJECT_ROOT / "data" / "processed" / "val",
    "variant": "V4",
    "loss": "mse_plus_squim",
    "squim_alpha": <ganador de Fase 3>,
    "squim_scale": <ganador de Fase 3>,
    "n_epochs": 30,          # igual que V1, salvo que Fase 3 diga lo contrario
    "batch_size": 4,
    "lr": 2e-4,
    "scheduler_step": 2,
    "scheduler_gamma": 0.98,
    "seed": 42,
    "train_shuffle": True,
    "val_shuffle": False,
    "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "v4",
    "save_every_n_epochs": 5,
    "description": "Rama paralela a V2: V1 + término perceptual Squim "
                    "(proxy diferenciable de PESQ), alpha=<X>. "
                    "Ver docs/decisions.md 30/08/2026 y docs/PLAN_V4.md.",
}
```

- Agregar `"V4": CONFIG_V4` al dict `configs` en el `__main__` de
  `trainer.py`, junto al import.
- Si Fase 3 concluyó que hace falta warm-start, decidir cómo
  implementarlo: lo más simple es una config separada
  `CONFIG_V4_WARMUP` corta (2-3 épocas, `loss: "mse_magnitude"`) seguida
  de `CONFIG_V4` con `init_checkpoint` apuntando al resultado del
  warmup — mismo mecanismo que ya existe (`init_checkpoint` en
  `trainer.py`), no hace falta código nuevo para esto.

**Criterio de salida:** `python -m training.trainer --config V4`
arranca sin errores de config (puede cancelarse a los pocos segundos,
solo se está verificando que levanta).

---

## Fase 6 — Entrenamiento completo

```bash
setsid nohup python -m training.trainer --config V4 > logs/v4_training.log 2>&1 < /dev/null &
```

(patrón obligatorio para jobs largos desde el 19/08/2026 — `nohup`
solo no sobrevive el cierre de la terminal, ver nota operativa en
`CLAUDE.md`).

- Duración esperada: similar a V1 (~16h) más el overhead del forward de
  Squim por batch — puede ser sensiblemente más lento, no asumir que da
  igual. Si el mini-sweep de Fase 3 midió el tiempo/época, extrapolar
  desde ahí en vez de adivinar.
- Mientras corre, no hace falta intervención — el chequeo real pasa en
  Fase 7, sobre los checkpoints guardados.

**Criterio de salida:** entrenamiento terminado, `checkpoints/v4/best.pt`
y los `epoch_XX.pt` intermedios presentes, `history.json` completo.

---

## Fase 7 — Evaluación y chequeo cruzado de gaming

1. **Evaluación estándar** (idéntica a V1/V2/V3):
   ```bash
   python -m evaluation.evaluate_variant --variant v4
   ```
   sobre `test_v1_en` (V4 es rama EN, no toca español — eso es V5).
2. **Correlación** (`monitor_correlation.py`, Fase 4) sobre cada
   `epoch_XX.pt` disponible más `best.pt` — ver si la correlación
   Squim-vs-PESQ real se mantiene estable o se degrada a medida que
   avanza el entrenamiento.
3. **Chequeo cruzado explícito** (el criterio central, no un anexo):
   comparar, checkpoint por checkpoint, el componente Squim de
   `history.json` contra SI-SDR y STOI de `evaluate_variant.py`. Señal
   de alarma tal como la documentó PESQetarian: el componente Squim
   mejorando mientras SI-SDR o STOI caen por debajo de V1 o de Noisy.
4. Comparar resultado final contra **V1 y V2** (no solo contra Noisy) —
   es lo que pide la nota de cierre de V2 en `EXPERIMENTS.md`: si V4
   aporta sobre V2, demuestra valor perceptual adicional al SI-SDR.

**Criterio de salida:** tabla de resultados V1 vs V2 vs V4 sobre
`test_v1_en`, con veredicto explícito sobre el chequeo cruzado (pasó /
no pasó, y por qué).

---

## Fase 8 — Escucha dirigida (no una escucha genérica)

```bash
python -m scripts.generate_listening_samples --lang en --variant v4
```
(reutiliza el script ya extendido, no hace falta código nuevo).

- Escuchar 5-10 archivos, uno por bucket de SNR si es posible.
- Buscar específicamente, no en general:
  - **Artefactos agudos/metálicos de alta frecuencia** — firma reportada
    del modelo PESQetarian puro (de Oliveira et al. 2024, sección 4.1).
  - **Clicks o saltos abruptos de rango dinámico al inicio del
    archivo** — firma del exploit de nivel del modelo PESQ-SDR del
    mismo paper (sección 4.2).
- Si aparece cualquiera de las dos firmas, es evidencia de gaming
  aunque el chequeo cruzado de Fase 7 no lo haya detectado — las dos
  validaciones son complementarias, no redundantes.

**Criterio de salida:** veredicto anotado (con o sin firmas de gaming)
antes de declarar V4 cerrado.

---

## Fase 9 — Documentación de cierre

- `docs/EXPERIMENTS.md`: nueva sección "V4 — CRN + MSE + Squim", mismo
  formato que V1/V2/V3 (configuración, curva de entrenamiento,
  resultados globales, análisis por bucket SNR, conclusión). Incluir
  explícitamente el resultado del chequeo cruzado y de la escucha
  dirigida como sub-secciones — no como nota al pie.
- `docs/decisions.md`: si el mini-sweep de Fase 3 o el entrenamiento
  final revelaron algo no anticipado (ej. inestabilidad, un `alpha`
  ganador sorprendente, mismatch de escala de Fase 0), documentarlo con
  el mismo criterio que las entradas existentes.
- `CLAUDE.md`: actualizar "Estado actual del proyecto" (checkboxes de
  V4 completado) y "Hallazgos técnicos clave" si corresponde.
- Regenerar dashboard: `python -m scripts.generate_dashboard`.
- Commit + tag `v4.0.0`, siguiendo el git workflow ya establecido en
  `CLAUDE.md`.

---

## Riesgos conocidos y mitigación

| Riesgo | Mitigación | Fase |
|---|---|---|
| Mismatch de escala entre audio normalizado del proyecto y lo que espera Squim | Smoke test manual con batch real antes de tocar `trainer.py` | 0 |
| Inestabilidad en época 1 (CRN random → audio basura → Squim) | Mini-sweep con/sin warm-start MSE | 3 |
| Gaming del proxy (PESQ/Squim↑, todo lo demás↓) | Chequeo cruzado por checkpoint, no solo al final | 2, 6, 7 |
| Proxy pierde fidelidad específicamente en audio "enhanced" | `monitor_correlation.py` mide sobre salida del modelo, no solo noisy/clean | 4, 7 |
| Artefactos audibles no capturados por ninguna métrica objetiva | Escucha dirigida con firmas concretas a buscar | 8 |
| `cudnn.flags(enabled=False)` alrededor de todo el step en vez de solo Squim, rompiendo `cudnn_deterministic` del CRN | Envolver únicamente la llamada a `self.squim_model(...)` | 2 |
| Tiempo de entrenamiento subestimado (overhead de Squim por batch no medido) | Extrapolar desde el tiempo/época del sweep de Fase 3, no asumir paridad con V1 | 3, 6 |

## Preguntas abiertas explícitas (no asumir, verificar)

- ¿`SQUIM_OBJECTIVE` funciona sin degradación con el nivel de
  normalización RMS+peak del proyecto, o hace falta un ajuste de escala
  antes de alimentarlo? (Fase 0)
- ¿Hace falta warm-start con MSE puro, o el término Squim es estable
  desde época 1 con `alpha` alto? (Fase 3, se decide con datos, no a
  priori)
- ¿Cuánto overhead real agrega el forward de Squim por batch? (Fase 3)

## Referencias usadas en este plan (todas leídas completas, en `Papers V4/`)

- de Oliveira, D., Welker, S., Richter, J., Gerkmann, T. (2024). *The
  PESQetarian: On the Relevance of Goodhart's Law for Speech
  Enhancement*. Interspeech 2024.
- Martín-Doñas, J.M., Gomez, A.M., Gonzalez, J.A., Peinado, A.M. (2018).
  *A Deep Learning Loss Function Based on the Perceptual Evaluation of
  the Speech Quality*. IEEE Signal Processing Letters, 25(11).
- López-Espejo, I., Edraki, A., Chan, W.-Y., Tan, Z.-H., Jensen, J.
  (2023). *On the deficiency of intelligibility metrics as proxies for
  subjective intelligibility*. Speech Communication, 150, 9–22.
- Fu, S.-W., Tsao, Y., Hwang, H.-T., Wang, H.-M. (2018). *Quality-Net:
  An End-to-End Non-intrusive Speech Quality Assessment Model based on
  BLSTM*.
