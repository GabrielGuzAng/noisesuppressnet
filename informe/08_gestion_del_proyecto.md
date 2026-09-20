# Capítulo 8 — Gestión del proyecto

**Extensión estimada**: 14 pp · **Estado**: BLOQUEADO
**Fuentes primarias**: `DocsProfesores/` (anteproyecto, EDT, Gantt, riesgos,
calidad, gestión de tiempos, FODA), `DESVIACIONES.md`,
`analysis/training_cost_report.py`, `results/*_training_cost.json`.

Capítulo obligatorio en un informe UTN e inexistente en un paper. Se escribe
**contra lo ya entregado y defendido**, no de cero: el informe tiene que ser
coherente con lo presentado el 30/06/2026, y donde el proyecto se desvió hay que
decir dónde y por qué.

---

## Bloqueo: falta la línea base contra la cual medir el desvío

El anteproyecto (§3.3) cita un documento **"EDT v3 — NoiseSuppressNet"** con 65
paquetes de trabajo en 8 ramas. En `DocsProfesores/` sólo existe
`EDT_v2_NoiseSuppressNet.docx`. Tampoco está la presentación de la defensa del
preproyecto, que `CLAUDE.md` da por entregada.

**Sin resolver esto el capítulo no arranca**, y no lo destraban los datos de
costo: el costo es un insumo del §8.4, mientras que el bloqueo es sobre §8.1 y
§8.2 —la EDT y el cronograma planificado contra el real—. Son cosas distintas y
el capítulo necesita las dos. Dos salidas:

1. Aparece el v3 en otro lado y se escribe contra él.
2. Se declara que la v2 es la versión vigente y que el anteproyecto la cita mal.
   Es defendible, pero hay que decirlo, no disimularlo.

Pendiente aparte, menor: dos documentos en `DocsProfesores/` que ninguna fuente
del proyecto menciona —`Gestion_Tiempos.docx` y
`NoiseSuppressNet_Descripcion_FODA (1).docx`—. Hay que ver qué son y si entran al
cuerpo o a anexos.

---

## 8.1 Estructura de desglose del trabajo

- **Afirma**: 8 ramas, 65 paquetes de trabajo, con el avance real contra el
  declarado en el anteproyecto.
- **Fuente comprometida**: anteproyecto §3.3, tabla de ramas con avance
  (1: 5/11, 2: 4/8, 3: 5/9, 4: 1/8, 5: 4/10, 6: 0/8, 7: 1/4, 8: 2/7 al 30/06).
- **Falta**: el archivo del EDT v3 y el recuento de avance al cierre. La rama 4
  (entrenamiento V0–V5) estaba en 1/8 y hoy está sobrecumplida; la rama 6
  (análisis de resultados) estaba en 0/8 y es donde vive el reanálisis.

## 8.2 Cronograma: planificado contra real

- **Afirma**: el cronograma se sostuvo con desvíos de días, no de semanas, y dos
  hitos se adelantaron. El desvío relevante no es de tiempo sino de contenido.
- **Fuente**: `DESVIACIONES.md` D9, tabla de hitos H1 a H8.
- **El punto a desarrollar**: H7 —"V5 congelada, propuesta completa (español +
  PESQNet) ★"— se congeló seis días antes de fecha, pero con una V5 que **no es
  la que el hito define**, porque el eje PESQNet se cerró como negativo en H6.
  Cumplir la fecha y no el contenido es un tipo de desvío que hay que nombrar.
- **Falta**: verificar la fecha real de H2 (pipeline listo, 22/07).

## 8.3 Riesgos que se materializaron

- **Fuente comprometida**: `DocsProfesores/Riesgos_NoiseSuppressNet.docx`.
- **Falta**: leer la matriz y cruzar cada riesgo con lo que efectivamente pasó.
  Los incidentes documentados hasta hoy son:

| Incidente | Fecha | Impacto | Mitigación |
|---|---|---|---|
| Dos cortes de energía | 16/09/2026 | Corridas de V7 interrumpidas | Reanudación bit-exacta en `Trainer` (`--resume-from`), verificada por `tests/test_resume.py` con cada entrenamiento en su propio proceso. **Salvó 10,5 h de GPU.** |
| Job largo muerto al cerrar la terminal | 18/08/2026 | Generación de mixturas ES perdida | `setsid nohup … < /dev/null &` como patrón estándar desde el 19/08 |
| PID equivocado con `$!` en ese patrón | 30/08/2026 | Se persiguió un OOM inexistente durante el sweep de V4 | Verificación por `pgrep -af <módulo>` y `nvidia-smi`, nunca por el PID guardado |
| Espacio en disco para Common Voice ES | 08/2026 | Bloqueante | Dataset y mixturas en `/mnt/Datos` por symlink |
| Bug de reproducibilidad en `make_mixtures.py` | 09/2026 | Orden no determinista de `rglob()` | `sorted()` y `random.seed(42)` al inicio |
| Bug de reproducibilidad en `trainer.py` | 08/2026 | V1/V2/V3 no bit-exactos | `cudnn.deterministic` y `np.random.seed()`; **no se corrigió retroactivamente** |

- **Afirma**: el incidente del 16/09 es el mejor caso de gestión de riesgo del
  proyecto — un riesgo se materializó, la mitigación se construyó en respuesta, se
  verificó con un test y el ahorro está cuantificado.
- **Falta**: confirmar que los seis incidentes están en la matriz original o
  declarar cuáles no estaban previstos.

## 8.4 Recursos consumidos

- **Afirma**: horas de GPU, kWh y costo equivalente en nube, por variante y total.
- **Fuente**: `analysis/training_cost_report.py`; `results/*_training_cost.json`.
- **Disponible** (20/09/2026): 20 corridas medidas, **150,69 h de GPU,
  26,370 kWh, 7,91 kg CO₂, USD 52,74** de equivalente en nube. Desglose completo
  en `FUENTES.md`.
- **La línea de proxy perceptual se reporta como la suma de sus 8 brazos**:
  22,60 h, 3,955 kWh, USD 7,91. V4 y V4b nunca fueron una corrida única, y ese es
  el precio de haber cerrado la línea con diseño experimental —sweep de α más
  2×2 con placebo— en vez de con una sola corrida. Es el número que hay que poner
  al lado del resultado negativo del capítulo 6.
- **Falta**: V0, los 5 brazos del sweep de lr de V3b y las 4 corridas smoke.
- **Decisión pendiente**: si los sweeps y los smoke entran en el total. La
  recomendación es reportar dos totales — "corridas reportadas" y "total de
  proyecto"— y explicar la diferencia.
- **La eficiencia útil se reporta sólo para las 8 corridas seleccionadas por
  `best.pt`**, donde mide entrenamiento gastado después del checkpoint que
  efectivamente se usó: va de 48,0 % (V5 s43) a 95,0 % (V2). En V6 y V7 **no se
  reporta**, porque el estimando es la trayectoria promediada y las épocas
  posteriores al mínimo de `val_loss` no son desperdicio sino el dato; en V4 y
  V4b tampoco, porque son corridas de 3 épocas evaluadas época por época. Ver
  `FUENTES.md`, "Sobre la métrica de eficiencia útil".
- **Supuestos a declarar**: 105 W de GPU + 70 W de sistema = 175 W;
  0,3 kg CO₂/kWh (Cammesa, mix argentino 2024); USD 0,35/h (GCP T4).
- **Limitación a declarar**: el consumo es estimado a partir de potencia nominal
  sostenida, no medido con instrumento. El propio reporte acota el error en ±15 %.

## 8.5 Gestión de la calidad

- **Fuente comprometida**: `DocsProfesores/Plan_Gestion_Calidad_NoiseSuppressNet.docx`
  y anteproyecto §6.
- **Afirma**: los cuatro procedimientos de control declarados —test set sellado,
  ablation formal, reglas de aceptación previas a cada entrenamiento, validación
  externa— se ejecutaron. Con dos observaciones que van dichas:
  - El sellado se cumplió y se amplió (tres conjuntos, no uno), pero con 250 pares
    y no los 300 comprometidos (D4).
  - El ablation formal se violó una vez, en V3b, con dos variables cambiadas a la
    vez. Detectado por el autor y corregido en V3e (§4.8).
- **Afirma sobre el ciclo PDCA**: el plan declara Plan-Do-Check-Act por variante,
  donde *Act* decide incorporar el cambio o descartarlo documentando el negativo.
  El capítulo 6 es la evidencia de que el *Act* se ejecutó cuatro veces en la
  dirección de descartar.
- **Falta**: el KPI propio cambió de definición (D2) y eso toca el plan de
  calidad. Hay que decirlo acá, no sólo en el capítulo 1.

## 8.6 Herramientas de seguimiento desarrolladas

- **Afirma**: el proyecto construyó su propio instrumental de seguimiento —
  `scripts/generate_dashboard.py` produce `docs/dashboard.html`, un único archivo
  autocontenido sin servidor ni dependencias externas, con tres pestañas (una por
  sellado), una ficha por experimento de V0 a V7 con los negativos al mismo nivel
  de detalle que los positivos, línea de tiempo y wiki de métricas.
- **Fuente**: `scripts/generate_dashboard.py`.
- **Detalle que conviene mencionar**: la paleta está validada para daltonismo en
  claro y oscuro.
- **Falta**: nada. Va con una captura como figura.

## 8.7 Relación con la práctica profesional

- **Afirma**: el proyecto es independiente de VoxHub (CyT Comunicaciones). No
  comparte código, datos, infraestructura ni horas. La relación es de motivación
  y de eventual aplicación futura, no de ejecución compartida.
- **Falta**: nada. Un párrafo. Se retoma del capítulo 1 sin repetirlo.
