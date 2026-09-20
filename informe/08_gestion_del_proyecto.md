# Capítulo 8 — Gestión del proyecto

**Extensión estimada**: 14 pp · **Estado**: parcial
**Fuentes primarias**: `DocsProfesores/` (anteproyecto, EDT, Gantt, riesgos,
calidad, gestión de tiempos, FODA), `DESVIACIONES.md`,
`analysis/training_cost_report.py`, `results/*_training_cost.json`.

Capítulo obligatorio en un informe UTN e inexistente en un paper. Se escribe
**contra lo ya entregado y defendido**, no de cero: el informe tiene que ser
coherente con lo presentado el 30/06/2026, y donde el proyecto se desvió hay que
decir dónde y por qué.

---

## Pendiente de clasificar

Dos documentos en `DocsProfesores/` que ninguna fuente del proyecto menciona:
`Gestion_Tiempos.docx` y `NoiseSuppressNet_Descripcion_FODA (1).docx`. Hay que
ver qué son y si entran al cuerpo o a anexos.

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
- **Disponible hoy** (19/09/2026): doce corridas reportadas, 128,09 h de GPU,
  22,415 kWh, 6,73 kg CO₂, USD 44,83 de equivalente en nube. Tabla completa en
  `FUENTES.md`.
- **Falta**: los 8 brazos de V4 y V4b, los 5 del sweep de lr de V3b, V0 y las
  4 corridas smoke. Ver `FUENTES.md` para la ubicación de cada `history.json`.
- **Decisión pendiente**: si los sweeps y los smoke entran en el total. La
  recomendación es reportar dos totales — "corridas reportadas" y "total de
  proyecto"— y explicar la diferencia.
- **Punto a desarrollar, no sólo a tabular**: la eficiencia útil varía entre
  33,3 % (placebo de V6) y 95,0 % (V2). La dispersión no es ruido: mide cuánto se
  entrenó de más después de la mejor época, y es consecuencia directa del
  problema de selección de checkpoint del §4.7. Las dos corridas de V6 están en
  ~33 % porque `best.pt` cayó en la época 2 de 6.
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
