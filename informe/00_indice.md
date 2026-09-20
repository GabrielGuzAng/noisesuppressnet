# Informe Final — NoiseSuppressNet

Proyecto Final de Grado, Ingeniería Electrónica, UTN FRBA.
Autor: Gabriel Guzmán Anglese (legajo 149741-8).
Docente: Mg. Ing. Sebastián Verrastro. Defensa: 29/12/2026.

**Estado del documento**: esqueleto con índice anotado. Sin prosa de cuerpo.
**Creado**: 19/09/2026.

---

## Cómo se usa este directorio

Un archivo por capítulo. Cada capítulo lista sus secciones, y cada sección declara
tres cosas:

- **Afirma** — qué sostiene la sección. Escrito como afirmación, no como tema.
- **Fuente** — de dónde sale cada número. Si no hay fuente, la sección no se escribe.
- **Falta** — qué hay que medir, correr o decidir antes de poder redactarla.

Dos archivos transversales:

- `FUENTES.md` — tabla de trazabilidad de todo número que va al cuerpo.
- `DESVIACIONES.md` — anteproyecto comprometido contra proyecto ejecutado.
  Alimenta el capítulo 1 (estado de cada objetivo) y el 8 (gestión).

---

## Índice y estado

| # | Capítulo | pp. est. | Estado | Bloqueado por |
|---|---|---|---|---|
| 1 | Introducción y objetivos | 8 | Redactable | — |
| 2 | Marco teórico | 14 | Redactable | — |
| 3 | Estado del arte y justificación de la arquitectura | 10 | Redactable | — |
| 4 | Metodología experimental | 16 | Redactable | — |
| 5 | Desarrollo por variante (V0–V7) | 28 | Parcial | V7 confirmatorio; secciones V3b/V3e/V4/V4b/V5 de `EXPERIMENTS.md` |
| 6 | Resultados negativos y qué se aprendió | 14 | Redactable | — |
| 7 | Análisis estadístico | 12 | Redactable | — |
| 8 | Gestión del proyecto | 14 | Parcial | costos de V4/V4b sin generar (8 brazos) |
| 9 | Conclusiones y trabajo futuro | 6 | Bloqueado | cierre de V7 y de los pendientes de alcance |
| 10 | Anexos | 20 | Parcial | reportes de costo V3–V7 sin generar |
|  | **Total cuerpo + anexos** | **~142** | | |

"Redactable" significa que todo número que la sección necesita ya está medido y
tiene fuente. No significa que esté escrito.

---

## Los cuatro problemas que hay que resolver antes de la entrega

Ordenados por cuánto bloquean. La discrepancia de versiones del EDT queda fuera
de esta lista: la resuelve el autor por su cuenta.

### 1. El baseline Butterworth nunca se evaluó sobre un test set sellado

`results/summary_metrics.csv` es del 27/06/2026; `test_v1_en` se selló el
26/07/2026. La única comparación existente contra el pasabajo es sobre el
conjunto de validación de V0, y ahí V0 **pierde** (PESQ-NB 1,417 contra 2,052 del
Butterworth y 2,050 del ruidoso).

El objetivo de calidad OP-6 del plan de calidad pide superioridad sobre el
baseline pasabajo. Hoy no está demostrado sobre el material sellado.

`evaluation/evaluate_variant.py` ya acepta `--variant butterworth` (19/09/2026):
los baselines sin pesos se aplican como función en lugar de cargarse como
checkpoint. **El pasabajo por default es de fase cero (`sosfiltfilt`), que no es
causal** y por lo tanto no cumple la restricción a la que sí está sujeto el CRN.
Se mantiene así por continuidad con los números de V0 y porque ganarle a un
baseline que ve el futuro es la afirmación más fuerte; `--baseline_causal` da la
comparación pareja. El informe reporta las dos y declara cuál es cuál.

### 2. Faltan métricas que estaban dentro del alcance

El anteproyecto incluye DNSMOS y SegSNR entre las métricas de evaluación.
`evaluation/metrics.py` implementa PESQ-NB, PESQ-WB, STOI y SI-SDR únicamente.
Dos caminos, y hay que elegir uno antes del capítulo 4: implementarlas y
re-evaluar, o declarar la reducción de alcance con su justificación.

### 3. Falta el benchmark contra RNNoise y DeepFilterNet2

Está explícitamente dentro del alcance ("comparativa cuantitativa contra dos
modelos publicados de referencia"). No está corrido. Mismo par de caminos que el
punto anterior, pero este es más caro de omitir: es lo que ubica al trabajo
contra el estado del arte, y un tribunal lo va a preguntar.

### 4. V7 queda a mitad de camino

El screening dio positivo en los cuatro endpoints preregistrados, pero el propio
preregistro exige confirmación con tres semillas antes de escribirlo como aporte.
Esas ~45 h de GPU no están lanzadas y el preregistro de la confirmación no está
cerrado ni hasheado.

Esto define la forma del capítulo 5 y del 9. Si la confirmación no entra, V7 se
escribe como resultado de screening con su reserva explícita, no como aporte
arquitectónico. Las dos versiones son defendibles; lo que no es defendible es
escribirlo como aporte sin la confirmación que el propio preregistro pide.

---

## Reglas de redacción para todos los capítulos

Heredadas del rol `informe_utn` en `ROLES.md`:

- Español rioplatense formal, usted implícito. Sin "vos" en el cuerpo.
- Toda afirmación numérica con su fuente: tabla, JSON de `results/` o log.
- **Números retractados que no vuelven, ni con hedge** (ver `FUENTES.md`, sección
  "Prohibidos"): el 67 % de retención entre canales, el costo en inglés de V6, y
  el −0,032 de V3e (el valor correcto es −0,031).
- Figuras con epígrafe autocontenido.
- Nada de "se logró exitosamente": se dice qué se midió y cuánto dio.
- El "score compuesto" que sumaba ganancia en español y costo en inglés está
  retirado desde el reanálisis de septiembre. No se usa para ordenar variantes.
