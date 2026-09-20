# Capítulo 1 — Introducción y objetivos

**Extensión estimada**: 8 pp · **Estado**: redactable
**Fuentes primarias**: `DocsProfesores/Anteproyecto Gabriel Guzmán.pdf` (§1, §2, §3.1, §6.2),
`DESVIACIONES.md`, `CLAUDE.md`.

El capítulo tiene una función que el paper no tiene: decirle al tribunal, antes
de que lo pregunte, qué se prometió, qué se entregó y dónde no coinciden.

---

## 1.1 Planteo del problema

- **Afirma**: los métodos clásicos de supresión de ruido (Wiener, sustracción
  espectral) alcanzaron su techo frente a ruido no estacionario, y la literatura
  de deep learning que los superó está sesgada al inglés.
- **Fuente**: anteproyecto §1.4. El dato del sesgo ("más del 85 % de los trabajos
  evaluados corresponden a LibriSpeech, WSJ0 o DNS Challenge") viene del
  anteproyecto y **hay que reverificarlo antes de reescribirlo**: es una
  afirmación cuantitativa sobre la literatura y no consta cómo se midió.
- **Falta**: o se recupera la metodología del conteo, o se reformula
  cualitativamente. Un tribunal puede pedir la lista.

## 1.2 Objetivo general

- **Afirma**: desarrollar un sistema causal de supresión de ruido en habla,
  16 kHz mono, con RTF < 1,0 en CPU Intel i5-4460 monohilo, adaptado al español.
- **Fuente**: anteproyecto §1.1 y §2.1.
- **Falta**: nada.

## 1.3 Objetivos específicos y estado de cada uno

Tabla de cierre, una fila por objetivo de calidad del producto, con lo medido al
lado del criterio comprometido.

| Obj. | Criterio mínimo | Aspiracional | Medido | Estado |
|---|---|---|---|---|
| OP-1 PESQ | ≥ 2,5 | ≥ 3,0 | 2,849 (V2, EN) · 2,686 (V5, ES) | mínimo cumplido, aspiracional no |
| OP-2 STOI | ≥ 0,85 | ≥ 0,90 | 0,908 (V2, EN) · 0,896 (V5, ES) | mínimo cumplido; aspiracional sí en EN, no en ES |
| OP-3 SI-SDR | ≥ 8 dB | ≥ 12 dB | 14,44 dB (V2, EN) · 14,27 dB (V5, ES) | ambos cumplidos |
| OP-4 RTF | < 1,0 | < 0,5 | 0,331 mediana, 0,340 p95 | ambos cumplidos |
| OP-5 Causalidad | diff = 0,00e+00 | bit-exact | bit-exact a nivel de frame, **+10 ms de lookahead por `center=True`** | cumplido con corrección de enunciado |
| OP-6 vs Butterworth | PESQ y STOI > Butter LP | en todas las categorías de ruido | **no medido sobre sellado** | abierto |

- **Fuente**: criterios del anteproyecto §6.2; valores de `FUENTES.md`.
- **Falta**: OP-6. La única comparación contra el pasabajo es
  `results/summary_metrics.csv`, del 27/06/2026, un mes anterior al sellado de
  `test_v1_en`, y ahí V0 pierde contra el Butterworth (1,417 contra 2,052).
  Correr `baselines/butterworth.py` sobre los tres sellados cierra el objetivo.

## 1.4 Alcance: qué se hizo y qué no

- **Afirma**: el alcance comprometido se cumplió salvo en tres puntos, y los tres
  se declaran acá y se justifican en el capítulo 8.
- **Fuente**: anteproyecto §3.1; `DESVIACIONES.md` D7 y D8.
- **Falta**: la decisión sobre DNSMOS/SegSNR y sobre el benchmark contra RNNoise
  y DeepFilterNet2. Hasta que esté tomada, este apartado no se puede cerrar.

## 1.5 Aclaración: la propuesta central cambió de contenido

- **Afirma**: el anteproyecto define dos ejes de innovación. El primero
  —adaptación lingüística— se sostuvo y es el núcleo del trabajo. El segundo
  —pérdida perceptual diferenciable— **no se sostuvo empíricamente**, y su cierre
  con datos propios es en sí mismo un resultado del proyecto. En consecuencia, V5
  conserva el nombre del hito H7 pero no su contenido: es V2 fine-tuneado a
  español, sin término perceptual.
- **Fuente**: `DESVIACIONES.md` D1; `decisions.md`, entradas "Cierre de V4"
  (31/08) y "V4b" (01/09).
- **Falta**: nada. Va en la primera página del capítulo, no escondido en el 6.
- **Nota de redacción**: sin disculpas y sin dramatismo. Se enuncia qué se probó,
  con qué diseño, qué dio, y qué se hizo con eso.

## 1.6 Relación con la práctica profesional

- **Afirma**: NoiseSuppressNet nace del interés de contar con un modelo propio de
  speech enhancement, sin dependencia de APIs externas, en el marco del trabajo
  del autor como Tech Lead en CyT Comunicaciones. **Es un proyecto independiente**:
  no comparte código, datos, infraestructura ni horas con VoxHub.
- **Fuente**: `CLAUDE.md`, sección de contexto del autor.
- **Falta**: nada. Un párrafo, sin extenderse. El tribunal valora que la relación
  esté declarada y delimitada; no valora que el informe hable de VoxHub.

## 1.7 Organización del documento

- **Afirma**: mapa de capítulos.
- **Falta**: se escribe último.
