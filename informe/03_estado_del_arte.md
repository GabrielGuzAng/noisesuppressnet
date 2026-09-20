# Capítulo 3 — Estado del arte y justificación de la arquitectura

**Extensión estimada**: 10 pp · **Estado**: redactable
**Fuentes primarias**: `Papers/`, `PapersAporteFinal/`, `Papers V4/`,
bibliografía central de `CLAUDE.md`, `decisions.md`.

El capítulo tiene que terminar respondiendo una sola pregunta: **por qué CRN y no
otra cosa**, en 2026, sabiendo que hay arquitecturas más nuevas.

---

## 3.1 Panorama de arquitecturas para speech enhancement causal

- **Afirma**: mapa de familias — máscaras espectrales, modelos en dominio
  temporal, arquitecturas de dos etapas por banda.
- **Fuente**: Tan & Wang 2018; Schröter et al. 2022 (DeepFilterNet2);
  Valin 2018 (RNNoise); Reddy et al. 2020 (DNS Challenge).
- **Falta**: inventariar `Papers/` y `PapersAporteFinal/` para no citar de
  memoria. Es trabajo mecánico pero hay que hacerlo antes de redactar.

## 3.2 Por qué CRN

- **Afirma**: CRN es el punto de equilibrio entre causalidad garantizada por
  construcción, volumen de datos alcanzable con el hardware disponible, y
  reproducibilidad verificable contra un paper con implementación de referencia.
- **Fuente**: `EXPERIMENTS.md` §V0, "Decisiones técnicas tomadas y documentadas".
- **Afirma además**: la elección se validó externamente. Se consultó al
  Dr. DeLiang Wang, co-autor del paper original, y la respuesta llegó el
  12/06/2026 confirmando arquitectura y repositorio de referencia.
- **Fuente**: anteproyecto §6.4, "Validación externa"; `EXPERIMENTS.md` §V0,
  "Validación externa" y "Auditoría automatizada del repositorio de referencia".
- **Falta**: nada. Este es un punto fuerte del proyecto y conviene que tenga su
  propio apartado, no una nota al pie.

## 3.3 El hueco en la literatura: idioma frente a canal

- **Afirma**: la literatura de speech enhancement está sesgada al inglés, y el
  trabajo que más directamente objeta la lectura ingenua de ese sesgo es
  Wang et al. 2022 (*Disentangling the Impacts of Language and Channel
  Variability*, Interspeech): lo que parece efecto de idioma puede ser efecto de
  canal de grabación.
- **Afirma, y es la respuesta del proyecto**: por eso se sellaron **dos** test
  sets en español con canales distintos. `test_v2_es` es Common Voice
  —micrófono heterogéneo—; `test_v3_mls_es` es MLS —audiolibro, el mismo canal
  que LibriSpeech— con las condiciones de ruido de `test_v1_en` copiadas
  verbatim. La diferencia entre los dos aísla el canal.
- **Fuente**: `decisions.md`, "El control preregistrado idioma/canal" (07/09);
  `reanalisis_estadistico.md` F6; `docs/preregistro_mls_es.sha256`.
- **Falta**: nada. Esta subsección es la que convierte una objeción previsible en
  un diseño experimental, y conviene que se note.

## 3.4 Justificación de la loss perceptual, y su desenlace

- **Afirma**: la motivación para optimizar contra un proxy diferenciable de PESQ
  es sólida y está publicada (Xu et al. 2022; Fu et al., citado en Xu). El riesgo
  de gaming también está publicado en el mismo lugar.
- **Fuente**: Xu et al. 2022; Kumar et al. 2023.
- **Falta**: nada acá. El desenlace se ejecuta y se reporta en el capítulo 6; este
  apartado sólo instala la hipótesis y su riesgo conocido, para que el lector
  llegue al 6 sabiendo que el riesgo estaba identificado de antemano.
- **Nota de redacción**: no anticipar el resultado. El capítulo 3 plantea, el 6
  resuelve.

## 3.5 Posicionamiento del trabajo

- **Afirma**: cuatro afirmaciones de aporte, a contrastar en el capítulo 9.
  (1) El ablation sistemático V0–V7 con una variable por vez. (2) La adaptación al
  español separando idioma de canal. (3) La caracterización con datos propios del
  gaming de un proxy perceptual frozen. (4) La compuerta de paso directo, en
  estado de screening.
- **Falta**: la (4) depende de la confirmación de V7. Si no se corre, la
  afirmación se enuncia como hallazgo preliminar con su reserva, no como aporte.
