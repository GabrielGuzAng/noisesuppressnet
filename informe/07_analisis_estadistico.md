# Capítulo 7 — Análisis estadístico

**Extensión estimada**: 12 pp · **Estado**: redactable
**Fuentes primarias**: `analysis/reanalysis_stats.py`,
`results/reanalysis_stats.json`, `docs/reanalisis_estadistico.md`,
`analysis/hueco_bucket_checks.py`, `tests/test_reanalysis_stats.py`.

El análisis está corrido, validado por 358 tests y documentado. Este capítulo lo
traduce a prosa y declara sus límites.

---

## 7.1 Diseño del análisis

- **Afirma**: cinco familias de hipótesis (F1 a F5) más el control preregistrado
  (F6), con estimandos declarados por familia, corrección de Holm **dentro** de
  cada familia, e intervalos de confianza bootstrap BCa.
- **Fuente**: `analysis/reanalysis_stats.py`; `docs/reanalisis_estadistico.md`.
- **Validación**: 358 tests en `tests/test_reanalysis_stats.py`, escritos por
  alguien distinto de quien escribió el análisis.
- **Falta**: nada.

## 7.2 F1 — La interacción SNR × idioma

- **Afirma**: el aporte 1 se enuncia como **interacción, no como cruce de signo**.
  La ganancia de V1 no depende del SNR en inglés (ρ = −0,065, no significativo) y
  decae en español (ρ = −0,262, p_Holm = 8,3e−05). La interacción sobrevive Holm:
  ρ = +0,173, p_Holm = 0,012.
- **Afirma, y es una autocorrección**: el negativo puntual del bucket [15,20]
  **no es sostenible por sí solo**. Con n = 50 por bucket, el test de signo da
  24/50, p = 0,89.
- **Fuente**: `reanalisis_estadistico.md` F1; `results/reanalysis_stats.json`.
- **Nota de redacción**: la lectura anterior —una U invertida— era efecto techo
  (`decisions.md`, 06/09). El rendimiento por bucket es plano en inglés.

## 7.3 F2 y F3 — Adaptación difusa, olvido selectivo

- **Afirma**: en español mejora el 85–90 % de los archivos, con test de signo de
  p ≈ 1e−30 a 1e−41. En inglés la mediana no se mueve y el signo es una moneda,
  pero la media cae por una cola izquierda pesada (asimetría −2,2 a −3,5).
- **Afirma, y es la decisión metodológica del capítulo**: el estimando correcto
  del olvido es **la fracción de archivos que se rompen**, no la media. Da 17,6 %
  (V3), 11,6 % (V3e) y 6,8 % (V3b) — monótono con la agresividad del learning
  rate.
- **Fuente**: `reanalisis_estadistico.md` F2 y F3.
- **Consecuencia**: es lo que retiró el "score compuesto". Sumar una media de
  ganancia con una media de costo mezcla dos estimandos de naturaleza distinta.

## 7.4 F5 — Descartar que la cola sea ruido de medición

La parte más fuerte del capítulo, porque responde a la objeción obvia.

- **Afirma**: contra controles del mismo idioma —V2 y el placebo de V4b— la cola
  es de 0,0 a 0,4 %. Los archivos rotos se repiten entre variantes entre 4 y 7
  veces por encima del azar (p < 1e−6). Sólo 54 archivos de 250 se rompen en
  alguna variante.
- **Afirma el confusor, sin esconderlo**: los archivos que se rompen tienen PESQ
  basal más alto bajo V1 (+0,276, IC95 [+0,077, +0,480]).
- **Afirma cómo se descartó**: **no** por un control marginal. El primer intento
  con n = 10 no tenía potencia y dio un falso negativo. Lo descarta la
  estratificación: dentro del cuartil basal más alto, los controles siguen en
  0,0 % y los tratamientos en 9,5–28,6 %. Y el perfil por cuartil **no es
  monótono**, así que el efecto techo tampoco explica la forma.
- **Afirma cuál control carga el peso**: V2, no el placebo. El placebo es casi
  idéntico a V1 y por eso no constituye un sorteo independiente.
- **Fuente**: `reanalisis_estadistico.md` F5.a a F5.d;
  `analysis/hueco_bucket_checks.py` para los dos recomputos que no salen del JSON.
- **Falta**: nada. Conviene una figura del perfil estratificado por cuartil,
  siete series × cuatro cuartiles.

## 7.5 F6 — El control preregistrado de idioma contra canal

- **Afirma**: predicciones P1 y P2 preregistradas con hash **antes** de bajar el
  dato de MLS. Desenlaces: P1 → **IDIOMA**; P2 → **PARCIAL**.
- **Fuente**: `reanalisis_estadistico.md` F6; `docs/preregistro_mls_es.sha256`;
  `tests/test_f6_channel_control.py`.
- **Afirma sobre retención entre canales**: la mejora agnóstica al idioma
  transfiere al 132 %; los fine-tunings al español, al 34–38 %.
- **Prohibido**: el 67 %. Retractado (`FUENTES.md`).

## 7.6 Límites del análisis

- **Afirma, y va en el cuerpo, no en una nota**: el reanálisis de septiembre es
  **exploratorio**. Las hipótesis salieron de estos mismos datos. La confirmación
  fuera de muestra es `test_v3_mls_es`, con predicciones preregistradas antes de
  que el dato existiera.
- **Afirma también**: la corrección de Holm es **por familia**, no global sobre
  las cinco. F4 se reporta explícitamente sin corrección y etiquetado como
  exploratorio.
- **Falta**: nada. Declarar el límite antes de que lo pregunte el tribunal es el
  criterio del capítulo.
