# Capítulo 6 — Resultados negativos y qué se aprendió

**Extensión estimada**: 14 pp · **Estado**: redactable
**Fuentes primarias**: `decisions.md` (entradas "Cierre de V4" 31/08, "V4b" 01/09,
"Corrección: el 67 %…" 09/09, 06/09 sobre GCRN), `docs/v6_compuerta.md`,
`docs/proxy_enganado.html`.

Capítulo propio, no anexo. Un tribunal de electrónica evalúa criterio, y el
criterio se demuestra en lo que se decidió **no** seguir haciendo y por qué.

**Tono**: se dice qué se probó, con qué diseño, qué dio y qué se hizo con eso. Sin
disculpas, sin dramatismo y sin convertir el fracaso en épica.

---

## 6.1 Por qué este capítulo existe

- **Afirma**: el proyecto financió con horas de GPU cuatro líneas que no
  prosperaron, y las cerró con diseño experimental en vez de abandonarlas. El
  costo de cada cierre está medido.
- **Falta**: el total de horas de GPU gastadas en líneas cerradas. Requiere los
  reportes de costo faltantes (ver `FUENTES.md`).

## 6.2 El proxy perceptual: Goodhart con datos propios

El resultado negativo más importante del proyecto, porque **es uno de los dos ejes
de innovación comprometidos en el anteproyecto** (D1).

### 6.2.1 V4 — el régimen inestable

- **Afirma**: Squim congelado se gamea desde la época 1, en las cuatro corridas.
  Firma: `pesq_hat` trepa hacia su techo estructural de 4,64 mientras el PESQ real
  medido cae por debajo del audio ruidoso. El control con MSE puro gana en las
  cuatro métricas reales.
- **Fuente**: `decisions.md`, "Cierre de V4" (31/08/2026); tag `v4.0.0`.
- **Afirma además**: es el patrón exacto que Xu 2022 reporta citando a Fu et al.
  — *"estimated PESQ scores increase while true PESQ scores decrease"*. El riesgo
  estaba identificado antes de correr nada; lo que el proyecto aporta es la
  medición propia de su magnitud.

### 6.2.2 V4b — el protocolo estabilizado

- **Afirma**: se probó el protocolo de Xu menos la alternancia — arranque desde V1
  convergido más lr 2e-5. Diseño 2×2 con placebo. Recupera **~1/3** del daño.
  El término Squim sigue costando **−0,598 PESQ-NB** contra el placebo con
  idéntico protocolo, frente a −0,887 en el régimen inestable.
- **Fuente**: `decisions.md`, "V4b" (01/09/2026); tag `v4b.0.0`;
  `results/v4b/`.
- **Conclusión atribuible**: lo que hace funcionar el esquema de Xu es
  específicamente **reentrenar el proxy**, no el arranque ni el learning rate.
  Dos tercios del gaming sobreviven al protocolo estabilizado.

### 6.2.3 El placebo era imprescindible

- **Afirma**: contra el audio ruidoso, V4b-main da −0,059 y parece casi arreglado.
  Contra el placebo con idéntico protocolo (+0,539), el término cuesta −0,598.
  **Sin ese control la conclusión hubiera sido la opuesta.**
- **Fuente**: ídem.
- **Por qué va acá y no en el capítulo 4**: el capítulo 4 enuncia el protocolo;
  este es el caso que lo justifica. Se cruzan con referencia mutua.

### 6.2.4 El diagnóstico barato

- **Afirma**: la brecha `pesq_hat − PESQ-WB real` detecta el gaming sin necesidad
  de correr el entrenamiento completo. MAE nominal de Squim para WB-PESQ: 0,142
  (Kumar 2023, Tabla 2). Brechas observadas: +2,36 y +2,77 — entre 16× y 20× ese
  error.
- **Advertencia de implementación**: se compara contra PESQ-**WB**, nunca NB. La
  cabeza de Squim estima WB-PESQ, acotada en [1, 4,64] por sigmoide.
- **Fuente**: `decisions.md`, entradas de V4 y V4b.

### 6.2.5 El sesgo de selección que el gaming introduce

- **Afirma**: `Trainer` elige `best.pt` por `val_loss` mínima. Con
  `mse_plus_squim`, `val_loss` incluye `−pesq_hat`, así que **más gameado da
  "mejor" checkpoint**. Con esas losses hay que usar `save_every_n_epochs=1` y
  evaluar época por época.
- **Fuente**: ídem.
- **Por qué importa más allá de V4**: es el mismo problema de selección que
  aparece en §4.7 por otra vía. Conviene tratarlos juntos en las conclusiones.

### 6.2.6 Qué se decidió

- **Afirma**: la línea de proxy perceptual queda **cerrada**. El espacio de
  decisión quedó cubierto —régimen inestable y régimen estabilizado— y no se
  corren más variantes. Un proxy propio reentrenado queda como trabajo futuro.
- **Falta**: nada.

## 6.3 V3b: la hipótesis del learning rate conservador

- **Afirma**: redujo el olvido (−0,029 contra −0,079 de V3) pero también la
  ganancia (+0,153 contra +0,230). La hipótesis de partida no se sostuvo.
- **Causa**: confusión de variables en el propio diseño — 10 épocas contra 30.
- **Qué se aprendió**: V3e se diseñó para desconfundirlas, y funcionó. Es un
  negativo que produjo el mejor checkpoint del proyecto.
- **Fuente**: `results/v3b_*.json`; `decisions.md`, línea V3.

## 6.4 V6: el endpoint primario que falló

- **Afirma**: +2,5 pp contra un umbral preregistrado de +3 pp, p = 0,181. Por la
  regla de decisión, la hipótesis no queda sostenida — aunque el mecanismo
  propuesto sí se verifique (ρ ≈ −0,4, p < 1e−8).
- **Afirma además, y es lo que hace creíble al preregistro**: P3 falló **por un
  error de signo del autor** en el preregistro. Se reportó el fallo según lo
  escrito en vez de corregir el signo a posteriori.
- **Fuente**: `v6_compuerta.md` §1, §5, §6.
- **Qué se aprendió**: el piso de ruido por checkpoint (§4.6), que reescribió el
  estimando primario de todo el proyecto. Vale más que el veredicto.

## 6.5 El número que se retractó

- **Afirma**: el "67 % de retención de la ganancia entre canales" publicado en el
  commit `f09e98f` **no se sostiene**: el estimando no era el mismo para las dos
  recetas comparadas. Con V2 evaluado sobre `test_v3_mls_es`, una mejora agnóstica
  al idioma —la loss combinada— transfiere al 132 %, y los fine-tunings al español
  transfieren al 34–38 %.
- **Fuente**: `decisions.md`, 09/09/2026.
- **Por qué va en el cuerpo del informe**: retractar un número propio, publicado y
  commiteado, en el mismo documento donde se lo publicó, es exactamente el
  criterio que un tribunal evalúa. Se escribe sin atenuantes.

## 6.6 GCRN: descartado antes de gastar horas

- **Afirma**: se evaluó GCRN (Tan & Wang 2020), con paper y repositorio oficial
  leídos, y se descartó: es una mejora de arquitectura, no del eje cross-lingual,
  y competía por las mismas horas contra el control de canal y el downstream ASR.
  No alimenta ninguna de las cuatro afirmaciones del aporte.
- **Fuente**: `decisions.md`, 06/09/2026; `docs/PLAN_GCRN.md`.
- **Por qué va en este capítulo**: un descarte fundamentado antes de gastar el
  recurso es una decisión de gestión de alcance, y es tan evaluable como un
  experimento corrido. Se cruza con el capítulo 8.

## 6.7 Síntesis: cuatro lecciones transferibles

- **Afirma**: (1) un control con idéntico protocolo, no una variante anterior;
  (2) el estimando es la trayectoria, no el checkpoint; (3) la métrica de
  selección tiene que estar alineada con la de reporte, y con un proxy en la loss
  está activamente desalineada; (4) el preregistro sólo sirve si se cumple cuando
  no conviene; (5) una afirmación de reproducibilidad hay que medirla, no
  deducirla del código — la del evaluador no se sostuvo, y se descubrió recién
  cuando alguien la puso a prueba (§4.1.1).
- **Falta**: nada. Este apartado es el que el tribunal va a citar.
