# Capítulo 9 — Conclusiones y trabajo futuro

**Extensión estimada**: 6 pp · **Estado**: BLOQUEADO
**Depende de**: cierre de V7 (§5.10), decisión sobre los pendientes de alcance
(D7, D8), y OP-6 (baseline Butterworth sobre sellado).

Se escribe último. Este archivo fija de antemano **qué se va a poder afirmar**,
para que la redacción no se estire más allá de lo medido.

---

## 9.1 Qué queda demostrado

Lo que se puede afirmar hoy, sin que falte nada:

1. **Un CRN causal a 16 kHz cumple la restricción de tiempo real con margen de
   3×** en la CPU objetivo: RTF 0,331 mediana, 0,340 p95 en un i5-4460 monohilo.
   Con 10 ms de latencia algorítmica, no cero.
2. **La loss combinada MSE + SI-SDR aporta sobre MSE puro**: +0,199 PESQ-NB y
   +0,64 dB SI-SDR, con ganancia mayor en SNR alto. Es una mejora agnóstica al
   idioma y transfiere entre canales al 132 %.
3. **Los modelos entrenados sólo en inglés degradan activamente audio limpio en
   español.** En el bucket [15,20] dB de `test_v2_es`, V1 da Δ PESQ −0,167 y
   Δ SI-SDR −3,49 dB; V2 da −0,087 y −3,58 dB. Es transversal a la loss, no un
   artefacto de V1. El fine-tuning por idioma no es opcional.
4. **El efecto es de idioma, no de canal**, por el control preregistrado sobre
   `test_v3_mls_es` (F6, P1 → IDIOMA).
5. **El olvido catastrófico acá es selectivo, no difuso**: la mediana en inglés no
   se mueve, y la media cae por una cola de 6,8 % a 17,6 % de archivos que se
   rompen, monótona con la agresividad del learning rate.
6. **Un proxy perceptual congelado se gamea**, y el protocolo estabilizado sin
   reentrenamiento del proxy recupera sólo un tercio del daño.

## 9.2 Qué queda en estado preliminar

- **La compuerta de paso directo (V7)**: screening positivo en los cuatro
  endpoints preregistrados, con tres reservas explícitas. **Sin la confirmación de
  tres semillas se enuncia como hallazgo preliminar, no como aporte
  arquitectónico.**
- **Falta**: la decisión sobre si se corren las ~45 h.

## 9.3 Qué no se cumplió del alcance comprometido

- El eje de pérdida perceptual (D1) — cerrado como negativo caracterizado.
- El benchmark contra RNNoise y DeepFilterNet2 (D8) — pendiente.
- DNSMOS y SegSNR (D7) — no implementadas.
- El objetivo OP-6 contra el baseline Butterworth sobre material sellado.
- La auditoría auditiva subjetiva sobre muestra representativa.

**Regla de redacción**: se enumeran sin rodeos, cada uno con su causa. Un alcance
incumplido y declarado es gestión; uno incumplido y omitido es otra cosa.

## 9.4 Trabajo futuro

Priorizado por lo que más falta hace, no por lo que más interesa:

1. Confirmación de V7 con tres semillas.
2. Benchmark contra RNNoise y DeepFilterNet2.
3. Evaluación downstream sobre ASR — mide utilidad, no sólo calidad perceptual.
4. Proxy perceptual **reentrenado** durante el entrenamiento, que es lo que V4b
   identificó como el ingrediente activo del esquema de Xu.
5. GCRN como chequeo de validez externa del hallazgo por SNR sobre otra
   arquitectura — el único encuadre en que se justificaría, según el descarte
   del 06/09.
6. Filtrado de altas frecuencias, con la evidencia en contra ya documentada:
   DeepFilterNet2 no filtra la entrada, y Braun & Tashev 2021 reporta que un
   pasabajo degrada PESQ-WB entre 0,15 y 0,30 puntos.
7. Ablation V1b con 100k pares, para ubicar el punto de saturación.

## 9.5 Reflexión sobre el proceso

- **Afirma**: el proyecto cambió de estimando primario tres veces —de la media a
  la fracción de archivos rotos, del checkpoint a la trayectoria, del score
  compuesto a los dos ejes separados— y cada cambio salió de una medición, no de
  una preferencia.
- **Falta**: nada. Medio carilla. Es lo último que lee un tribunal antes de los
  anexos y conviene que sea corto.
