# Capítulo 10 — Anexos

**Extensión estimada**: 20 pp · **Estado**: parcial

Va lo que un tribunal puede querer verificar y no cabe en el cuerpo. Criterio de
corte: si alguien tiene que poder **reproducir** o **auditar** algo, va acá; si es
sólo ilustrativo, no va.

---

## A. Configuraciones de entrenamiento

- **Contenido**: volcado de `training/config.py`, una tabla por variante con
  learning rate, épocas, scheduler, batch size, loss, checkpoint de inicialización
  y semilla.
- **Fuente**: `training/config.py` (21 configuraciones).
- **Estado**: listo. Es una extracción automática.

## B. Hashes de sellado y verificación de integridad

- **Contenido**: los tres hashes SHA-256 completos, la metadata de cada sellado,
  y el procedimiento de verificación (`scripts/verify_test_set.py`).
- **Fuente**: `FUENTES.md`, sección de hashes.
- **Estado**: listo.

## C. Preregistros

- **Contenido**: el texto completo de los tres preregistros con su hash, y el
  momento en que se hashearon respecto de la disponibilidad del dato.
- **Fuente**: `docs/preregistro_mls_es.sha256`, `preregistro_compuerta.sha256`,
  `preregistro_v7_desde_cero.sha256`.
- **Estado**: listo para los tres existentes. El de la confirmación de V7 no
  existe.
- **Nota**: incluir el error de signo de P3 en V6 **tal como fue escrito**, no
  corregido. Es la evidencia de que el preregistro se respetó.

## D. Reportes de costo de entrenamiento

- **Contenido**: un reporte por variante con horas, kWh, CO₂ estimado, eficiencia
  útil y costo equivalente en nube, más el total del proyecto.
- **Fuente**: `analysis/training_cost_report.py`.
- **Estado**: 12 corridas generadas el 19/09/2026 (128,09 h, 22,415 kWh,
  6,73 kg CO₂). Faltan los 8 brazos de V4 y V4b, los 5 del sweep de V3b, V0 y
  las 4 smoke. Ver `FUENTES.md`.

## E. Resultados completos por variante

- **Contenido**: tablas globales, por bucket de SNR y por categoría de ruido, para
  las ocho variantes de la línea principal sobre los tres sellados.
- **Fuente**: los ~40 JSON de `results/`.
- **Estado**: el dato está; falta la extracción tabular.
- **Nota**: los JSON traen `all_pairs` con el detalle por archivo. **No va al
  anexo impreso** — 250 filas × 8 variantes × 3 sellados. Se referencia el
  repositorio.

## F. Salida del análisis estadístico

- **Contenido**: `results/reanalysis_stats.json` en forma tabular, familia por
  familia, con estimandos, p-valores crudos, p-valores corregidos por Holm e
  intervalos bootstrap BCa.
- **Fuente**: `docs/reanalisis_estadistico.md`.
- **Estado**: listo.

## G. Tests y su salida

- **Contenido**: inventario de la suite y la salida de una corrida completa.
- **Estado**: **falta correr la suite y capturar la salida.** Si algo falla, se
  arregla antes.

## H. Código

- **Contenido**: referencia al repositorio con el tag correspondiente a cada
  resultado reportado, no listados de código en el papel.
- **Fuente**: `github.com/GabrielGuzAng/nosiesuppressnet`; 13 tags.
- **Nota**: verificar que cada número del cuerpo sea reproducible desde el tag que
  se cita. Es la promesa de reproducibilidad del proyecto y conviene chequearla
  antes de imprimirla.

## I. Figuras suplementarias

- **Disponibles**: `figures/psd_comparison.png`, `figures/spec_pair_0000.png`.
- **Por generar**: contraste por época con la banda del piso de ruido (§4.6);
  perfil estratificado por cuartil de F5 (§7.4); diagrama de bloques del CRN
  (§2.3); cono de dependencia temporal para causalidad y latencia (§2.2);
  captura del dashboard (§8.6).

## J. Muestras de escucha

- **Contenido**: procedimiento para regenerar audios noisy / clean / enhanced, uno
  por bucket de SNR, para cualquier combinación de variantes sobre cualquiera de
  los tres sellados.
- **Fuente**: `scripts/generate_listening_samples.py --lang es/en`.
- **Estado**: la herramienta existe. Falta decidir si se entregan muestras en
  soporte digital junto con el informe.
