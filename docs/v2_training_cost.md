# Reporte de Consumo de Entrenamiento — V2

Generado: 2026-08-06T20:28:24.431940

## Configuración del hardware

| Componente | Valor |
|------------|-------|
| GPU | NVIDIA RTX 4060 Gaming X 8GB |
| CPU | Intel i5-4460 |
| RAM | 12 GB DDR3 |
| Sistema Operativo | Ubuntu 24.04 |

## Tiempo

| Métrica | Valor |
|---------|-------|
| Épocas ejecutadas | 20 |
| Época del mejor modelo | 19 |
| Val loss del mejor modelo | -0.0655 |
| Tiempo por época (promedio) | 1891.9 s (31.5 min) |
| **Tiempo total de entrenamiento** | **10.51 h** |
| Tiempo hasta convergencia | 9.99 h |
| Eficiencia útil | 95.0 % |

## Consumo energético (estimado)

| Métrica | Valor |
|---------|-------|
| Potencia GPU promedio | 105 W |
| Potencia sistema (CPU+RAM+disco) | 70 W |
| Potencia total del sistema | 175 W |
| **Consumo total** | **1.839 kWh** |
| Consumo útil (hasta convergencia) | 1.748 kWh |

## Impacto ambiental (estimado)

| Métrica | Valor |
|---------|-------|
| Factor de emisión (Argentina) | 0.3 kg CO₂ / kWh |
| **Emisiones totales** | **0.552 kg CO₂** |
| Emisiones útiles | 0.524 kg CO₂ |
| Equivalencia | ≈ 2.2 km en auto naftero |

## Costo equivalente en la nube

| Métrica | Valor |
|---------|-------|
| Tarifa referencia (GCP T4) | $0.35 USD/h |
| **Costo equivalente total** | **$3.68 USD** |

## Metodología y supuestos

- **Tiempo de entrenamiento:** medido directamente por el trainer (`epoch_time_s` en `history.json`).
- **Potencia GPU:** promedio observado con `nvidia-smi` durante entrenamiento (95-98% utilización, 105 W promedio bajo carga sostenida, TDP nominal 115 W).
- **Potencia sistema:** estimado según arquitectura (i5-4460 sostenido ~50 W, RAM DDR3 12 GB ~5 W, SSD ~3 W, motherboard/ventiladores ~15 W).
- **Factor de emisión:** Cammesa Argentina, mix generación 2024 (~30% hidro, ~40% térmico gas, ~10% nuclear, resto renovables). Referencia: [www.cammesa.com](https://www.cammesa.com).
- **Costo cloud:** tarifa GCP N1 con GPU T4 on-demand (comparación indicativa).

## Limitaciones del cálculo

- El consumo real puede variar ±15% del estimado por fluctuaciones de carga.
- No se incluye consumo de red, iluminación ambiental ni HVAC.
- El factor de emisión es un promedio horario; el consumo instantáneo depende del horario del día.
- La comparación cloud es solo aproximativa (T4 no es idéntico a RTX 4060).
