# Reporte de Consumo de Entrenamiento — V5_S43

Generado: 2026-09-19T13:42:31.571685

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
| Épocas ejecutadas | 25 |
| Época del mejor modelo | 12 |
| Val loss del mejor modelo | -0.0772 |
| Tiempo por época (promedio) | 1905.8 s (31.8 min) |
| **Tiempo total de entrenamiento** | **13.23 h** |
| Tiempo hasta convergencia | 6.35 h |
| Eficiencia útil | 48.0 % |

## Consumo energético (estimado)

| Métrica | Valor |
|---------|-------|
| Potencia GPU promedio | 105 W |
| Potencia sistema (CPU+RAM+disco) | 70 W |
| Potencia total del sistema | 175 W |
| **Consumo total** | **2.316 kWh** |
| Consumo útil (hasta convergencia) | 1.111 kWh |

## Impacto ambiental (estimado)

| Métrica | Valor |
|---------|-------|
| Factor de emisión (Argentina) | 0.3 kg CO₂ / kWh |
| **Emisiones totales** | **0.695 kg CO₂** |
| Emisiones útiles | 0.333 kg CO₂ |
| Equivalencia | ≈ 2.8 km en auto naftero |

## Costo equivalente en la nube

| Métrica | Valor |
|---------|-------|
| Tarifa referencia (GCP T4) | $0.35 USD/h |
| **Costo equivalente total** | **$4.63 USD** |

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
