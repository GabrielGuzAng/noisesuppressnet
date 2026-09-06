# Reanalisis estadistico

Generado: 2026-09-06T02:45:07.564969+00:00
Seed: 42 | n_resamples: 10000 | metrica principal: pesq_nb | umbral de rotura: -0.2

## Integridad de los datos

- Archivos de resultados leidos: 10
- Conjuntos de pair_id consistentes (0..249): True
- NaN encontrados en campos numericos: 0
- Pares con snr_db coincidente entre metadata EN/ES: 250 / 250
- Pares con noise_file coincidente entre metadata EN/ES: 18 / 250
- Issues: ninguno

## F1 — Interaccion SNR x idioma

Estimando: monotonia de Delta PESQ-NB respecto al SNR. Correccion: holm (dentro de la familia, m=3).

| id | descripcion | rho | n | p crudo | p Holm |
|---|---|---|---|---|---|
| F1.a | Delta PESQ-NB de V1 vs snr_db, en v1_v1_en | -0.0651 | 250 | 0.3051 | 0.3051 |
| F1.b | Delta PESQ-NB de V1 vs snr_db, en v1_v2_es | -0.2617 | 250 | 2.778e-05 | 8.333e-05 |
| F1.c | (Delta_EN - Delta_ES) vs snr_db, apareado por pair_id | 0.1727 | 250 | 0.006203 | 0.01241 |

## F2 — Adaptacion (ganancia en ES)

Estimando: P(mejora) y media del cambio. Correccion: holm (dentro de la familia, m=3).

| id | comparacion | prop_improve | n_pos | n_neg | n_zero | media | mediana | skew | IC95 media | p crudo | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F2.v3 | v3_v2_es - v1_v2_es, PESQ-NB est | 0.8920 | 223 | 27 | 0 | 0.2295 | 0.1781 | 1.8365 | [0.1961, 0.2683] | 1.492e-39 | 2.985e-39 |
| F2.v3b | v3b_v2_es - v1_v2_es, PESQ-NB est | 0.8480 | 212 | 38 | 0 | 0.1523 | 0.0954 | 2.8120 | [0.1272, 0.1853] | 1.754e-30 | 1.754e-30 |
| F2.v3e | v3e_v2_es - v1_v2_es, PESQ-NB est | 0.9000 | 225 | 25 | 0 | 0.2205 | 0.1598 | 2.5986 | [0.1928, 0.2561] | 2.056e-41 | 6.167e-41 |

## F3 — Olvido (costo en EN)

Estimando: P(rotura), P5, media. Correccion: holm (dentro de la familia, m=3).

| id | comparacion | prop_improve | n_pos | n_neg | n_zero | media | mediana | skew | IC95 media | p crudo | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F3.v3 | v3_v1_en - v1_v1_en, PESQ-NB est | 0.5120 | 128 | 122 | 0 | -0.0788 | 0.0050 | -3.5167 | [-0.1312, -0.0403] | 0.7519 | 1 |
| F3.v3b | v3b_v1_en - v1_v1_en, PESQ-NB est | 0.4720 | 118 | 132 | 0 | -0.0282 | -0.0081 | -2.2080 | [-0.0463, -0.0139] | 0.411 | 1 |
| F3.v3e | v3e_v1_en - v1_v1_en, PESQ-NB est | 0.5280 | 132 | 118 | 0 | -0.0308 | 0.0071 | -2.5560 | [-0.0598, -0.0078] | 0.411 | 1 |

Cola de la distribucion de diferencias (proporcion de pares por debajo de cada umbral, P5 y P10):

| id | prop_broken(-0.1) | prop_broken(-0.2) | prop_broken(-0.3) | prop_broken(-0.5) | IC95 prop_broken(-0.2) | P5 | P10 |
|---|---|---|---|---|---|---|---|
| F3.v3 | 0.2960 | 0.1760 | 0.1200 | 0.0680 | [0.1320, 0.2280] | -0.5926 | -0.3556 |
| F3.v3b | 0.1840 | 0.0680 | 0.0360 | 0.0160 | [0.0400, 0.1040] | -0.2237 | -0.1562 |
| F3.v3e | 0.2440 | 0.1160 | 0.0800 | 0.0280 | [0.0800, 0.1600] | -0.3888 | -0.2205 |

## F4 — Perfil por bucket (EXPLORATORIO, sin correccion)

Estimando: medias por bucket de PESQ-NB delta (est - noisy). **No se corrige por Holm**: esta tabla no se usa para afirmar significancia, es el perfil descriptivo que alimenta la figura.

| variante | test_set | bucket_idx | n_pares | media PESQ-NB delta | IC95 |
|---|---|---|---|---|---|
| v1 | v1_en | 0 | 50 | 0.4391 | [0.3539, 0.5251] |
| v1 | v1_en | 1 | 50 | 0.5563 | [0.4427, 0.6655] |
| v1 | v1_en | 2 | 50 | 0.6250 | [0.5073, 0.7554] |
| v1 | v1_en | 3 | 50 | 0.5225 | [0.4253, 0.6338] |
| v1 | v1_en | 4 | 50 | 0.3426 | [0.2319, 0.4533] |
| v2 | v1_en | 0 | 50 | 0.5768 | [0.4842, 0.6752] |
| v2 | v1_en | 1 | 50 | 0.7342 | [0.6256, 0.8526] |
| v2 | v1_en | 2 | 50 | 0.8138 | [0.6837, 0.9588] |
| v2 | v1_en | 3 | 50 | 0.7645 | [0.6505, 0.8887] |
| v2 | v1_en | 4 | 50 | 0.5951 | [0.4732, 0.7184] |
| v3 | v1_en | 0 | 50 | 0.3908 | [0.3076, 0.4826] |
| v3 | v1_en | 1 | 50 | 0.5486 | [0.4364, 0.6699] |
| v3 | v1_en | 2 | 50 | 0.5120 | [0.3054, 0.6689] |
| v3 | v1_en | 3 | 50 | 0.4939 | [0.3924, 0.5985] |
| v3 | v1_en | 4 | 50 | 0.1464 | [-0.0800, 0.3156] |
| v3b | v1_en | 0 | 50 | 0.4117 | [0.3305, 0.4923] |
| v3b | v1_en | 1 | 50 | 0.5409 | [0.4313, 0.6490] |
| v3b | v1_en | 2 | 50 | 0.5905 | [0.4629, 0.7246] |
| v3b | v1_en | 3 | 50 | 0.4856 | [0.3928, 0.5956] |
| v3b | v1_en | 4 | 50 | 0.3159 | [0.2074, 0.4237] |
| v3e | v1_en | 0 | 50 | 0.4056 | [0.3194, 0.4860] |
| v3e | v1_en | 1 | 50 | 0.5498 | [0.4376, 0.6642] |
| v3e | v1_en | 2 | 50 | 0.5844 | [0.4497, 0.7185] |
| v3e | v1_en | 3 | 50 | 0.5096 | [0.4083, 0.6234] |
| v3e | v1_en | 4 | 50 | 0.2824 | [0.1458, 0.4052] |
| v1 | v2_es | 0 | 50 | 0.3067 | [0.2086, 0.4057] |
| v1 | v2_es | 1 | 50 | 0.2667 | [0.1155, 0.3748] |
| v1 | v2_es | 2 | 50 | 0.3683 | [0.2692, 0.4552] |
| v1 | v2_es | 3 | 50 | 0.0732 | [-0.0779, 0.1960] |
| v1 | v2_es | 4 | 50 | -0.1668 | [-0.3370, -0.0165] |
| v2 | v2_es | 0 | 50 | 0.3925 | [0.2959, 0.5120] |
| v2 | v2_es | 1 | 50 | 0.3700 | [0.2108, 0.5019] |
| v2 | v2_es | 2 | 50 | 0.5209 | [0.3959, 0.6228] |
| v2 | v2_es | 3 | 50 | 0.2564 | [0.1014, 0.3883] |
| v2 | v2_es | 4 | 50 | -0.0870 | [-0.2830, 0.0859] |
| v3 | v2_es | 0 | 50 | 0.4065 | [0.3162, 0.5042] |
| v3 | v2_es | 1 | 50 | 0.4807 | [0.3502, 0.5857] |
| v3 | v2_es | 2 | 50 | 0.5378 | [0.4395, 0.6233] |
| v3 | v2_es | 3 | 50 | 0.4236 | [0.3325, 0.5080] |
| v3 | v2_es | 4 | 50 | 0.1473 | [0.0135, 0.2589] |
| v3b | v2_es | 0 | 50 | 0.3492 | [0.2602, 0.4411] |
| v3b | v2_es | 1 | 50 | 0.4023 | [0.2711, 0.5007] |
| v3b | v2_es | 2 | 50 | 0.4670 | [0.3786, 0.5398] |
| v3b | v2_es | 3 | 50 | 0.3135 | [0.2085, 0.3975] |
| v3b | v2_es | 4 | 50 | 0.0778 | [-0.0656, 0.1963] |
| v3e | v2_es | 0 | 50 | 0.4109 | [0.3257, 0.5061] |
| v3e | v2_es | 1 | 50 | 0.4539 | [0.3085, 0.5604] |
| v3e | v2_es | 2 | 50 | 0.5332 | [0.4398, 0.6139] |
| v3e | v2_es | 3 | 50 | 0.4270 | [0.3228, 0.5092] |
| v3e | v2_es | 4 | 50 | 0.1257 | [-0.0184, 0.2499] |

## F5 — Control de la cola (olvido vs regresion a la media)

Estimando: exceso de rotura del tratamiento sobre el control pareado; solapamiento de los conjuntos rotos entre tratamientos; confusor medido (nivel basal de V1) y su control estratificado. Correccion: holm (dentro de la familia, m=9: 3 McNemar + 3 hipergeometrico + 3 McNemar estratificado; F5.c queda afuera, sin correccion). Control primario: v4b_placebo_epoch_03. Umbral de rotura: -0.2.

### F5.a — McNemar contra el control primario

| id | descripcion | b | c | n_discordant | prop_treatment | prop_control | p crudo | p Holm |
|---|---|---|---|---|---|---|---|---|
| F5.a.v3 | v3_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, mismos pair_id) | 44 | 0 | 44 | 0.1760 | 0.0000 | 1.137e-13 | 1.023e-12 |
| F5.a.v3b | v3b_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, mismos pair_id) | 17 | 0 | 17 | 0.0680 | 0.0000 | 1.526e-05 | 4.578e-05 |
| F5.a.v3e | v3e_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, mismos pair_id) | 29 | 0 | 29 | 0.1160 | 0.0000 | 3.725e-09 | 2.235e-08 |

### F5.b — Solapamiento entre pares de tratamientos (hipergeometrica)

| id | descripcion | observado | esperado | n_a | n_b | p crudo | p Holm |
|---|---|---|---|---|---|---|---|
| F5.b.v3_v3b | Solapamiento de broken_set(v3) y broken_set(v3b) | 12 | 2.9920 | 44 | 17 | 7.039e-07 | 3.52e-06 |
| F5.b.v3_v3e | Solapamiento de broken_set(v3) y broken_set(v3e) | 20 | 5.1040 | 44 | 29 | 3.952e-11 | 2.766e-10 |
| F5.b.v3b_v3e | Solapamiento de broken_set(v3b) y broken_set(v3e) | 14 | 1.9720 | 17 | 29 | 1.487e-12 | 1.19e-11 |

### F5.d — McNemar contra el control primario, estrato PESQ-NB V1 alto

| id | descripcion | b | c | n_discordant | prop_treatment | prop_control | n_estrato | corte basal | p crudo | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|
| F5.d.v3 | v3_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, estrato PESQ-NB V1 >= cuantil 0.75) | 18 | 0 | 18 | 0.2857 | 0.0000 | 63 | 3.1448 | 7.629e-06 | 3.052e-05 |
| F5.d.v3b | v3b_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, estrato PESQ-NB V1 >= cuantil 0.75) | 6 | 0 | 6 | 0.0952 | 0.0000 | 63 | 3.1448 | 0.03125 | 0.03125 |
| F5.d.v3e | v3e_v1_en roto vs v4b_placebo_epoch_03 roto (McNemar, estrato PESQ-NB V1 >= cuantil 0.75) | 12 | 0 | 12 | 0.1905 | 0.0000 | 63 | 3.1448 | 0.0004883 | 0.0009766 |

### F5.c — Confusor medido (nivel basal de V1), SIN corregir

| grupo | n_a | n_b | media_a | media_b | diferencia | IC95 BCa | p crudo |
|---|---|---|---|---|---|---|---|
| union_broken_vs_never_broken | 54 | 196 | 2.8661 | 2.5898 | 0.2763 | [0.0771, 0.4796] | 0.009429 |
| triple_vs_rest | 10 | 240 | 2.6938 | 2.6477 | 0.0461 | [-0.2387, 0.3830] | 0.8286 |
| triple_vs_never_broken | 10 | 196 | 2.6938 | 2.5898 | 0.1039 | [-0.1879, 0.4438] | 0.5959 |

(corrected=False)

### F5 — Perfil estratificado (7 series x 4 cuartiles de PESQ-NB basal de V1)

| serie | rol | cuartil | n | prop_broken |
|---|---|---|---|---|
| v3 | treatment | 1 | 63 | 0.1111 |
| v3 | treatment | 2 | 62 | 0.1774 |
| v3 | treatment | 3 | 62 | 0.1290 |
| v3 | treatment | 4 | 63 | 0.2857 |
| v3b | treatment | 1 | 63 | 0.0159 |
| v3b | treatment | 2 | 62 | 0.1290 |
| v3b | treatment | 3 | 62 | 0.0323 |
| v3b | treatment | 4 | 63 | 0.0952 |
| v3e | treatment | 1 | 63 | 0.0476 |
| v3e | treatment | 2 | 62 | 0.1935 |
| v3e | treatment | 3 | 62 | 0.0323 |
| v3e | treatment | 4 | 63 | 0.1905 |
| v4b_placebo_epoch_01 | control | 1 | 63 | 0.0000 |
| v4b_placebo_epoch_01 | control | 2 | 62 | 0.0000 |
| v4b_placebo_epoch_01 | control | 3 | 62 | 0.0000 |
| v4b_placebo_epoch_01 | control | 4 | 63 | 0.0000 |
| v4b_placebo_epoch_02 | control | 1 | 63 | 0.0000 |
| v4b_placebo_epoch_02 | control | 2 | 62 | 0.0161 |
| v4b_placebo_epoch_02 | control | 3 | 62 | 0.0000 |
| v4b_placebo_epoch_02 | control | 4 | 63 | 0.0000 |
| v4b_placebo_epoch_03 | control | 1 | 63 | 0.0000 |
| v4b_placebo_epoch_03 | control | 2 | 62 | 0.0000 |
| v4b_placebo_epoch_03 | control | 3 | 62 | 0.0000 |
| v4b_placebo_epoch_03 | control | 4 | 63 | 0.0000 |
| v2 | control | 1 | 63 | 0.0000 |
| v2 | control | 2 | 62 | 0.0000 |
| v2 | control | 3 | 62 | 0.0000 |
| v2 | control | 4 | 63 | 0.0000 |

### F5 — Sensibilidad de prop_broken por umbral (7 series x 4 umbrales)

| serie | rol | prop_broken(-0.1) | prop_broken(-0.2) | prop_broken(-0.3) | prop_broken(-0.5) |
|---|---|---|---|---|---|
| v3 | treatment | 0.2960 | 0.1760 | 0.1200 | 0.0680 |
| v3b | treatment | 0.1840 | 0.0680 | 0.0360 | 0.0160 |
| v3e | treatment | 0.2440 | 0.1160 | 0.0800 | 0.0280 |
| v4b_placebo_epoch_01 | control | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v4b_placebo_epoch_02 | control | 0.0080 | 0.0040 | 0.0000 | 0.0000 |
| v4b_placebo_epoch_03 | control | 0.0040 | 0.0000 | 0.0000 | 0.0000 |
| v2 | control | 0.0120 | 0.0000 | 0.0000 | 0.0000 |
