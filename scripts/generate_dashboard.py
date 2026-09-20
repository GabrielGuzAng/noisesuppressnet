"""
scripts/generate_dashboard.py

Genera un panel HTML autocontenido (un solo archivo, sin dependencias
externas ni servidor) con las métricas de todas las variantes evaluadas
sobre los test sets sellados, más una sección tipo wiki explicando cada
métrica. Pensado para abrir local (doble clic, o `python -m http.server`)
y mostrar avances en la defensa.

Lee resultados de results/*.json (los que tienen "global"/"by_bucket" --
ignora *_training_cost.json y results/v3_sweep/, que son otro tipo de dato).

Grafica solo la LINEA PRINCIPAL del ablation (noisy, V1, V2, V3, V3b, V3e, V5),
que son variantes comparables entre si sobre los mismos sellados. Las que no lo
son quedan fuera de los graficos a proposito y se cuentan en la seccion de
experimentos: las replicas de semilla de V5 (v5_s43/s44), y los disenos de
tratamiento/control de V6 y V7, cuyo estimando es un contraste contra su propio
control y no un valor absoluto que tenga sentido al lado de V1.

USO:
    python -m scripts.generate_dashboard
    # regenerar cuando haya resultados nuevos (ej. V3e) -- correrlo de nuevo
"""
import json
import math
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_PATH = PROJECT_ROOT / "docs" / "dashboard.html"

# Paleta categórica (dataviz skill, orden fijo, no ciclar). Validada con
# scripts/validate_palette.js del skill sobre los 6 slots de serie: pasa las
# seis comprobaciones en claro y oscuro. El gris de "noisy" queda fuera de esa
# validación a propósito — es la referencia, no una serie, y por eso es acromático.
# El WARN de contraste en modo claro (aqua, amarillo, magenta bajo 3:1) se cubre
# con la regla de relieve: cada barra lleva su valor escrito y hay tabla completa.
COLORS = {
    "light": {
        "noisy":  "#898781",  # muted, es la referencia, no una variante
        "v1":     "#2a78d6",  # slot 1 blue
        "v2":     "#eb6834",  # slot 2 orange
        "v3":     "#1baf7a",  # slot 3 aqua
        "v3b":    "#eda100",  # slot 4 yellow
        "v3e":    "#e87ba4",  # slot 5 magenta
        "v5":     "#008300",  # slot 6 green
    },
    "dark": {
        "noisy":  "#898781",
        "v1":     "#3987e5",
        "v2":     "#d95926",
        "v3":     "#199e70",
        "v3b":    "#c98500",
        "v3e":    "#d55181",
        "v5":     "#008300",
    },
}

VARIANT_ORDER = ["noisy", "v1", "v2", "v3", "v3b", "v3e", "v5"]
VARIANT_LABEL = {"noisy": "Noisy (sin procesar)", "v1": "V1", "v2": "V2",
                  "v3": "V3", "v3b": "V3b", "v3e": "V3e", "v5": "V5"}

# Los tres sellados, en el orden en que se muestran las pestañas.
TEST_SETS = ["v1_en", "v2_es", "v3_mls_es"]
TEST_SET_LABEL = {
    "v1_en": "Inglés · LibriSpeech",
    "v2_es": "Español · Common Voice",
    "v3_mls_es": "Español · audiolibro (MLS)",
}
TEST_SET_SUB = {
    "v1_en": "250 pares, 5 buckets de SNR. El idioma de entrenamiento.",
    "v2_es": "250 pares. Cambia el idioma Y el canal de grabación respecto de v1_en.",
    "v3_mls_es": "250 pares. Español con el mismo paradigma de grabación que v1_en "
                 "(audiolibro) y las condiciones de ruido copiadas verbatim: aísla "
                 "idioma de canal.",
}

METRICS = [
    # (clave, label, unidad, máximo fijo o None=dinámico, paso de la escala)
    ("pesq_nb", "PESQ-NB", "", 4.5, 0.5),
    ("pesq_wb", "PESQ-WB", "", 4.5, 0.5),
    ("stoi",    "STOI",    "", 1.0, 0.2),
    ("sisdr",   "SI-SDR",  " dB", None, None),  # paso calculado del rango real
]

TIMELINE = [
    ("V0", "22 jun 2026", "Baseline con 200 pares. Output collapse identificado: el modelo predice silencio uniforme en alta frecuencia por falta de datos."),
    ("V1", "27 jul 2026", "Dataset escalado a 50k pares. El output collapse se rompe — primera mejora perceptual real sobre el audio ruidoso en las 4 métricas."),
    ("V2", "2 ago 2026", "Loss combinada MSE + SI-SDR (α=0,7). Mejora adicional sobre V1, más marcada en SNR alto."),
    ("V3", "20 ago 2026", "Fine-tuning completo de V1 sobre español (Common Voice ES), mismo lr/épocas que V1. Gana en español, pierde algo en inglés (forgetting moderado)."),
    ("V3b", "22-23 ago 2026", "Fine-tuning conservador: lr elegido por barrido empírico (5 candidatos), 10 épocas. Olvida menos inglés que V3, pero gana menos en español."),
    ("V3e", "23-24 ago 2026", "lr 1e-4 con 25 épocas y decay tardío. Mejor balance de las tres: casi la ganancia en español de V3 con el forgetting de V3b."),
    ("V4", "31 ago 2026", "MSE + Squim como proxy perceptual diferenciable. Resultado negativo: el proxy fijo se gamea desde la primera época."),
    ("V4b", "1 sep 2026", "Protocolo estabilizado de Xu 2022 sobre el mismo proxy. Recupera un tercio del daño; el término perceptual sigue costando −0,598 PESQ-NB contra placebo. Línea cerrada."),
    ("Reanálisis", "5-6 sep 2026", "Reanálisis estadístico de todo lo acumulado: familias de hipótesis con estimandos declarados, corrección de Holm e IC bootstrap BCa. Retira el \"score compuesto\" de V3 y reformula el aporte 1 como interacción."),
    ("test v3", "13 sep 2026", "Tercer sellado: español de audiolibro (MLS) con las condiciones de ruido de v1_en copiadas verbatim. Separa el efecto del idioma del efecto del canal de grabación."),
    ("V5", "7-9 sep 2026", "La propuesta completa: V2 fine-tuneado a español con la receta de V3e, replicado en tres semillas."),
    ("V6", "14-15 sep 2026", "Compuerta causal de paso directo atornillada a V2. Negativo sobre el endpoint primario, con el mecanismo confirmado y el piso de ruido por checkpoint medido."),
    ("V7", "15-18 sep 2026", "La misma compuerta, pero presente desde la inicialización y entrenada desde cero contra un control desde cero. Screening positivo."),
]

# Secciones narrativas por experimento. Los números son los publicados en
# docs/ y en los mensajes de commit de cada variante; si se corrige un número
# allá, hay que corregirlo acá (son texto, no se leen de los JSON).
EXPERIMENTS = [
    ("V0", "El pipeline antes que el modelo", "negativo",
     "¿El camino completo —STFT, CRN, iSTFT— entrena y devuelve audio, con la causalidad intacta?",
     "200 pares, 10 épocas, MSE sobre la magnitud del espectrograma. Un baseline deliberadamente chico.",
     "PESQ-NB 1,42 · STOI 0,79 · SI-SDR 3,11 dB. El análisis de densidad espectral mostró <em>output collapse</em>: "
     "el modelo aprendió a predecir silencio uniforme en alta frecuencia en vez de reconstruir la voz.",
     "El pipeline funciona; el modelo no. La causa es el volumen de datos y el underfitting, no la arquitectura. "
     "Sirve como piso y como control de que el resto del sistema no miente."),

    ("V1", "Escalar el dataset", "positivo",
     "¿El output collapse es falta de datos?",
     "50.000 pares de entrenamiento y 2.000 de validación (LibriSpeech + MUSAN + ESC-50), 30 épocas, misma loss MSE. "
     "Una sola variable cambia respecto de V0: cuántos datos ve.",
     "+1,23 puntos de PESQ-NB sobre V0, y el collapse desaparece del análisis espectral.",
     "Sí era falta de datos. La ganancia <strong>no es extrapolable linealmente</strong>: la literatura "
     "(Reddy 2020, Kolbæk 2020) muestra saturación logarítmica, así que duplicar otra vez el dataset no "
     "vuelve a dar +1,2."),

    ("V2", "Loss combinada MSE + SI-SDR", "positivo",
     "¿Agregar un término que mire la forma de onda, y no solo la magnitud del espectro, aporta información complementaria?",
     "loss = α·MSE + (1−α)·(−SI-SDR·escala), con α = 0,7. O sea 70% del peso a la magnitud y 30% a la forma de onda. "
     "Todo lo demás idéntico a V1.",
     "+0,20 PESQ-NB y +0,64 dB SI-SDR sobre V1, con la ganancia concentrada en SNR alto.",
     "Confirmación empírica de Braun &amp; Tashev 2021 (sección 4.3). V2 es el backbone del que salen V5, V6 y V7."),

    ("V3 · V3b · V3e", "Fine-tuning al español", "mixto",
     "¿Hace falta adaptar el modelo al idioma, y a qué costo sobre el idioma original?",
     "Tres regímenes de fine-tuning desde V1 sobre Common Voice ES. V3: agresivo (lr 2e-4, 30 épocas). "
     "V3b: conservador (lr 5e-5, 10 épocas, lr elegido por barrido). V3e: lr 1e-4 con 25 épocas y decay tardío.",
     "ΔPESQ-NB en español / en inglés — V3: +0,230 / −0,079 · V3b: +0,153 / −0,029 · V3e: +0,221 / −0,031.",
     "V3e da el mejor balance. Dos advertencias que el reanálisis de septiembre dejó escritas: el diseño de V3b "
     "confundió épocas con lr (cambió las dos cosas a la vez), y el <em>score compuesto</em> que ordenaba las tres "
     "variantes quedó <strong>retirado</strong> — sumaba ganancia en español con olvido en inglés, que son estimandos "
     "de naturaleza distinta."),

    ("V4 · V4b", "El proxy perceptual, y qué es α", "negativo",
     "PESQ no es diferenciable, así que no se puede optimizar directamente. ¿Sirve entrenar contra un proxy "
     "diferenciable que estima PESQ (Squim, Kumar 2023)?",
     "<strong>α es el peso del término MSE en la loss combinada:</strong> loss = α·MSE + (1−α)·(−PESQ estimado). "
     "Con α = 0,95 el término perceptual pesa un 5%; con α = 0,70, un 30%. V4 barrió α ∈ {0,70 · 0,80 · 0,90 · 0,95} "
     "contra un control de MSE puro. V4b agregó el protocolo estabilizado de Xu 2022 (arranque desde un modelo "
     "convergido, lr 2e-5) y un placebo con α = 1.",
     "El proxy se gamea desde la primera época en las cuatro corridas: el PESQ <em>estimado</em> trepa hacia su techo "
     "estructural (4,64) mientras el PESQ <em>real</em> cae por debajo del audio sin procesar. La brecha entre los dos "
     "llega a +2,36 y +2,77 puntos, unas 16 a 20 veces el error nominal de Squim (0,142). El protocolo estabilizado "
     "recupera solo un tercio del daño: el término perceptual sigue costando −0,598 PESQ-NB contra el placebo.",
     "Ley de Goodhart con datos propios. La línea del proxy perceptual queda cerrada. El placebo fue decisivo: "
     "contra el audio sin procesar el resultado parecía casi arreglado, y solo el control con idéntico protocolo "
     "mostró el costo real."),

    ("V5", "La propuesta completa, y tres semillas", "mixto",
     "¿Cuánto rinde el sistema completo —loss combinada más adaptación al español— y cuánto de eso es la semilla?",
     "V2 fine-tuneado a español con la receta de V3e, replicado con las semillas 42, 43 y 44. Las réplicas miden la "
     "varianza del <em>fine-tuning</em> (el orden de los datos), no la de inicialización: las tres parten del mismo checkpoint.",
     "PESQ-NB por semilla — español 2,686 / 2,658 / 2,661 (sd 0,015) · inglés 2,774 / 2,770 / 2,766 (sd 0,004) · "
     "audiolibro español 2,840 / 2,803 / 2,816 (sd 0,019). La ventaja sobre V3e en español es de siete a nueve "
     "desvíos de semilla.",
     "El efecto es real y sobrevive a la semilla, pero <strong>la semilla 42 es la más alta de las tres en los tres "
     "sellados</strong>: el titular sale del máximo de tres corridas y eso hay que decirlo. El barrido de época mostró "
     "además que el pico está en la época 18 y no en la de mínima val_loss — cuarta evidencia de que el criterio de "
     "selección no está alineado con PESQ."),

    ("V6", "Compuerta causal de paso directo", "negativo",
     "El CRN hace mapeo espectral directo: la salida nunca toca la magnitud de entrada, así que no existe un camino "
     "barato para decir «acá no hagas nada». ¿Agregar ese camino de identidad acota la degradación fuera de dominio?",
     "M_out = g·M̂ + (1−g)·M_noisy, con g aprendida, causal y por banda: 193 parámetros sobre 17.579.459. "
     "Diseño tratamiento/placebo, 6 épocas de fine-tuning desde V2, entrenando solo en inglés.",
     "El endpoint primario preregistrado <strong>falla</strong>: +2,5 pp contra un umbral de +3 pp, p = 0,181. "
     "El mecanismo, en cambio, se cumple con fuerza: ρ(g, SNR) ≈ −0,4 con p &lt; 1e−8 en los tres sellados — la compuerta "
     "aprendió a replegarse justo donde la entrada está limpia.",
     "Negativo sobre la hipótesis, con dos subproductos que valen más que el veredicto. Uno: el <strong>piso de ruido "
     "por checkpoint</strong> (sd 0,0099 en español, 0,0166 en inglés) es del mismo tamaño que el efecto buscado, así que "
     "ninguna comparación de un solo checkpoint puede resolver efectos de esta magnitud. Dos: el diagnóstico — la "
     "compuerta aprendió a detectar «la entrada está limpia», no «estoy fuera de mi dominio», que es una señal que no "
     "existe en datos de un solo idioma."),

    ("V7", "La misma compuerta, desde cero", "positivo",
     "V6 atornilló la compuerta a un backbone que ya había convergido 20 épocas sin ella. ¿Con el camino de identidad "
     "disponible desde la inicialización, la red aprende una división del trabajo distinta?",
     "Dos entrenamientos desde cero de 20 épocas, idénticos salvo por la compuerta: tratamiento y control. "
     "El estimando es el contraste promediado sobre las épocas 15 a 20 — no un checkpoint único, precisamente por el "
     "piso de ruido que midió V6. Umbral de screening preregistrado: +0,050, unas tres veces esa sd.",
     "Los cuatro endpoints preregistrados pasan. El contraste promediado sobre las épocas 15-20 da "
     "<strong>+0,0795</strong> en español crowdsourced, <strong>+0,0753</strong> en español de audiolibro y "
     "<strong>+0,0620</strong> en inglés — seis de seis épocas positivas en los tres sellados, o sea que no hay costo "
     "en el idioma de entrenamiento sino ganancia. El mecanismo se replica: ρ(g, SNR) = −0,35 / −0,42 / −0,40, la "
     "compuerta se repliega donde la entrada ya está limpia. Y el dato sustantivo: en el bucket de SNR alto [15,20] dB "
     "en español, el control desde cero <em>replica la patología</em> de V1 y V2 —degrada el audio, −0,084 contra la "
     "entrada sin procesar— y la compuerta la cruza a cero (+0,006).",
     "Screening positivo, con tres reservas escritas. Es <strong>una sola semilla por brazo</strong>. La ganancia es pareja "
     "entre buckets y entre sellados —el orden va en la dirección que predice la hipótesis, pero el rango entero es más "
     "chico que el ruido entre épocas— y el modelo con compuerta también ajusta mejor en validación, así que parte del "
     "efecto es que el camino de identidad facilita la optimización y no solo que proteja la entrada limpia. Y la "
     "dispersión entre épocas resultó más ancha de lo que V6 había medido. La confirmación con tres semillas es el "
     "paso declarado; hasta que corra, esto no se escribe como resultado confirmado."),
]

METRIC_WIKI = [
    ("PESQ-NB / PESQ-WB", "Perceptual Evaluation of Speech Quality",
     "Predice qué tan buena sonaría la señal procesada para un oyente humano, comparándola contra el audio limpio de referencia. Estándar ITU-T P.862. "
     "\"NB\" (narrowband, 8 kHz) y \"WB\" (wideband, 16 kHz) son dos variantes del mismo algoritmo con distinto ancho de banda de referencia. "
     "Rango típico: ~1.0 (muy mala) a ~4.5 (excelente, indistinguible del original). Este proyecto usa la librería <code>pesq</code>.",
     "Kumar et al. 2023 (referencia de proxies perceptuales); ITU-T P.862"),
    ("STOI", "Short-Time Objective Intelligibility",
     "Mide inteligibilidad, no calidad: qué tan fácil es entender las palabras, no qué tan \"lindo\" suena. Compara la envolvente temporal de la señal procesada "
     "contra la limpia en bandas de frecuencia cortas. Rango 0 a 1 (correlaciona con % de palabras entendidas correctamente por oyentes humanos). "
     "Este proyecto usa <code>pystoi</code> en su variante clásica (no extendida).",
     "Taal et al. 2011 (paper original de STOI)"),
    ("SI-SDR", "Scale-Invariant Signal-to-Distortion Ratio",
     "Mide, en decibeles, cuánta señal \"limpia\" hay respecto al \"ruido/distorsión\" residual en la salida — invariante a cambios de volumen entre estimado y "
     "referencia (a diferencia de SNR clásico). Valores más altos = mejor. No tiene techo teórico; en este proyecto los valores útiles rondan 0-20 dB.",
     "Le Roux et al. 2019, \"SDR — Half-baked or Well Done?\", ICASSP"),
]


def load_results():
    """variant -> {test_set_id: resultado}

    Solo entran las variantes de VARIANT_ORDER: los resultados de V6/V7 y de
    las réplicas de semilla también viven en results/, pero no son comparables
    como barras absolutas (ver docstring del módulo).
    """
    data = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if "training_cost" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "global" not in d or "variant" not in d:
            continue
        variant = d["variant"]
        if variant not in VARIANT_ORDER:
            continue
        ts = Path(d.get("test_set", "")).name
        if ts not in TEST_SETS:
            continue
        data.setdefault(variant, {})[ts] = d

    # "noisy" no tiene JSON propio -- sus valores viven anidados como
    # "noisy_mean"/"noisy_std" dentro de cada variante evaluada sobre el
    # mismo sellado (son el mismo audio de entrada, iguales entre
    # variantes). Se sintetiza una entrada "noisy" por sellado para que
    # las tablas/gráficos la traten como una fila más.
    for ts in {t for v in data.values() for t in v}:
        any_result = next(d[ts] for d in data.values() if ts in d)
        noisy_global = {
            mk: {"noisy_mean": m["noisy_mean"], "est_mean": m["noisy_mean"],
                 "noisy_std": m["noisy_std"], "est_std": m["noisy_std"], "delta_mean": 0.0}
            for mk, m in any_result["global"].items()
        }
        data.setdefault("noisy", {})[ts] = {
            "variant": "noisy", "test_set": any_result["test_set"],
            "global": noisy_global, "by_bucket": None,
        }
    return data


def _nice_step(raw_max):
    """Paso de escala razonable para un rango dinámico (SI-SDR)."""
    for step in (1, 2, 5, 10, 20):
        if raw_max / step <= 8:
            return step
    return 20


def svg_bar_panel(metric_key, metric_label, unit, fixed_max, grid_step, lang_data, colors, panel_id):
    """Grupo de barras horizontales: una fila por variante, para una métrica.
    Incluye escala de referencia (gridlines + números) cada `grid_step`,
    no solo el valor puntual en la punta de cada barra."""
    rows = []
    for v in VARIANT_ORDER:
        if v not in lang_data:
            continue
        d = lang_data[v]
        if v == "noisy":
            val = d["global"][metric_key]["noisy_mean"]
        else:
            val = d["global"][metric_key]["est_mean"]
        rows.append((v, val))

    if not rows:
        return ""

    raw_max = max(v for _, v in rows)
    if fixed_max:
        axis_max, step = fixed_max, grid_step
    else:
        step = grid_step or _nice_step(raw_max * 1.15)
        axis_max = math.ceil((raw_max * 1.15) / step) * step

    W, row_h, bar_h, label_w, pad_top, pad_bottom = 480, 34, 22, 130, 12, 26
    chart_h = row_h * len(rows)
    H = pad_top + chart_h + pad_bottom
    chart_x0 = label_w
    chart_w = W - label_w - 40

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
           f'aria-label="Comparación de {metric_label}">']

    # Escala de referencia: gridlines verticales + números cada `step`
    n_ticks = round(axis_max / step)
    for i in range(n_ticks + 1):
        tick_val = i * step
        x = chart_x0 + (tick_val / axis_max) * chart_w
        svg.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{pad_top + chart_h}" class="gridline"/>')
        tick_str = f"{tick_val:.1f}" if step < 1 else f"{tick_val:.0f}"
        svg.append(f'<text x="{x:.1f}" y="{pad_top + chart_h + 16}" text-anchor="middle" '
                    f'class="axis-label">{tick_str}</text>')

    for i, (v, val) in enumerate(rows):
        y = pad_top + i * row_h
        bar_len = max(2, (val / axis_max) * chart_w)
        color = colors[v]
        is_noisy = v == "noisy"
        opacity = "0.55" if is_noisy else "1"
        svg.append(f'<text x="{chart_x0 - 10}" y="{y + bar_h/2 + 4}" text-anchor="end" '
                    f'class="bar-label" font-weight="{"400" if is_noisy else "600"}">{VARIANT_LABEL[v]}</text>')
        svg.append(f'<rect x="{chart_x0}" y="{y}" width="{bar_len:.1f}" height="{bar_h}" rx="4" '
                    f'fill="{color}" opacity="{opacity}"/>')
        val_str = f"{val:.2f}{unit}" if metric_key != "stoi" else f"{val:.3f}"
        svg.append(f'<text x="{chart_x0 + bar_len + 8:.1f}" y="{y + bar_h/2 + 4}" '
                    f'class="bar-value">{val_str}</text>')
    svg.append("</svg>")
    return (f'<div class="panel"><h4>{metric_label}</h4>' + "".join(svg) + "</div>")


def svg_bucket_line(lang_data, colors):
    """Línea: ΔPESQ-NB por bucket de SNR, una serie por variante."""
    buckets_ref = None
    series = {}
    for v in VARIANT_ORDER:
        if v == "noisy" or v not in lang_data:
            continue
        d = lang_data[v]
        by_bucket = d.get("by_bucket")
        if not by_bucket:
            continue
        if buckets_ref is None:
            buckets_ref = by_bucket
        series[v] = [b["pesq_nb_delta_mean"] for b in by_bucket]

    if not series or buckets_ref is None:
        return ""

    labels = [f'{b["snr_range_db"][0]} a {b["snr_range_db"][1]} dB' for b in buckets_ref]
    all_vals = [x for vals in series.values() for x in vals]
    y_min, y_max = min(0, min(all_vals)), max(all_vals)
    y_pad = (y_max - y_min) * 0.15 or 0.05
    y_min -= y_pad
    y_max += y_pad

    W, H, pad_l, pad_b, pad_t, pad_r = 620, 260, 50, 40, 20, 20
    chart_w, chart_h = W - pad_l - pad_r, H - pad_t - pad_b
    n = len(labels)

    def xpos(i):
        return pad_l + (chart_w * i / (n - 1) if n > 1 else 0)

    def ypos(val):
        return pad_t + chart_h * (1 - (val - y_min) / (y_max - y_min))

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
           f'aria-label="Delta PESQ-NB por bucket de SNR">']
    # eje cero
    zero_y = ypos(0)
    svg.append(f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{W-pad_r}" y2="{zero_y:.1f}" '
                f'class="zero-line"/>')
    # gridlines horizontales + etiquetas del eje Y (antes no se veían)
    for gy_val in [y_min + y_pad, 0, y_max - y_pad]:
        gy = ypos(gy_val)
        svg.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W-pad_r}" y2="{gy:.1f}" class="gridline"/>')
        sign = "+" if gy_val > 0 else ""
        svg.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                    f'class="axis-label">{sign}{gy_val:.2f}</text>')
    svg.append(f'<text x="14" y="{H/2:.1f}" text-anchor="middle" class="axis-label" '
                f'transform="rotate(-90 14 {H/2:.1f})">Δ PESQ-NB</text>')

    for v, vals in series.items():
        color = colors[v]
        pts = " ".join(f"{xpos(i):.1f},{ypos(val):.1f}" for i, val in enumerate(vals))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>')
        for i, val in enumerate(vals):
            svg.append(f'<circle cx="{xpos(i):.1f}" cy="{ypos(val):.1f}" r="4" fill="{color}" '
                        f'stroke="var(--surface-1)" stroke-width="2"/>')
        # etiqueta al final de la línea
        last_x, last_y = xpos(n - 1), ypos(vals[-1])
        svg.append(f'<text x="{last_x + 8:.1f}" y="{last_y + 4:.1f}" class="line-end-label" '
                    f'fill="{color}">{VARIANT_LABEL[v]}</text>')

    for i, lab in enumerate(labels):
        svg.append(f'<text x="{xpos(i):.1f}" y="{H - 10}" text-anchor="middle" '
                    f'class="axis-label">{lab}</text>')
    svg.append("</svg>")
    return "".join(svg)


def build_table(data, ts):
    metrics_order = [("pesq_nb", "PESQ-NB"), ("pesq_wb", "PESQ-WB"),
                      ("stoi", "STOI"), ("sisdr", "SI-SDR (dB)")]
    rows = []
    for v in VARIANT_ORDER:
        if v not in data or ts not in data[v]:
            continue
        d = data[v][ts]
        cells = [f"<td class='varcell'>{VARIANT_LABEL[v]}</td>"]
        for mk, _ in metrics_order:
            g = d["global"][mk]
            val = g["noisy_mean"] if v == "noisy" else g["est_mean"]
            delta = None if v == "noisy" else g["delta_mean"]
            fmt = "{:.3f}" if mk == "stoi" else "{:.3f}"
            cell = fmt.format(val)
            if delta is not None:
                sign = "+" if delta >= 0 else ""
                cls = "delta-pos" if delta >= 0 else "delta-neg"
                cell += f' <span class="{cls}">({sign}{delta:.3f})</span>'
            cells.append(f"<td>{cell}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "<tr><th>Variante</th>" + "".join(f"<th>{m}</th>" for _, m in metrics_order) + "</tr>"
    return f"<table>{header}{''.join(rows)}</table>"


def main():
    data = load_results()
    sets_present = [t for t in TEST_SETS if any(t in v for v in data.values())]
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    set_sections = []
    for ts in sets_present:
        ts_data = {v: d[ts] for v, d in data.items() if ts in d}

        panels = []
        for mk, mlabel, unit, fixed_max, grid_step in METRICS:
            panels.append(svg_bar_panel(mk, mlabel, unit, fixed_max, grid_step, ts_data,
                                         COLORS["light"], f"{ts}-{mk}"))

        bucket_chart = svg_bucket_line(ts_data, COLORS["light"])
        table_html = build_table(data, ts)

        legend_items = "".join(
            f'<span class="legend-item"><span class="swatch" style="background:{COLORS["light"][v]}"></span>{VARIANT_LABEL[v]}</span>'
            for v in VARIANT_ORDER if v in ts_data
        )

        set_sections.append(f"""
        <section class="lang-section" data-lang="{ts}">
          <h2>{TEST_SET_LABEL[ts]}</h2>
          <p class="set-sub">{TEST_SET_SUB[ts]}</p>
          <div class="legend">{legend_items}</div>
          <div class="panel-grid">{"".join(panels)}</div>
          <h3>Por bucket de SNR (ΔPESQ-NB vs noisy)</h3>
          <div class="panel">{bucket_chart}</div>
          <h3>Tabla completa</h3>
          {table_html}
        </section>
        """)

    timeline_html = "".join(
        f'<div class="tl-item"><div class="tl-tag">{tag}</div><div class="tl-date">{date}</div>'
        f'<div class="tl-desc">{desc}</div></div>'
        for tag, date, desc in TIMELINE
    )

    wiki_html = "".join(
        f'<div class="wiki-card"><h3>{name}</h3><p class="wiki-full">{full}</p>'
        f'<p>{desc}</p><p class="wiki-ref">Ref: {ref}</p></div>'
        for name, full, desc, ref in METRIC_WIKI
    )

    tabs_html = "".join(
        f'<button class="tab-btn" data-lang="{ts}">{TEST_SET_LABEL[ts]}</button>'
        for ts in sets_present
    )

    chip = {"positivo": "resultado positivo", "negativo": "resultado negativo",
            "mixto": "resultado mixto"}
    experiments_html = "".join(
        f'<article class="exp-card">'
        f'<div class="exp-head"><h3>{tag} — {title}</h3>'
        f'<span class="chip chip-{kind}">{chip[kind]}</span></div>'
        f'<p><span class="exp-k">La pregunta</span> {question}</p>'
        f'<p><span class="exp-k">El diseño</span> {design}</p>'
        f'<p><span class="exp-k">El resultado</span> {result}</p>'
        f'<p><span class="exp-k">Qué queda</span> {verdict}</p>'
        f'</article>'
        for tag, title, kind, question, design, result, verdict in EXPERIMENTS
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>NoiseSuppressNet — Panel de resultados</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --gridline: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --success: #006300;
  --danger: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --success: #0ca30c;
    --danger: #e66767;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}}
header {{ padding: 32px 24px 16px; max-width: 1100px; margin: 0 auto; }}
header h1 {{ margin: 0 0 4px; font-size: 1.6rem; }}
header p {{ margin: 0; color: var(--text-secondary); font-size: 0.9rem; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 64px; }}
.tabs {{ display: flex; gap: 8px; margin: 16px 0 24px; }}
.tab-btn {{
  padding: 8px 18px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--text-primary); font-size: 0.9rem;
  cursor: pointer; font-weight: 600;
}}
.tab-btn.active {{ background: #2a78d6; color: #fff; border-color: #2a78d6; }}
.lang-section {{ display: none; }}
.lang-section.active {{ display: block; }}
h2 {{ font-size: 1.25rem; margin: 8px 0 12px; }}
h3 {{ font-size: 1.02rem; margin: 28px 0 10px; color: var(--text-secondary); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 16px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: var(--text-secondary); }}
.swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
.panel-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
.panel {{
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; overflow-x: auto;
}}
.panel h4 {{ margin: 0 0 10px; font-size: 0.9rem; color: var(--text-secondary); }}
.bar-label {{ font-size: 12px; fill: var(--text-primary); }}
.bar-value {{ font-size: 12px; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }}
.gridline {{ stroke: var(--gridline); stroke-width: 1; }}
.zero-line {{ stroke: var(--muted); stroke-width: 1; }}
.axis-label {{ font-size: 10px; fill: var(--muted); }}
.line-end-label {{ font-size: 11px; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
th, td {{ padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--gridline);
  font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.varcell {{ font-weight: 600; }}
.delta-pos {{ color: var(--success); }}
.delta-neg {{ color: var(--danger); }}
.tl-item {{ display: grid; grid-template-columns: 60px 140px 1fr; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--gridline); align-items: baseline; }}
.tl-tag {{ font-weight: 700; color: #2a78d6; }}
.tl-date {{ color: var(--muted); font-size: 0.85rem; }}
.tl-desc {{ color: var(--text-secondary); font-size: 0.9rem; }}
.wiki-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px; margin-bottom: 14px; }}
.wiki-card h3 {{ margin: 0 0 2px; color: var(--text-primary); font-size: 1.05rem; }}
.wiki-full {{ margin: 0 0 8px; color: var(--muted); font-size: 0.82rem; font-style: italic; }}
.wiki-ref {{ margin: 8px 0 0; color: var(--muted); font-size: 0.78rem; }}
code {{ background: var(--gridline); padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
.set-sub {{ color: var(--text-secondary); font-size: 0.88rem; margin: -6px 0 16px; }}
.section-sub {{ color: var(--text-secondary); font-size: 0.9rem; margin: 0 0 18px; max-width: 78ch; }}
.exp-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px 20px; margin-bottom: 14px; }}
.exp-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin-bottom: 10px; }}
.exp-card h3 {{ margin: 0; color: var(--text-primary); font-size: 1.05rem; }}
.exp-card p {{ margin: 0 0 9px; font-size: 0.9rem; color: var(--text-secondary); max-width: 82ch; }}
.exp-card p:last-child {{ margin-bottom: 0; }}
.exp-k {{ display: block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); font-weight: 600; margin-bottom: 1px; }}
.chip {{ font-size: 0.72rem; font-weight: 600; padding: 2px 9px; border-radius: 999px;
  border: 1px solid currentColor; white-space: nowrap; }}
.chip-positivo {{ color: var(--success); }}
.chip-negativo {{ color: var(--danger); }}
.chip-mixto {{ color: var(--text-secondary); }}
@media (max-width: 640px) {{
  .tabs {{ flex-wrap: wrap; }}
  .tl-item {{ grid-template-columns: 1fr; gap: 2px; }}
}}
footer {{ max-width: 1100px; margin: 0 auto; padding: 24px; color: var(--muted); font-size: 0.8rem;
  border-top: 1px solid var(--gridline); }}
</style>
</head>
<body>
<header>
  <h1>NoiseSuppressNet — Panel de resultados</h1>
  <p>Generado {generated_at} · PFG Ing. Electrónica, UTN FRBA · Gabriel Guzmán Anglese</p>
</header>
<main>
  <div class="tabs">{tabs_html}</div>
  {"".join(set_sections)}

  <h2 style="margin-top:48px">Los experimentos, uno por uno</h2>
  <p class="section-sub">Cada variante responde una pregunta y cambia <strong>una sola cosa</strong> respecto de la
  anterior. Los resultados negativos están acá con el mismo detalle que los positivos: son
  la mitad de la evidencia.</p>
  {experiments_html}

  <h2 style="margin-top:48px">Línea de tiempo del proyecto</h2>
  <div class="timeline">{timeline_html}</div>

  <h2 style="margin-top:48px">Wiki de métricas</h2>
  {wiki_html}
</main>
<footer>
  Regenerar este panel: <code>python -m scripts.generate_dashboard</code> desde la raíz del repo,
  después de evaluar una variante nueva. Los gráficos y la tabla se leen de <code>results/*.json</code>
  automáticamente; las fichas de experimento y la línea de tiempo son texto en el generador y hay que
  editarlas a mano cuando cambia un número.
</footer>
<script>
const tabs = document.querySelectorAll('.tab-btn');
const sections = document.querySelectorAll('.lang-section');
function activate(lang) {{
  tabs.forEach(t => t.classList.toggle('active', t.dataset.lang === lang));
  sections.forEach(s => s.classList.toggle('active', s.dataset.lang === lang));
}}
tabs.forEach(t => t.addEventListener('click', () => activate(t.dataset.lang)));
if (tabs.length) activate(tabs[0].dataset.lang);
</script>
</body>
</html>
"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard generado: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Variantes graficadas: {sorted(data.keys())}")
    print(f"Sellados: {sets_present}")
    print(f"Fichas de experimento: {len(EXPERIMENTS)}")


if __name__ == "__main__":
    main()
