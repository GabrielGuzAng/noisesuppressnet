"""
analysis/plot_gate_figures.py

The two figures of the gate (V6/V7) that the dashboard's bar charts cannot show.

Why they exist. The dashboard deliberately plots only the main ablation line
(noisy, V1, V2, V3, V3b, V3e, V5) because V6 and V7 are treatment/control designs
whose estimand is a paired contrast against their own control, not an absolute
value comparable to the other variants. Three reasons, any one of them enough:

  1. V7's control is NOT V2. V7 trained from scratch with its own control, so
     putting V7-gate's absolute PESQ next to V2's confounds the gate with
     everything else that differs between those runs (V2 ran 20 epochs, V1 ran 30).
  2. Different estimator. V1..V5 are ONE checkpoint selected by val_loss; V6/V7 are
     the mean of 18 (3 seeds x 6 epochs), precisely because of the per-checkpoint
     noise floor that V6 measured (sd 0.0099 es / 0.0166 en).
  3. V7 trained on English only, so on a Spanish sealed set it is an out-of-domain
     model while V3e and V5 are adapted ones. Not peers.

So the gate needs its own two figures, each matching what it actually measures:

  Figure A -- delta vs the unprocessed input, by SNR bucket, on test_v2_es.
    Shows the pathology and its mitigation in one panel: the from-scratch control
    degrades clean high-SNR Spanish speech, the gate crosses zero, and language
    adaptation (V5) solves it with a much larger margin. The mixed estimators are
    legitimate here because the claim rides on the DIRECTION (the zero crossing),
    not the precision -- and the asymmetry is made visible: V7's lines carry a
    min-max band across seeds, the single-checkpoint variants carry none.

  Figure B -- the preregistered paired contrast (gate - control) by seed and
    sealed set, against the +0.050 screening threshold. This is the figure that
    matches the estimand: it shows that the sign replicates 9/9 while seed 45
    falls below the threshold, with the dispersion visible instead of hidden
    behind a mean.

Palette. Reuses the project's own tokens, revalidated for this figure with the
dataviz validator at all-pairs (not just adjacent): V7 takes the `--externo`
purple (#4a3aa7 light / #9085e9 dark) and V5 keeps its green (#008300 in both).
Worst CVD separation 26.9 dE, all six checks pass in both modes. V2 was dropped
from Figure A for two reasons that coincide: it is redundant with V7-control for
the pathology claim, and #eb6834 vs #008300 fails protanopia at dE 3.2.

Outputs (figures/):
    fig_gate_buckets.pdf   .svg   .png
    fig_gate_contrast.pdf  .svg   .png

The .pdf is versioned (figures/*.png and *.svg are gitignored) because it is the
artifact the paper includes. The .svg is rewritten with CSS custom properties so
the dashboard can inline it and have it follow the page theme.

Usage:
    python -m analysis.plot_gate_figures
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
SWEEP = RESULTS / "v7_epochs"
FIGURES = PROJECT_ROOT / "figures"

EPOCHS = [15, 16, 17, 18, 19, 20]          # el tramo del estimando preregistrado
CONFIRMATORY = [43, 44, 45]                # la 42 generó la hipótesis: se reporta, no entra
SCREENING = 42
THRESHOLD = 0.050                          # umbral de screening preregistrado de V7
SEALED = ["v2_es", "v1_en", "v3_mls_es"]
SEALED_LABEL = {
    "v2_es": "test_v2_es\n(español, crowdsourced)",
    "v1_en": "test_v1_en\n(inglés, audiolibro)",
    "v3_mls_es": "test_v3_mls_es\n(español, audiolibro)",
}

# Tokens del proyecto. Validados a all-pairs con scripts/validate_palette.js.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
C_GATE = "#4a3aa7"      # --externo, el par control/compuerta de V7
C_V5 = "#008300"        # --c-v5
C_THRESHOLD = "#eda100"  # --bloqueado, sólo para la línea de umbral

# Para la copia temable del SVG: hex -> custom property con fallback.
CSS_VARS = {
    C_GATE: "var(--c-gate, #4a3aa7)",
    C_V5: "var(--c-v5, #008300)",
    C_THRESHOLD: "var(--c-v3b, #eda100)",
    INK: "var(--text-primary, #0b0b0b)",
    MUTED: "var(--muted, #898781)",
    GRID: "var(--gridline, #e1e0d9)",
}


# ─────────────────────────── lectura de datos ───────────────────────────

def _sweep_bucket(branch: str, test_set: str, bucket: int) -> float:
    """Mean ΔPESQ-NB of `branch` in `bucket`, averaged over the declared epochs."""
    vals = []
    for ep in EPOCHS:
        path = SWEEP / f"{branch}_ep{ep}_{test_set}.json"
        data = json.loads(path.read_text())
        vals.append(next(b["pesq_nb_delta_mean"] for b in data["by_bucket"]
                         if b["bucket_idx"] == bucket))
    return st.mean(vals)


def _variant_bucket(variant: str, test_set: str, bucket: int) -> float:
    """ΔPESQ-NB of a single-checkpoint variant in `bucket`."""
    data = json.loads((RESULTS / f"{variant}_{test_set}.json").read_text())
    return next(b["pesq_nb_delta_mean"] for b in data["by_bucket"]
                if b["bucket_idx"] == bucket)


def bucket_ranges(test_set: str = "v2_es") -> list:
    data = json.loads((SWEEP / f"v7_gate_ep15_{test_set}.json").read_text())
    return [b["snr_range_db"] for b in sorted(data["by_bucket"], key=lambda b: b["bucket_idx"])]


def seed_contrast(seed: int, test_set: str) -> float:
    """Gate − control, averaged over the declared epoch range, for one seed."""
    sfx = "" if seed == SCREENING else f"_s{seed}"
    gate = st.mean([_sweep_bucket(f"v7_gate{sfx}", test_set, b) for b in range(5)])
    ctrl = st.mean([_sweep_bucket(f"v7_control{sfx}", test_set, b) for b in range(5)])
    # el estimando es sobre el global, no el promedio de buckets: se relee del JSON
    g = c = 0.0
    for ep in EPOCHS:
        g += json.loads((SWEEP / f"v7_gate{sfx}_ep{ep}_{test_set}.json").read_text()
                        )["global"]["pesq_nb"]["delta_mean"]
        c += json.loads((SWEEP / f"v7_control{sfx}_ep{ep}_{test_set}.json").read_text()
                        )["global"]["pesq_nb"]["delta_mean"]
    return (g - c) / len(EPOCHS)


# ─────────────────────────── estilo común ───────────────────────────

def _base_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def _save(fig, stem: str):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None, transparent=True)
    plt.close(fig)
    # copia temable: el SVG del dashboard usa custom properties del propio panel
    svg = FIGURES / f"{stem}.svg"
    text = svg.read_text()
    for hex_value, css in CSS_VARS.items():
        text = text.replace(hex_value, css).replace(hex_value.upper(), css)
    (FIGURES / f"{stem}.themed.svg").write_text(text)
    print(f"  {stem}: pdf · png · svg · themed.svg")


# ─────────────────────────── figura A ───────────────────────────

def figure_buckets(test_set: str = "v2_es"):
    """Delta vs the unprocessed input, by SNR bucket. The pathology and its fix."""
    ranges = bucket_ranges(test_set)
    x = list(range(5))
    labels = [f"[{a:g}, {b:g}]" for a, b in ranges]

    def band(role):
        per_seed = [[_sweep_bucket(f"v7_{role}_s{s}", test_set, b) for b in x]
                    for s in CONFIRMATORY]
        mean = [st.mean(vals) for vals in zip(*per_seed)]
        lo = [min(vals) for vals in zip(*per_seed)]
        hi = [max(vals) for vals in zip(*per_seed)]
        return mean, lo, hi

    ctrl, ctrl_lo, ctrl_hi = band("control")
    gate, gate_lo, gate_hi = band("gate")
    v5 = [_variant_bucket("v5", test_set, b) for b in x]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _base_style(ax)

    # el cero es la referencia que carga la afirmación: por encima mejora, por debajo degrada
    ax.axhline(0, color=MUTED, linewidth=1.4, zorder=1)

    ax.fill_between(x, ctrl_lo, ctrl_hi, color=C_GATE, alpha=0.13, linewidth=0, zorder=2)
    ax.fill_between(x, gate_lo, gate_hi, color=C_GATE, alpha=0.13, linewidth=0, zorder=2)

    ax.plot(x, v5, color=C_V5, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.plot(x, ctrl, color=C_GATE, linewidth=2, linestyle=(0, (5, 2)), marker="s",
            markersize=5, markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.plot(x, gate, color=C_GATE, linewidth=2.4, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    # etiquetas directas: tres series, así que van todas (y además hay leyenda)
    for txt, ys, col, dy in [("V5", v5[4], C_V5, 0.0),
                             ("V7 compuerta", gate[4], C_GATE, 0.030),
                             ("V7 control", ctrl[4], C_GATE, -0.030)]:
        ax.annotate(txt, xy=(4.10, ys + dy), fontsize=9, color=col,
                    weight="600", va="center", ha="left", annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, 5.5)
    ax.set_ylim(-0.12, 0.78)
    ax.set_xlabel("bucket de SNR de la mixtura (dB)", fontsize=9.5, color=INK, labelpad=8)
    ax.set_ylabel("Δ PESQ-NB contra el audio sin procesar", fontsize=9.5, color=INK, labelpad=8)
    ax.set_title("La degradación de habla limpia, y qué la corrige",
                 fontsize=11.5, color=INK, weight="600", loc="left", pad=26)
    ax.annotate(f"sobre {test_set} · banda = mín-máx entre las semillas 43/44/45 · "
                f"V5 es un checkpoint único, sin réplicas",
                xy=(0, 1.035), xycoords="axes fraction", fontsize=8.5, color=MUTED,
                va="bottom", ha="left")

    handles = [
        Line2D([], [], color=C_GATE, lw=2.4, marker="o", ms=6, mec="white",
               label="V7 con compuerta · sólo inglés, desde cero"),
        Line2D([], [], color=C_GATE, lw=2, ls=(0, (5, 2)), marker="s", ms=5, mec="white",
               label="V7 control · idéntico, sin compuerta"),
        Line2D([], [], color=C_V5, lw=2, marker="o", ms=5, mec="white",
               label="V5 · fine-tuneado a español"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.8,
                    handlelength=2.6, borderaxespad=0.2, labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)
    _save(fig, "fig_gate_buckets")


# ─────────────────────────── figura B ───────────────────────────

def figure_contrast():
    """The preregistered paired contrast, by seed and sealed set."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _base_style(ax)

    ax.axhline(0, color=MUTED, linewidth=1.4, zorder=1)
    ax.axhline(THRESHOLD, color=C_THRESHOLD, linewidth=1.6,
               linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"umbral preregistrado  +{THRESHOLD:.3f}", xy=(-0.52, THRESHOLD - 0.0030),
                fontsize=8.8, color=C_THRESHOLD, weight="600", va="top", ha="left")

    offs = [-0.10, 0.06, 0.22]
    for i, ts in enumerate(SEALED):
        conf = [seed_contrast(s, ts) for s in CONFIRMATORY]
        scr = seed_contrast(SCREENING, ts)
        mean = st.mean(conf)

        for dx, val, seed in zip(offs, conf, CONFIRMATORY):
            ax.plot(i + dx, val, marker="o", markersize=8.5, markerfacecolor="none",
                    markeredgecolor=C_GATE, markeredgewidth=1.8, zorder=4)
            ax.annotate(str(seed), xy=(i + dx, val - 0.0055), fontsize=7.5,
                        color=MUTED, va="top", ha="center")
        ax.plot(i + 0.38, scr, marker="X", markersize=9, color=MUTED, zorder=3)
        ax.annotate("42", xy=(i + 0.38, scr - 0.0055), fontsize=7.5, color=MUTED,
                    va="top", ha="center")
        ax.plot(i - 0.30, mean, marker="D", markersize=10, color=C_GATE, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(f"{mean:+.4f}", xy=(i - 0.30, mean + 0.0045), fontsize=9,
                    color=C_GATE, weight="600", va="bottom", ha="center")

    ax.set_xticks(range(len(SEALED)))
    ax.set_xticklabels([SEALED_LABEL[t] for t in SEALED], fontsize=9)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(-0.004, 0.094)
    ax.set_ylabel("contraste  compuerta − control  (Δ PESQ-NB)", fontsize=9.5,
                  color=INK, labelpad=8)
    ax.set_title("El estimando confirmatorio, con su dispersión a la vista",
                 fontsize=11.5, color=INK, weight="600", loc="left", pad=26)
    ax.annotate("media de las épocas 15-20 · el rombo es la media de 43/44/45, "
                "que es el estimando · la 42 disparó la confirmación y no entra",
                xy=(0, 1.035), xycoords="axes fraction", fontsize=8.5, color=MUTED,
                va="bottom", ha="left")

    handles = [
        Line2D([], [], color=C_GATE, marker="D", ms=10, ls="none", mec="white",
               label="estimando confirmatorio (media de 43/44/45)"),
        Line2D([], [], marker="o", ms=8.5, ls="none", mfc="none", mec=C_GATE,
               mew=1.8, label="semilla individual"),
        Line2D([], [], color=MUTED, marker="X", ms=9, ls="none",
               label="semilla 42 · screening, excluida del estimando"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                    frameon=False, fontsize=8.8, handlelength=1.6, ncol=3,
                    columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(INK)
    _save(fig, "fig_gate_contrast")


if __name__ == "__main__":
    print("Figuras de la compuerta (V6/V7):")
    figure_buckets()
    figure_contrast()
    print(f"\nEscritas en {FIGURES}/")
    print("El .pdf va al paper (queda versionado); el .themed.svg lo inlinea el dashboard.")
