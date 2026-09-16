"""
tests/test_f6_channel_control.py

Capa 9: cobertura de la familia F6 (control de idioma vs canal).

F6 se escribió sin la separación implementador/validador que sí tuvieron F1-F5,
y una revisión posterior encontró tres defectos: la regla de decisión de P1
contradecía el preregistro, el p-value sin agrupar se reportaba junto al IC
agrupado sin marca, y F6 no se renderizaba en el reporte Markdown. Estos tests
cierran esa deuda.

Principio, el mismo del resto de la suite: cada número se recomputa desde los
JSON crudos con numpy/scipy plano y se compara contra lo que el módulo escribió.
Prohibido usar las funciones del módulo para construir la verdad de referencia.

El control más fuerte que tienen estos tests es que la regla de decisión que
verifican no salió de la implementación: está fijada en el preregistro, escrito
antes de que el test set existiera (hash en docs/preregistro_mls_es.sha256).

LO QUE ESTA CAPA NO CUBRE. Declarado acá y no en el historial porque su lector
natural es quien abra este archivo para extenderlo.

1. El IC95 agrupado no se recomputa de forma independiente. Se verifican
   propiedades del bootstrap (más ancho que el ingenuo con estructura de
   cluster, equivalente al ingenuo con clusters unitarios, determinista,
   reporta n_clusters) pero no el valor [-0.309, -0.115], del que cuelga el
   desenlace de P1: la regla pide rho <= -0.15 Y ci_hi < 0. Un error sutil en
   el remuestreo cumpliría las cuatro propiedades y daría el mismo veredicto.
   Se arregla con dos tests: uno que reimplemente el bootstrap con la misma
   semilla y reproduzca los bordes, y otro que compare contra un método
   distinto (jackknife por bloques sobre hablantes) exigiendo acuerdo
   aproximado.

2. No hay test sobre la premisa del sellado: que las condiciones de ruido de
   test_v3_mls_es son las de test_v1_en verbatim. Se verificó a mano contra
   las dos metadatas y se cumple par por par -- noise_file, noise_offset,
   snr_db y noise_category, 250/250. No es una propiedad que se cumpla sola:
   el mismo chequeo entre test_v2_es y test_v1_en da snr_db 250/250 pero
   noise_file 18/250. Ese reuso total es lo que hace de MLS un control de
   canal limpio y de la comparación una comparación apareada; si algún día se
   rompe, P1 y P2 miden otra cosa y estos 32 tests siguen verdes.

3. El chequeo del preregistro es un substring sobre un nombre de archivo
   (assert "sha256" in "docs/preregistro_mls_es.sha256"): pasaría con el
   archivo borrado. Debería validar el contenido del .sha256 (digests de 64
   hex bien formados) y que el archivo del oráculo no esté dentro del repo.

4. ci95["point"] no se contrasta contra el rho reportado, siendo que son el
   mismo número por construcción. El estadístico se pasa como lambda que
   captura x e y del loop de los tres test sets: hoy funciona porque se
   invoca antes de que el loop avance, pero es frágil ante un refactor.

5. retained no tiene chequeo de premisa: la razón solo tiene sentido con
   gain_crowdsourced > 0. Este hueco dejó de ser hipotético -- la evaluación
   de V2 sobre test_v3_mls_es mostró que el estimando de retention_by_recipe
   está mal planteado.

VERIFICADO Y DESCARTADO COMO PROBLEMA: clustered_bootstrap_ci descarta en
silencio las muestras no finitas, con lo que el n_resamples reportado podría no
ser el real. Sobre el dato de F6, 0 descartadas de 10.000, y todos los hablantes
tienen 6 o 7 pares.

Uso:
    python -m pytest tests/test_f6_channel_control.py -v
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import analysis.reanalysis_stats as rs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
METADATA = PROJECT_ROOT / "seal_test_metadata"
CONTROL_SET = "v3_mls_es"


# ── Utilidades de verdad de referencia: leen JSON crudo, nada del módulo ────

def raw_pairs(variant: str, test_set: str) -> dict[int, dict]:
    with open(RESULTS / f"{variant}_{test_set}.json") as f:
        return {r["pair_id"]: r for r in json.load(f)["all_pairs"]}


def raw_speakers() -> dict[int, str]:
    with open(METADATA / "test_v3_mls_es_metadata.json") as f:
        return {p["id"]: str(p["speaker_id"]) for p in json.load(f)["pairs"]}


@pytest.fixture(scope="module")
def payload() -> dict:
    with open(RESULTS / "reanalysis_stats.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def f6(payload) -> dict:
    assert "F6" in payload["families"], "F6 no está en el payload"
    return payload["families"]["F6"]


# ── 9.1 · Reglas de decisión contra el preregistro ─────────────────────────
# Es el test de mayor valor de la capa: la regla no sale de la implementación,
# sale del preregistro, y acá se verifica cada rama por separado.

@pytest.mark.parametrize("rho,ci_hi,expected", [
    (-0.212, -0.115, "idioma"),                    # el caso observado
    (-0.400, -0.300, "idioma"),                    # efecto grande y claro
    (-0.150, -0.001, "idioma"),                    # justo en el umbral
    (-0.200, +0.020, "indeterminado por potencia"),  # el bug que se corrigió
    (-0.900, +0.100, "indeterminado por potencia"),  # rho enorme, IC inútil
    (-0.050, -0.010, "canal"),
    (-0.100, -0.010, "canal"),                     # justo en el umbral de canal
    (+0.300, +0.400, "canal"),
    (-0.120, -0.030, "indeterminado"),             # tierra de nadie
    (-0.140, -0.001, "indeterminado"),
])
def test_p1_decision_rule_matches_preregistration(rho, ci_hi, expected):
    assert rs.p1_outcome(rho, ci_hi) == expected


def test_p1_never_returns_canal_when_rho_is_below_language_threshold():
    """La regla que el bug violaba: rho <= -0.15 NUNCA puede dar 'canal'."""
    rng = np.random.default_rng(0)
    for rho in rng.uniform(-1.0, rs.P1_LANGUAGE_RHO, size=200):
        for ci_hi in rng.uniform(-0.5, 0.5, size=5):
            assert rs.p1_outcome(float(rho), float(ci_hi)) != "canal"


@pytest.mark.parametrize("mean_gain,prop,expected", [
    (0.084, 0.748, "parcial"),          # el caso observado
    (0.221, 0.890, "idioma"),
    (0.150, 0.710, "idioma"),
    (0.020, 0.900, "canal"),            # media por debajo del piso
    (0.300, 0.500, "canal"),            # proporción por debajo del piso
    (0.080, 0.600, "parcial"),
])
def test_p2_decision_rule_matches_preregistration(mean_gain, prop, expected):
    assert rs.p2_outcome(mean_gain, prop) == expected


def test_thresholds_are_the_preregistered_ones():
    """Si alguien los mueve, el preregistro deja de valer y hay que saberlo."""
    assert rs.P1_LANGUAGE_RHO == -0.15
    assert rs.P1_CHANNEL_RHO == -0.10
    assert rs.P2_LANGUAGE_MEAN == 0.10
    assert rs.P2_LANGUAGE_PROP == 0.70
    assert rs.P2_CHANNEL_MEAN == 0.03
    assert rs.P2_CHANNEL_PROP == 0.55


# ── 9.2 · El bootstrap agrupa por hablante, no por par ─────────────────────

def test_clustered_bootstrap_resamples_speakers_not_pairs():
    """Con valores idénticos dentro de cada hablante, agrupar tiene que dar un
    intervalo MUCHO más ancho que remuestrear pares: el N efectivo es el número
    de hablantes, no el de observaciones."""
    n_speakers, per_speaker = 8, 30
    rng = np.random.default_rng(7)
    per_speaker_value = rng.normal(0.0, 1.0, size=n_speakers)
    values = np.repeat(per_speaker_value, per_speaker)
    clusters = np.repeat([f"spk{i}" for i in range(n_speakers)], per_speaker)

    grouped = rs.clustered_bootstrap_ci(
        values, clusters, lambda idx: float(values[idx].mean()),
        seed=42, n_resamples=2000)
    naive = rs.bootstrap_ci(values, np.mean, seed=42, n_resamples=2000)

    width_grouped = grouped["hi"] - grouped["lo"]
    width_naive = naive["hi"] - naive["lo"]
    assert width_grouped > 3 * width_naive, (
        f"El IC agrupado ({width_grouped:.3f}) no es sustancialmente más ancho "
        f"que el de pares ({width_naive:.3f}): probablemente no está agrupando"
    )


def test_clustered_bootstrap_with_singleton_clusters_matches_pair_level():
    """Si cada par es su propio cluster, agrupar debe dar casi lo mismo que no."""
    rng = np.random.default_rng(11)
    values = rng.normal(0.5, 1.0, size=300)
    clusters = np.array([f"c{i}" for i in range(300)])
    grouped = rs.clustered_bootstrap_ci(
        values, clusters, lambda idx: float(values[idx].mean()),
        seed=42, n_resamples=4000)
    width_grouped = grouped["hi"] - grouped["lo"]
    naive_width = 2 * 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    assert abs(width_grouped - naive_width) < 0.25 * naive_width


def test_clustered_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    clusters = np.repeat([f"s{i}" for i in range(20)], 10)
    stat = lambda idx: float(values[idx].mean())
    a = rs.clustered_bootstrap_ci(values, clusters, stat, seed=42, n_resamples=1000)
    b = rs.clustered_bootstrap_ci(values, clusters, stat, seed=42, n_resamples=1000)
    assert a == b


def test_clustered_bootstrap_reports_cluster_count_and_method():
    values = np.arange(100, dtype=float)
    clusters = np.repeat([f"s{i}" for i in range(10)], 10)
    out = rs.clustered_bootstrap_ci(values, clusters, lambda idx: float(values[idx].mean()))
    assert out["n_clusters"] == 10
    assert "hablante" in out["method"]
    assert out["lo"] <= out["point"] <= out["hi"]


# ── 9.3 · P1 recomputado desde los JSON crudos ─────────────────────────────

def test_p1_slopes_match_independent_recomputation(f6):
    for slope in f6["p1"]["slopes"]:
        pairs = raw_pairs("v1", slope["test_set"])
        ids = sorted(pairs)
        x = np.array([pairs[i]["snr_db"] for i in ids], dtype=np.float64)
        y = np.array([pairs[i]["pesq_nb_delta"] for i in ids], dtype=np.float64)
        expected = stats.spearmanr(x, y)
        assert slope["rho"] == pytest.approx(expected.correlation, abs=1e-12)
        assert slope["n"] == len(ids)
        reported_p = slope.get("p_value", slope.get("p_value_unclustered"))
        assert reported_p == pytest.approx(expected.pvalue, rel=1e-9)


def test_only_the_control_set_carries_a_clustered_ci(f6):
    """El IC agrupado corresponde solo al set donde los pares no son independientes."""
    for slope in f6["p1"]["slopes"]:
        if slope["test_set"] == CONTROL_SET:
            assert "ci95" in slope
            assert slope["ci95"]["lo"] < slope["ci95"]["hi"]
        else:
            assert "ci95" not in slope


def test_unclustered_p_is_flagged_on_the_control_set(f6):
    """Bug corregido: el p sobre-confiado no puede ir sin marca junto al IC."""
    control = next(s for s in f6["p1"]["slopes"] if s["test_set"] == CONTROL_SET)
    assert "p_value" not in control, (
        "El p sin agrupar no debe llamarse 'p_value' en el set de control: "
        "un lector lo tomaría como válido"
    )
    assert "p_value_unclustered" in control
    assert "sobre-confiado" in control["p_value_note"]


def test_p1_outcome_follows_from_its_own_numbers(f6):
    control = next(s for s in f6["p1"]["slopes"] if s["test_set"] == CONTROL_SET)
    assert f6["p1"]["outcome"] == rs.p1_outcome(control["rho"], control["ci95"]["hi"])


# ── 9.4 · P2 recomputado desde los JSON crudos ─────────────────────────────

def test_p2_gain_and_proportion_match_recomputation(f6):
    v1_mls, v3e_mls = raw_pairs("v1", CONTROL_SET), raw_pairs("v3e", CONTROL_SET)
    ids = sorted(v1_mls)
    gain = np.array([v3e_mls[i]["pesq_nb_est"] - v1_mls[i]["pesq_nb_est"] for i in ids])
    assert f6["p2"]["gain_audiobook"]["point"] == pytest.approx(gain.mean(), abs=1e-12)
    assert f6["p2"]["prop_improve"]["point"] == pytest.approx((gain > 0).mean(), abs=1e-12)

    v1_cv, v3e_cv = raw_pairs("v1", "v2_es"), raw_pairs("v3e", "v2_es")
    cv_ids = sorted(v1_cv)
    gain_cv = np.mean([v3e_cv[i]["pesq_nb_est"] - v1_cv[i]["pesq_nb_est"] for i in cv_ids])
    assert f6["p2"]["gain_crowdsourced"] == pytest.approx(gain_cv, abs=1e-12)


def test_p2_outcome_follows_from_its_own_numbers(f6):
    assert f6["p2"]["outcome"] == rs.p2_outcome(
        f6["p2"]["gain_audiobook"]["point"], f6["p2"]["prop_improve"]["point"])


def test_retention_by_recipe_matches_recomputation(f6):
    v1_mls, v1_cv = raw_pairs("v1", CONTROL_SET), raw_pairs("v1", "v2_es")
    for row in f6["retention_by_recipe"]:
        variant = row["variant"]
        mls, cv = raw_pairs(variant, CONTROL_SET), raw_pairs(variant, "v2_es")
        g_ml = np.mean([mls[i]["pesq_nb_est"] - v1_mls[i]["pesq_nb_est"] for i in sorted(v1_mls)])
        g_cv = np.mean([cv[i]["pesq_nb_est"] - v1_cv[i]["pesq_nb_est"] for i in sorted(v1_cv)])
        assert row["gain_audiobook"] == pytest.approx(g_ml, abs=1e-12)
        assert row["gain_crowdsourced"] == pytest.approx(g_cv, abs=1e-12)
        assert row["retained"] == pytest.approx(g_ml / g_cv, rel=1e-12)


# ── 9.5 · Premisas del agrupamiento y schema ───────────────────────────────

def test_speaker_metadata_covers_every_pair_and_matches_reported_count(f6):
    speakers = raw_speakers()
    pairs = raw_pairs("v1", CONTROL_SET)
    assert set(speakers) == set(pairs), "hay pares sin speaker_id en la metadata"
    assert f6["n_speakers"] == len(set(speakers.values()))
    counts = {}
    for spk in speakers.values():
        counts[spk] = counts.get(spk, 0) + 1
    assert min(counts.values()) >= 2, (
        "Con hablantes de un solo par el agrupamiento no aporta nada; "
        "revisar el diseño del sellado"
    )


def test_f6_schema_and_types(f6):
    for key in ("label", "estimand", "correction", "exploratory", "preregistered",
                "inference", "n_speakers", "p1", "p2", "retention_by_recipe"):
        assert key in f6, f"falta la clave {key}"
    assert f6["correction"] is None, "F6 no lleva Holm: son endpoints preregistrados"
    assert f6["exploratory"] is False
    assert "sha256" in f6["preregistered"]
    assert type(f6["n_speakers"]) is int
    for slope in f6["p1"]["slopes"]:
        assert type(slope["rho"]) is float
    json.dumps(f6)  # falla si quedó algún tipo de numpy


def test_f6_is_rendered_in_the_markdown_report():
    """Bug corregido: F6 estaba en el JSON y no en el reporte legible."""
    text = (PROJECT_ROOT / "docs" / "reanalisis_estadistico.md").read_text(encoding="utf-8")
    assert "F6" in text
    assert "P1" in text and "P2" in text
    assert "agrupado" in text, "el reporte debe decir que la inferencia agrupa por hablante"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-q"]))
