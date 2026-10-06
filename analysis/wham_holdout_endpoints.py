"""
analysis/wham_holdout_endpoints.py

Adjudica los tres contrastes de `test_wham_es` (ruido holdout, WHAM!) contra
el preregistro hasheado en `docs/preregistro_wham_holdout.sha256`.

Por qué es un módulo aparte y no una línea más en `v7_gate_endpoints.TEST_SETS`.
`v7_seeds_endpoints.py` escribe siempre a `results/v7_seeds_endpoints.json`, que
es el registro adjudicado de la confirmación de V7 del 26/09. Agregar un cuarto
sellado a esa constante haría que cualquier corrida posterior reescriba ese
archivo con un panel de cuatro sellados, lo que re-adjudicaría un resultado
cerrado — justamente lo que la sección 11 del preregistro prohíbe. Acá el
intercambio de `TEST_SETS` está dentro de un context manager y la salida va a un
archivo propio, así que la maquinaria se reusa sin tocar el registro.

Estimando primario de cada contraste: la **media entre semillas del contraste
sobre esta sede**, comparada contra un umbral de retención del 50 % del efecto
medido en `v3_mls_es`. Se reporta como diferencia apareada por semilla porque esa
es la forma legible de "cuánto se retuvo", pero el apareo **no reduce la varianza
de la decisión**: como las medias en `v3_mls_es` son constantes ya medidas, el
criterio es algebraicamente un umbral sobre la media cruda en esta sede.

Un borrador de este módulo afirmaba que el apareo bajaba el piso de ruido de
0,0103 a 0,0042 (2,47×). Esa cifra sale del par `v1_en` ↔ `v3_mls_es`, que es el
único de los tres pares de sedes donde el apareo ayuda, y ayuda porque
`test_v3_mls_es` copió el ruido de `test_v1_en` verbatim. En los dos pares que no
comparten ruido el apareo lo empeora (0,55× y 0,39×). Esta sede cambia justamente
el ruido. La afirmación quedó retirada; ver preregistro sección 3.

Intervalos: bootstrap BCa resampleando **grabaciones fuente de WHAM!** (61), con
una sensibilidad por **locación** (19), que es el nivel de anidamiento de arriba.
El estimador puntual es la media a nivel par, que es sobre la que está enunciado
el criterio.

Uso:
    python -m analysis.wham_holdout_endpoints
    python -m analysis.wham_holdout_endpoints --selftest

`--selftest` corre la misma maquinaria sobre el par de sedes que YA existe
(`v1_en` ↔ `v3_mls_es`) y verifica que reproduce los números publicados. No toca
`wham_es` ni necesita que exista: sirve para demostrar, antes de hashear, que
los estimandos declarados son computables con el código que se declara.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import stats

from analysis import v7_gate_endpoints as v7ge
from analysis import v7_seeds_endpoints as v7se

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "results" / "wham_holdout_endpoints.json"

SEDE = "wham_es"
SEDE_REFERENCIA = "v3_mls_es"

SEMILLAS = v7se.SEMILLAS_CONFIRMATORIAS        # (43, 44, 45)
SEMILLA_SCREENING = v7se.SEMILLA_SCREENING     # 42, descriptiva

# Umbrales del preregistro, sección 4. Retención >= 50% del efecto observado
# en la sede de referencia.
C1_REFERENCIA = 0.0441      # contraste compuerta en v3_mls_es
C1_UMBRAL = -C1_REFERENCIA / 2          # sobre d (diferencia apareada)
C1_UMBRAL_SEDE = C1_REFERENCIA / 2      # el mismo umbral, sobre E(sede)
C2_REFERENCIA = 0.0583      # V5(3 semillas) - V2 en v3_mls_es
C2_UMBRAL = -C2_REFERENCIA / 2
C2_UMBRAL_SEDE = C2_REFERENCIA / 2
C3_MAX_RHO = -0.15          # mismo umbral que E3 del preregistro de V7

RAMAS_V5 = ("v5", "v5_s43", "v5_s44")
RAMA_V2 = "v2"


@contextmanager
def _sedes(*test_sets: str):
    """Restrict ``v7ge.TEST_SETS`` to `test_sets` for the duration.

    ``v7se._inventario`` iterates that constant to decide which JSONs are
    required, so narrowing it is what keeps this analysis from demanding the
    full four-venue panel. Restored on the way out, always.
    """
    original = v7ge.TEST_SETS
    v7ge.TEST_SETS = list(test_sets)
    try:
        yield
    finally:
        v7ge.TEST_SETS = original


def _contraste_compuerta(semilla: int, sede: str, referencia: str) -> dict:
    """Gate - control contrast on two venues for one seed, and their difference.

    Args:
        semilla: seed whose branch pair is read.
        sede: the venue under test.
        referencia: the venue it is compared against.

    Returns:
        {"sede", "referencia", "d"} with the two estimands and their paired
        difference d = sede - referencia.

    Raises:
        v7se.DatosFaltantes: if any JSON of either venue is missing.
    """
    with _sedes(sede, referencia):
        faltantes, datos = v7se._inventario(semilla)
        if faltantes:
            raise v7se.DatosFaltantes(
                f"semilla {semilla}: faltan {len(faltantes)} evaluaciones\n  "
                + "\n  ".join(sorted(faltantes)[:8])
            )
        exclusiones = v7se._exclusiones(datos)
        v7se._verificar_panel(exclusiones, semilla)
        with v7se._ramas(semilla, exclusiones):
            a = v7ge.summarize(sede)["estimand"]
            b = v7ge.summarize(referencia)["estimand"]
    return {"sede": a, "referencia": b, "d": a - b}


def _pares(variante: str, test_set: str) -> dict:
    path = PROJECT_ROOT / "results" / f"{variante}_{test_set}.json"
    if not path.exists():
        raise v7se.DatosFaltantes(f"falta {path.relative_to(PROJECT_ROOT)}")
    with open(path) as f:
        return {p["pair_id"]: p for p in json.load(f)["all_pairs"]}


def _contraste_adaptacion(sede: str, referencia: str) -> dict:
    """V5 (three seeds) - V2 on two venues, paired by pair id, and the difference.

    V2 enters with a single checkpoint and V5 with three seeds. The asymmetry is
    declared in the preregistration: neither saved per-epoch checkpoints, and the
    effect is ~22x the per-checkpoint noise floor, unlike the gate's.
    """
    out = {}
    for nombre, ts in (("sede", sede), ("referencia", referencia)):
        v2 = _pares(RAMA_V2, ts)
        por_semilla, excluidos = [], set()
        for rama in RAMAS_V5:
            v5 = _pares(rama, ts)
            ids = sorted(set(v5) & set(v2))
            # Un NaN en cualquiera de las dos ramas propagaba hasta el veredicto:
            # NaN >= umbral es False, o sea "FALLA" en silencio. Se excluye el par
            # de las TRES semillas, para que el panel quede balanceado (seccion 8).
            for i in ids:
                if not (np.isfinite(v5[i]["pesq_nb_est"])
                        and np.isfinite(v2[i]["pesq_nb_est"])):
                    excluidos.add(i)
        for rama in RAMAS_V5:
            v5 = _pares(rama, ts)
            ids = [i for i in sorted(set(v5) & set(v2)) if i not in excluidos]
            por_semilla.append(
                float(np.mean([v5[i]["pesq_nb_est"] - v2[i]["pesq_nb_est"] for i in ids]))
            )
        if not np.all(np.isfinite(por_semilla)):
            raise ValueError(f"{ts}: la adaptacion dio no-finito tras excluir NaN")
        out[nombre] = {"por_semilla": por_semilla,
                       "media": float(np.mean(por_semilla)),
                       "sd": float(np.std(por_semilla, ddof=1)),
                       "pares_excluidos_por_nan": sorted(excluidos),
                       "n_pares": len(set(v5) & set(v2)) - len(excluidos)}
    out["d"] = out["sede"]["media"] - out["referencia"]["media"]
    return out


def _clusters(sede: str) -> tuple[dict, dict]:
    """Map each pair id to its WHAM! source recording and location.

    The 7998 segments of cv+tt come from 61 recordings in 19 locations, so pairs
    are not independent: the resampling unit is the recording, and the location
    is the nesting level above it.

    Raises:
        v7se.DatosFaltantes: if the sealed metadata is missing.
    """
    path = PROJECT_ROOT / "seal_test_metadata" / f"test_{sede}_metadata.json"
    if not path.exists():
        raise v7se.DatosFaltantes(f"falta {path.relative_to(PROJECT_ROOT)}")
    with open(path) as f:
        pairs = json.load(f)["pairs"]
    return ({p["id"]: p["wham_file_id"] for p in pairs},
            {p["id"]: p["wham_location_id"] for p in pairs})


def _contraste_por_par(sede: str) -> dict:
    """Gate - control per pair, averaged over epochs and confirmatory seeds."""
    acc: dict[int, list[float]] = {}
    for semilla in SEMILLAS:
        with _sedes(sede):
            faltantes, datos = v7se._inventario(semilla)
            if faltantes:
                raise v7se.DatosFaltantes(
                    f"semilla {semilla}: faltan {len(faltantes)} evaluaciones")
            exclusiones = v7se._exclusiones(datos)
            with v7se._ramas(semilla, exclusiones):
                per_epoch = v7ge.contrast_per_epoch(sede)
        for epoch in v7ge.EPOCHS:
            for pid, v in per_epoch["per_file"][epoch].items():
                acc.setdefault(pid, []).append(v)
    return {pid: float(np.mean(v)) for pid, v in acc.items()}


def _bootstrap_cluster(por_par: dict, grupos: dict, n: int = 10000) -> dict:
    """BCa interval resampling whole clusters, not pairs.

    The statistic is the unweighted mean of the cluster means. Cluster sizes are
    4 or 5 pairs, so that centre differs negligibly from the pair-level mean,
    which is what the criterion of section 4 is stated on and what is reported
    as the point estimate.
    """
    por_grupo: dict[str, list[float]] = {}
    for pid, v in por_par.items():
        if pid in grupos:
            por_grupo.setdefault(grupos[pid], []).append(v)
    medias = np.array([np.mean(v) for v in por_grupo.values()])
    if len(medias) < 3:
        return {"n_clusters": len(medias), "ic95": None,
                "nota": "muy pocos clusters para un intervalo"}
    res = stats.bootstrap((medias,), np.mean, n_resamples=n, method="BCa",
                          random_state=42, confidence_level=0.95)
    return {
        "n_clusters": int(len(medias)),
        "media_de_medias": float(medias.mean()),
        "ic95": [float(res.confidence_interval.low),
                 float(res.confidence_interval.high)],
    }


def _signo_adaptacion(sede: str) -> dict:
    """Section 4's C2 secondary: fraction of pairs where V5 beats V2.

    A previous version used "mean(V5 - V2) > 0", which the primary already
    implies (its threshold is positive) and which therefore could not fail on
    its own. The independent validator flagged it; this is the replacement the
    document declares. One binomial test per seed: pooling the three would reuse
    the same 250 pairs and inflate n.
    """
    v2 = _pares(RAMA_V2, sede)
    por_semilla = []
    for rama in RAMAS_V5:
        v5 = _pares(rama, sede)
        ids = [i for i in sorted(set(v5) & set(v2))
               if np.isfinite(v5[i]["pesq_nb_est"]) and np.isfinite(v2[i]["pesq_nb_est"])]
        gana = sum(1 for i in ids if v5[i]["pesq_nb_est"] > v2[i]["pesq_nb_est"])
        por_semilla.append({
            "rama": rama,
            "n": len(ids),
            "gana_v5": gana,
            "fraccion": gana / len(ids),
            "p_signo": float(stats.binomtest(gana, len(ids), 0.5).pvalue),
        })
    fr = [r["fraccion"] for r in por_semilla]
    return {
        "por_semilla": por_semilla,
        "fraccion_media": float(np.mean(fr)),
        "fraccion_rango": [float(min(fr)), float(max(fr))],
        "p_max": float(max(r["p_signo"] for r in por_semilla)),
    }


def _mecanismo(sede: str) -> dict:
    """rho(g, SNR) over every gate checkpoint of the confirmatory seeds."""
    filas = []
    for semilla in SEMILLAS:
        with _sedes(sede):
            with v7se._ramas(semilla):
                m = v7ge.mechanism(sede)
        for e in m["per_epoch"]:
            filas.append({"semilla": semilla, **e})
    rhos = np.array([f["rho"] for f in filas])
    return {
        "por_checkpoint": filas,
        "n": len(filas),
        "rho_medio": float(rhos.mean()),
        "rho_rango": [float(rhos.min()), float(rhos.max())],
        "cumple": bool((rhos <= C3_MAX_RHO).all()),
        "n_que_no_cumplen": int((rhos > C3_MAX_RHO).sum()),
    }


def _veredicto(en_sede, umbral_sede: float) -> dict:
    """Apply section 5's non-adjudication rule to a primary criterion.

    Section 3 declares which statistic governs the primary: the between-seed sd
    of the contrast **on the venue under test**, not of the paired difference.
    Section 5 then says that if the estimate sits closer than one standard error
    to the threshold, the report says so and does not adjudicate — with three
    seeds, that distance cannot separate partial retention from sampling noise.

    A previous version of this module compared against the threshold and printed
    CUMPLE or FALLA, which is not what the document licenses.

    Args:
        en_sede: the per-seed contrast values on the venue under test.
        umbral_sede: the threshold expressed on that same scale.

    Returns:
        The margin, the governing sem, their ratio, and one of
        "CUMPLE" / "FALLA" / "NO ADJUDICA".
    """
    v = np.array(en_sede, dtype=float)
    media = float(v.mean())
    sd = float(v.std(ddof=1))
    sem = sd / np.sqrt(len(v))
    margen = media - umbral_sede
    razon = float(margen / sem) if sem > 0 else float("inf")
    if abs(razon) < 1.0:
        veredicto = "NO ADJUDICA"
    else:
        veredicto = "CUMPLE" if margen >= 0 else "FALLA"
    return {
        "media_en_sede": media,
        "sd_entre_semillas": sd,
        "sem": float(sem),
        "umbral_en_sede": umbral_sede,
        "margen": float(margen),
        "margen_en_sem": razon,
        "veredicto": veredicto,
        "regla": "seccion 5: |margen| < 1 sem -> no adjudica",
    }


def _t_una_cola(valores, umbral: float) -> dict:
    """One-sided t-test of mean(valores) > umbral. Secondary: 2 df with n=3."""
    v = np.array(valores, dtype=float)
    if len(v) < 2:
        return {"t": None, "p": None, "df": 0}
    t, p = stats.ttest_1samp(v, umbral)
    return {"t": float(t), "p": float(p / 2 if t > 0 else 1 - p / 2),
            "df": len(v) - 1}


def adjudicar(sede: str = SEDE, referencia: str = SEDE_REFERENCIA) -> dict:
    c1 = {s: _contraste_compuerta(s, sede, referencia) for s in SEMILLAS}
    ds = [c1[s]["d"] for s in SEMILLAS]
    en_sede = [c1[s]["sede"] for s in SEMILLAS]

    c2 = _contraste_adaptacion(sede, referencia)
    c3 = _mecanismo(sede)

    por_par = _contraste_por_par(sede)
    por_grabacion, por_locacion = _clusters(sede)
    intervalos = {
        "estimador_puntual_nivel_par": float(np.mean(list(por_par.values()))),
        "por_grabacion": _bootstrap_cluster(por_par, por_grabacion),
        "por_locacion": _bootstrap_cluster(por_par, por_locacion),
    }

    resultado = {
        "sede": sede,
        "sede_referencia": referencia,
        "semillas": list(SEMILLAS),
        "C1": {
            "por_semilla": c1,
            "d_medio": float(np.mean(ds)),
            "d_sd": float(np.std(ds, ddof=1)),
            "umbral": C1_UMBRAL,
            "primario": _veredicto(en_sede, C1_UMBRAL_SEDE),
            "retencion": float(np.mean(en_sede) / C1_REFERENCIA),
            "positivo_en_sede": int(sum(1 for x in en_sede if x > 0)),
            "cumple_secundario": bool(all(x > 0 for x in en_sede)),
            "t_secundario": _t_una_cola(ds, C1_UMBRAL),
            "intervalos": intervalos,
        },
        "C2": {
            **c2,
            "umbral": C2_UMBRAL,
            "primario": _veredicto(c2["sede"]["por_semilla"], C2_UMBRAL_SEDE),
            "retencion": float(c2["sede"]["media"] / C2_REFERENCIA),
            "secundario_signo": {
                "sede": _signo_adaptacion(sede),
                "referencia": _signo_adaptacion(referencia),
            },
            "t_secundario": _t_una_cola(
                [a - b for a, b in zip(c2["sede"]["por_semilla"],
                                       c2["referencia"]["por_semilla"])],
                C2_UMBRAL),
        },
        "C3": {**c3, "umbral": C3_MAX_RHO},
    }

    # Descriptivo, declarado en la sección 7 del preregistro: no puntuado.
    resultado["descriptivo"] = {
        "semilla_screening": _contraste_compuerta(
            SEMILLA_SCREENING, sede, referencia),
        "perfil_por_bucket_sede": _perfil_por_bucket(sede),
        "perfil_por_bucket_referencia": _perfil_por_bucket(referencia),
    }
    return resultado


def _perfil_por_bucket(sede: str) -> list:
    """D2 of section 7: both contrasts per SNR bucket, plus the noisy baseline.

    Declared descriptive and not scored, but it is where the structure lives:
    the two contrasts behave very differently across the SNR range, and the
    noisy baseline is what shows that holding `snr_db` constant between venues
    does **not** hold difficulty constant — WHAM! babble destroys more PESQ than
    MUSAN music at the same nominal SNR.
    """
    # C1 por bucket: media sobre las tres semillas del desglose por epocas
    por_semilla = []
    for semilla in SEMILLAS:
        with _sedes(sede):
            faltantes, datos = v7se._inventario(semilla)
            if faltantes:
                raise v7se.DatosFaltantes(
                    f"semilla {semilla}: faltan {len(faltantes)} evaluaciones")
            with v7se._ramas(semilla, v7se._exclusiones(datos)):
                por_semilla.append({r["bucket_idx"]: r["contrast"]
                                    for r in v7ge.bucket_breakdown(sede)})

    # C2 por bucket y linea de base del ruidoso
    v2 = _pares(RAMA_V2, sede)
    buckets = sorted({q["bucket_idx"] for q in v2.values()})
    c2_por_bucket, base, snr_rango = {}, {}, {}
    for b in buckets:
        ids_b = [i for i, q in v2.items() if q["bucket_idx"] == b]
        base[b] = float(np.mean([v2[i]["pesq_nb_noisy"] for i in ids_b]))
        snr = [v2[i]["snr_db"] for i in ids_b]
        snr_rango[b] = [float(min(snr)), float(max(snr))]
        s_seeds = []
        for rama in RAMAS_V5:
            v5 = _pares(rama, sede)
            ids = [i for i in ids_b if i in v5
                   and np.isfinite(v5[i]["pesq_nb_est"])
                   and np.isfinite(v2[i]["pesq_nb_est"])]
            s_seeds.append(np.mean([v5[i]["pesq_nb_est"] - v2[i]["pesq_nb_est"]
                                    for i in ids]))
        c2_por_bucket[b] = float(np.mean(s_seeds))

    return [{
        "bucket_idx": b,
        "snr_rango": snr_rango[b],
        "n": sum(1 for q in v2.values() if q["bucket_idx"] == b),
        "pesq_nb_noisy": base[b],
        "c1_compuerta": float(np.mean([d[b] for d in por_semilla if b in d])),
        "c2_adaptacion": c2_por_bucket[b],
    } for b in buckets]


def _imprimir(r: dict) -> None:
    print("=" * 74)
    print(f"ADJUDICACIÓN — {r['sede']} (ruido holdout) vs {r['sede_referencia']}")
    print("=" * 74)

    c = r["C1"]
    v = c["primario"]
    print("\nC1 — la compuerta sobrevive al ruido no visto")
    for s_ in r["semillas"]:
        q = c["por_semilla"][s_]
        print(f"    s{s_}: {r['sede_referencia']} {q['referencia']:+.4f} -> "
              f"{r['sede']} {q['sede']:+.4f}   d = {q['d']:+.4f}")
    print(f"    primario: E({r['sede']}) = {v['media_en_sede']:+.6f} vs umbral "
          f"{v['umbral_en_sede']:+.6f}")
    print(f"              margen {v['margen']:+.6f} / sem {v['sem']:.6f} = "
          f"{v['margen_en_sem']:.2f} sem   ->  {v['veredicto']}")
    if v["veredicto"] == "NO ADJUDICA":
        print(f"              ({v['regla']}; con 3 semillas esa distancia no separa")
        print(f"               retención parcial de ruido de muestreo)")
    print(f"    retención {c['retencion']*100:.1f}% | positivo en "
          f"{c['positivo_en_sede']}/3 semillas -> "
          f"{'CUMPLE' if c['cumple_secundario'] else 'FALLA'} (secundario, no decide)")
    iv = c["intervalos"]
    for k, et in (("por_grabacion", "grabación"), ("por_locacion", "locación")):
        b = iv[k]
        if b.get("ic95"):
            print(f"    IC95 BCa por {et} (n={b['n_clusters']}): "
                  f"[{b['ic95'][0]:+.4f}, {b['ic95'][1]:+.4f}]")

    c = r["C2"]
    v = c["primario"]
    print("\nC2 — la adaptación sobrevive al ruido no visto")
    print(f"    por semilla {['%+.4f' % x for x in c['sede']['por_semilla']]}")
    print(f"    primario: {c['sede']['media']:+.6f} vs umbral "
          f"{v['umbral_en_sede']:+.6f}")
    print(f"              margen {v['margen']:+.6f} / sem {v['sem']:.6f} = "
          f"{abs(v['margen_en_sem']):.2f} sem   ->  {v['veredicto']}")
    print(f"    retención {c['retencion']*100:.1f}%")
    sg = c["secundario_signo"]
    print(f"    secundario (test de signo, fracción de pares con V5 > V2):")
    print(f"       {r['sede_referencia']:<12} {sg['referencia']['fraccion_media']*100:5.1f}%  "
          f"p_max {sg['referencia']['p_max']:.2e}")
    print(f"       {r['sede']:<12} {sg['sede']['fraccion_media']*100:5.1f}%  "
          f"p_max {sg['sede']['p_max']:.2e}")

    c = r["C3"]
    print("\nC3 — el mecanismo sobrevive")
    print(f"    rho medio {c['rho_medio']:+.4f}  rango "
          f"[{c['rho_rango'][0]:+.4f}, {c['rho_rango'][1]:+.4f}]  n={c['n']}")
    veredicto = ("CUMPLE" if c["cumple"]
                 else f"FALLA ({c['n_que_no_cumplen']} checkpoints no cumplen)")
    print(f"    <= {c['umbral']:+.2f} en los {c['n']}  ->  {veredicto}")

    print("\n" + "-" * 74)
    print("D2 (descriptivo, declarado, NO puntuado) — perfil por bucket de SNR")
    print("-" * 74)
    a = {x["bucket_idx"]: x for x in r["descriptivo"]["perfil_por_bucket_referencia"]}
    b = {x["bucket_idx"]: x for x in r["descriptivo"]["perfil_por_bucket_sede"]}
    print(f"{'SNR':<12}{'noisy ref':>10}{'noisy sede':>11}{'C1 ref':>9}"
          f"{'C1 sede':>9}{'ret':>7}{'C2 ref':>9}{'C2 sede':>9}{'ret':>8}")
    for k in sorted(b):
        lo, hi = b[k]["snr_rango"]
        rc1 = b[k]["c1_compuerta"] / a[k]["c1_compuerta"] * 100 if a[k]["c1_compuerta"] else float("nan")
        rc2 = b[k]["c2_adaptacion"] / a[k]["c2_adaptacion"] * 100 if a[k]["c2_adaptacion"] else float("nan")
        print(f"[{lo:+.0f},{hi:+.0f}]".ljust(12)
              + f"{a[k]['pesq_nb_noisy']:>10.4f}{b[k]['pesq_nb_noisy']:>11.4f}"
              + f"{a[k]['c1_compuerta']:>+9.4f}{b[k]['c1_compuerta']:>+9.4f}{rc1:>6.0f}%"
              + f"{a[k]['c2_adaptacion']:>+9.4f}{b[k]['c2_adaptacion']:>+9.4f}{rc2:>7.0f}%")
    print("\n" + "=" * 74)


def selftest() -> None:
    """Reproduce published numbers on the venue pair that already exists.

    Proves the declared estimands are computable with the declared code, using
    nothing but already-adjudicated data. Does not touch wham_es.
    """
    print("SELFTEST — mismo código sobre v1_en <-> v3_mls_es (datos publicados)\n")
    c1 = {s: _contraste_compuerta(s, "v3_mls_es", "v1_en") for s in SEMILLAS}
    esperado = {43: 0.0558, 44: 0.0367, 45: 0.0397}
    ok = True
    for s in SEMILLAS:
        got = c1[s]["sede"]
        bien = abs(got - esperado[s]) < 5e-4
        ok &= bien
        print(f"  s{s}: v3_mls_es {got:+.4f} (publicado {esperado[s]:+.4f}) "
              f"{'ok' if bien else 'DISTINTO'}")
    ds = [c1[s]["d"] for s in SEMILLAS]
    print(f"\n  sd del contraste crudo en v3_mls_es: "
          f"{np.std([c1[s]['sede'] for s in SEMILLAS], ddof=1):.4f}")
    print(f"  sd de la diferencia apareada:        {np.std(ds, ddof=1):.4f}")
    print(f"  -> el apareo reduce el ruido "
          f"{np.std([c1[s]['sede'] for s in SEMILLAS], ddof=1)/np.std(ds, ddof=1):.2f}x "
          f"SOLO en este par, que comparte el ruido verbatim; en los otros dos "
          f"lo empeora (0,55x y 0,39x). No se usa para la potencia.")
    c2 = _contraste_adaptacion("v3_mls_es", "v1_en")
    print(f"\n  C2 en v3_mls_es: {c2['sede']['media']:+.4f} (publicado +0.0583) "
          f"{'ok' if abs(c2['sede']['media'] - 0.0583) < 5e-4 else 'DISTINTO'}")
    m = _mecanismo("v3_mls_es")
    print(f"  C3 en v3_mls_es: rho medio {m['rho_medio']:+.4f} "
          f"(publicado -0.3954) n={m['n']} "
          f"{'ok' if abs(m['rho_medio'] + 0.3954) < 5e-4 else 'DISTINTO'}")
    print(f"\n{'SELFTEST OK' if ok else 'SELFTEST CON DIFERENCIAS'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="verifica la maquinaria sobre las sedes que ya existen")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    r = adjudicar()
    _imprimir(r)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"Escrito: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
