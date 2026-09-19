"""
analysis/v7_gate_endpoints.py

Computes the preregistered endpoints of V7 (gate trained from scratch) from the
per-epoch sweep in ``results/v7_epochs/``.

Preregistration: ``~/nosiesuppressnet-oracle/preregistro_compuerta_desde_cero.md``
(hash in ``docs/preregistro_v7_desde_cero.sha256``, written 15/09/2026 before any
from-scratch run existed). The estimands, thresholds and the sign convention are
taken verbatim from it:

    E1 (primary) global PESQ-NB contrast on test_v2_es, averaged over epochs
                 15-20, paired file by file. Screening success: >= +0.050.
    E2 (cost)    same estimand on test_v1_en. Success: >= -0.020.
    E3 (mechanism) rho(g, SNR) NEGATIVE with |rho| >= 0.15 on the three sealed
                 sets. g = 1 is pure enhancement, g = 0 is the untouched input,
                 so backing off on clean input means g falling as SNR rises.
    E4 (degeneracy) mean of g inside (0.05, 0.99).
    E5           E1 against V6's trajectory-averaged contrast, +0.0187.

The contrast is always treatment - control: v7_gate - v7_control.

The six checkpoints of a trajectory are NOT independent samples, so the p-value
over the epoch-averaged per-file contrast is reported for scale only, not as a
confirmatory test. This is the same caveat V6 carries in section 9 of
``docs/v6_compuerta.md``.

Usage:
    python -m analysis.v7_gate_endpoints
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "results" / "v7_epochs"
OUTPUT = PROJECT_ROOT / "results" / "v7_endpoints.json"

EPOCHS = [15, 16, 17, 18, 19, 20]
# E1 vive en v2_es y E2 en v1_en. v3_mls_es entra por E3, que el preregistro
# pide sobre los tres sellados, y de paso da el contraste sobre el tercero.
TEST_SETS = ["v2_es", "v1_en", "v3_mls_es"]
E1_THRESHOLD = 0.050
E2_THRESHOLD = -0.020
E3_MIN_ABS_RHO = 0.15
E4_BOUNDS = (0.05, 0.99)
V6_TRAJECTORY_CONTRAST = 0.0187


def _load(branch: str, epoch: int, test_set: str) -> dict:
    path = SWEEP_DIR / f"{branch}_ep{epoch}_{test_set}.json"
    with open(path) as f:
        return json.load(f)


def _pairs_by_id(result: dict) -> dict:
    return {p["pair_id"]: p for p in result["all_pairs"]}


def contrast_per_epoch(test_set: str) -> dict:
    """Per-file paired contrast gate - control, one entry per epoch."""
    out = {"epochs": [], "per_file": {}}
    for epoch in EPOCHS:
        gate = _load("v7_gate", epoch, test_set)
        control = _load("v7_control", epoch, test_set)
        g_pairs, c_pairs = _pairs_by_id(gate), _pairs_by_id(control)
        if set(g_pairs) != set(c_pairs):
            raise RuntimeError(f"{test_set} ep{epoch}: los pair_id no coinciden")

        ids = sorted(g_pairs)
        diff = np.array([g_pairs[i]["pesq_nb_est"] - c_pairs[i]["pesq_nb_est"]
                         for i in ids])
        # El SNR y el noisy de referencia salen del sellado: tienen que ser
        # idénticos entre ramas, o no están evaluando el mismo par.
        for i in ids:
            if abs(g_pairs[i]["snr_db"] - c_pairs[i]["snr_db"]) > 1e-9:
                raise RuntimeError(f"{test_set} ep{epoch} par {i}: SNR distinto")

        t = stats.ttest_rel([g_pairs[i]["pesq_nb_est"] for i in ids],
                            [c_pairs[i]["pesq_nb_est"] for i in ids])
        out["epochs"].append({
            "epoch": epoch,
            "gate_mean": float(np.mean([g_pairs[i]["pesq_nb_est"] for i in ids])),
            "control_mean": float(np.mean([c_pairs[i]["pesq_nb_est"] for i in ids])),
            "contrast": float(diff.mean()),
            "p_paired": float(t.pvalue),
            "n_pairs": len(ids),
        })
        out["per_file"][epoch] = {i: d for i, d in zip(ids, diff)}
    return out


def summarize(test_set: str) -> dict:
    per_epoch = contrast_per_epoch(test_set)
    contrasts = np.array([e["contrast"] for e in per_epoch["epochs"]])

    ids = sorted(per_epoch["per_file"][EPOCHS[0]])
    averaged = np.array([np.mean([per_epoch["per_file"][ep][i] for ep in EPOCHS])
                         for i in ids])
    t = stats.ttest_1samp(averaged, 0.0)
    w = stats.wilcoxon(averaged)

    return {
        "test_set": test_set,
        "per_epoch": per_epoch["epochs"],
        "estimand": float(contrasts.mean()),
        "sd_across_epochs": float(contrasts.std(ddof=1)),
        "range": [float(contrasts.min()), float(contrasts.max())],
        "sign_pattern": "".join("+" if c > 0 else "-" for c in contrasts),
        "n_positive": int((contrasts > 0).sum()),
        # No confirmatorio: las seis épocas no son muestras independientes.
        "p_epoch_averaged_paired": float(t.pvalue),
        "p_wilcoxon": float(w.pvalue),
        "files_favoring_gate": int((averaged > 0).sum()),
        "n_files": len(averaged),
    }


def bucket_breakdown(test_set: str) -> list:
    """Descriptive only: the contrast per SNR bucket, averaged over epochs."""
    rows = {}
    for epoch in EPOCHS:
        gate, control = _load("v7_gate", epoch, test_set), _load("v7_control", epoch, test_set)
        g_pairs, c_pairs = _pairs_by_id(gate), _pairs_by_id(control)
        for i, gp in g_pairs.items():
            rows.setdefault(gp["bucket_idx"], []).append(
                gp["pesq_nb_est"] - c_pairs[i]["pesq_nb_est"])
    return [{"bucket_idx": b,
             "contrast": float(np.mean(v)),
             "n": len(v) // len(EPOCHS)}
            for b, v in sorted(rows.items())]


def mechanism(test_set: str) -> dict:
    """E3 and E4 on the gate arm: rho(g, SNR) and the mean of g."""
    per_epoch = []
    for epoch in EPOCHS:
        gate = _load("v7_gate", epoch, test_set)
        g = np.array([p["gate_mean"] for p in gate["all_pairs"]])
        snr = np.array([p["snr_db"] for p in gate["all_pairs"]])
        rho, p = stats.pearsonr(g, snr)
        per_epoch.append({"epoch": epoch, "rho": float(rho), "p": float(p),
                          "g_mean": float(g.mean()),
                          "g_min": float(g.min()), "g_max": float(g.max())})
    rhos = np.array([e["rho"] for e in per_epoch])
    g_means = np.array([e["g_mean"] for e in per_epoch])
    return {
        "test_set": test_set,
        "per_epoch": per_epoch,
        "rho_mean": float(rhos.mean()),
        "rho_range": [float(rhos.min()), float(rhos.max())],
        "g_mean": float(g_means.mean()),
        "g_mean_range": [float(g_means.min()), float(g_means.max())],
        "e3_pass": bool((rhos < 0).all() and (np.abs(rhos) >= E3_MIN_ABS_RHO).all()),
        "e4_pass": bool((g_means > E4_BOUNDS[0]).all() and (g_means < E4_BOUNDS[1]).all()),
    }


def _available() -> list:
    """Los sellados con el barrido completo de las dos ramas."""
    ready = []
    for ts in TEST_SETS:
        if all((SWEEP_DIR / f"{br}_ep{ep}_{ts}.json").exists()
               for br in ("v7_gate", "v7_control") for ep in EPOCHS):
            ready.append(ts)
        else:
            print(f"[aviso] {ts}: barrido incompleto, se omite")
    return ready


def main() -> None:
    test_sets = _available()
    results = {s: summarize(s) for s in test_sets}
    buckets = {s: bucket_breakdown(s) for s in test_sets}
    mech = {s: mechanism(s) for s in test_sets}

    e1, e2 = results["v2_es"]["estimand"], results["v1_en"]["estimand"]
    if e1 >= E1_THRESHOLD:
        e5 = "desde cero rinde sustancialmente mas; corresponde confirmar con tres semillas"
    elif e1 > V6_TRAJECTORY_CONTRAST:
        e5 = "mejora menor que la predicha por la division del trabajo; se reporta, no se confirma"
    else:
        e5 = "desde cero NO rinde mas que atornillar la compuerta tarde"

    verdict = {
        "E1_primary": {"estimand": e1, "threshold": E1_THRESHOLD,
                       "pass": bool(e1 >= E1_THRESHOLD)},
        "E2_cost": {"estimand": e2, "threshold": E2_THRESHOLD,
                    "pass": bool(e2 >= E2_THRESHOLD)},
        "E3_mechanism": {
            "por_sellado": {s: mech[s]["e3_pass"] for s in test_sets},
            "sellados_evaluados": test_sets,
            "pass": bool(len(test_sets) == len(TEST_SETS)
                         and all(mech[s]["e3_pass"] for s in test_sets)),
            "nota": None if len(test_sets) == len(TEST_SETS)
                    else "el preregistro pide los TRES sellados; faltan barridos"},
        "E4_degeneracy": {
            "por_sellado": {s: mech[s]["e4_pass"] for s in test_sets},
            "pass": bool(all(mech[s]["e4_pass"] for s in test_sets))},
        "E5_from_scratch": {"v6_trajectory_contrast": V6_TRAJECTORY_CONTRAST, "reading": e5},
    }

    payload = {"contrast": "v7_gate - v7_control, PESQ-NB, pareado por archivo",
               "epochs": EPOCHS, "endpoints": verdict,
               "by_test_set": results, "by_bucket": buckets, "mechanism": mech}
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2)

    for s in test_sets:
        r = results[s]
        print(f"\n=== {s} — contraste global PESQ-NB (compuerta − control) ===")
        for e in r["per_epoch"]:
            print(f"  ep{e['epoch']}  compuerta={e['gate_mean']:.4f}  "
                  f"control={e['control_mean']:.4f}  contraste={e['contrast']:+.4f}  "
                  f"p={e['p_paired']:.2g}")
        print(f"  estimando (media 15-20): {r['estimand']:+.4f}   "
              f"sd entre épocas {r['sd_across_epochs']:.4f}   "
              f"rango [{r['range'][0]:+.4f}, {r['range'][1]:+.4f}]   "
              f"signos {r['sign_pattern']}")
        print(f"  archivos que favorecen a la compuerta: "
              f"{r['files_favoring_gate']}/{r['n_files']}   "
              f"p(promediado, NO confirmatorio)={r['p_epoch_averaged_paired']:.2g}")
        print("  por bucket:", "  ".join(
            f"b{b['bucket_idx']}={b['contrast']:+.4f}" for b in buckets[s]))
        m = mech[s]
        print(f"  ρ(g,SNR) media {m['rho_mean']:+.3f} "
              f"rango [{m['rho_range'][0]:+.3f}, {m['rho_range'][1]:+.3f}]   "
              f"media de g {m['g_mean']:.3f} "
              f"rango [{m['g_mean_range'][0]:.3f}, {m['g_mean_range'][1]:.3f}]")

    print("\n=== veredicto contra el preregistro ===")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
