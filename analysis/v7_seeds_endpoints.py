"""
analysis/v7_seeds_endpoints.py

Aggregation wrapper for the confirmatory endpoint of V7 over the seed replicas
(43, 44, 45). Writes ``results/v7_seeds_endpoints.json``.

Preregistration: ``~/nosiesuppressnet-oracle/preregistro_v7_semillas.md`` (hash in
``docs/preregistro_v7_semillas.sha256``, commit ``0424c74``, written before any
result over any sealed set existed). Section 3, verbatim:

    El estimando confirmatorio es la media no ponderada de los estimandos de las
    semillas 43, 44 y 45. La semilla 42 se reporta siempre, y nunca entra en ese
    numero.

    Para cada semilla s se calcula E(s) con summarize() de
    analysis/v7_gate_endpoints.py sobre el par de ramas v7_gate_s{s} /
    v7_control_s{s}. El estimando confirmatorio es (E(43)+E(44)+E(45))/3 — media
    de las tres medias, NO el promedio agrupado sobre archivos x epocas de las
    tres corridas juntas.

Por que la media de medias y no el pool: las dos coinciden solo si las tres
semillas tienen los 250 pares utilizables en las seis epocas. El pool le daria
mas peso a la semilla con mas pares sobrevivientes, que es exactamente la
semilla con mas problemas de datos.

Como se reusa el analisis del screening sin tocarlo
---------------------------------------------------
``analysis/v7_gate_endpoints.py`` esta congelado: es el codigo con el que se
calculo el screening y la seccion 8 del preregistro dice que la funcion que
calcula el estimando por par de ramas no se toca. Este envoltorio lo importa y
parametriza las ramas sustituyendo ``v7ge._load`` dentro de un context manager.
``contrast_per_epoch``, ``summarize``, ``bucket_breakdown`` y ``mechanism``
resuelven ``_load`` por nombre en el modulo, asi que corren su aritmetica
original sobre los JSON de la semilla pedida. Nada del modulo importado se
modifica en disco y el parche se deshace siempre.

El mismo parche aplica la exclusion de NaN de la seccion 7, porque la exclusion
es responsabilidad del envoltorio y no de la funcion congelada: filtra los pares
excluidos del JSON antes de que ``summarize`` lo vea, de forma simetrica en los
dos brazos.

Verificacion
------------
``--verificar-s42`` corre el envoltorio completo restringido a la semilla 42
(ramas ``v7_gate`` / ``v7_control``, las del screening) y compara contra el E1
que publica ``docs/v7_compuerta_desde_cero.md``. Si no reproduce ese numero, el
que esta mal es este script. No escribe nada.

Usage:
    python -m analysis.v7_seeds_endpoints --verificar-s42
    python -m analysis.v7_seeds_endpoints --estado
    python -m analysis.v7_seeds_endpoints
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from analysis import v7_gate_endpoints as v7ge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "results" / "v7_seeds_endpoints.json"

# La semilla 42 genero la hipotesis. Se reporta siempre y nunca entra en el
# estimando confirmatorio (preregistro, seccion 3).
SEMILLAS_CONFIRMATORIAS = (43, 44, 45)
SEMILLA_SCREENING = 42

# E1 vive en test_v2_es; el resto se agrega igual y se reporta al lado.
SELLADO_PRIMARIO = "v2_es"

# Publicado en docs/v7_compuerta_desde_cero.md, seccion 5 (E1, media ep. 15-20).
E1_PUBLICADO_S42 = 0.0795

# Reglas de datos faltantes, preregistro seccion 7.
MAX_EXCLUIDOS_POR_EPOCA = 5
MAX_EPOCAS_INVALIDAS_POR_SEMILLA = 2

ROLES = ("v7_gate", "v7_control")


class DatosFaltantes(RuntimeError):
    """Falta al menos un JSON del barrido. No se devuelve un parcial."""


class PanelDesbalanceado(RuntimeError):
    """La exclusion por NaN deja distintos pares en distintas epocas."""


def _rama(rol: str, semilla: int) -> str:
    """Branch name of `rol` for a given seed. Seed 42 keeps the screening names."""
    if semilla == SEMILLA_SCREENING:
        return rol
    return f"{rol}_s{semilla}"


def _inventario(semilla: int) -> tuple[list[str], dict]:
    """Load every JSON the endpoint needs for one seed.

    Returns (missing_paths, data). Uses ``v7ge._load`` so the file naming
    convention has a single source of truth: whatever it opens is what the
    estimand will read.
    """
    faltantes, datos = [], {}
    for rol in ROLES:
        for test_set in v7ge.TEST_SETS:
            for epoch in v7ge.EPOCHS:
                try:
                    datos[(rol, test_set, epoch)] = v7ge._load(
                        _rama(rol, semilla), epoch, test_set)
                except FileNotFoundError as e:
                    faltantes.append(e.filename)
    return faltantes, datos


def _exclusiones(datos: dict) -> dict:
    """Section 7: pairs whose PESQ-NB is NaN in EITHER arm drop from that epoch
    in BOTH arms. Returns {(test_set, epoch): {"pares": frozenset, "n": int,
    "epoca_invalida": bool}}.
    """
    out = {}
    for test_set in v7ge.TEST_SETS:
        for epoch in v7ge.EPOCHS:
            malos = set()
            for rol in ROLES:
                for p in datos[(rol, test_set, epoch)]["all_pairs"]:
                    if math.isnan(p["pesq_nb_est"]):
                        malos.add(p["pair_id"])
            out[(test_set, epoch)] = {
                "pares": frozenset(malos),
                "n": len(malos),
                "epoca_invalida": len(malos) > MAX_EXCLUIDOS_POR_EPOCA,
            }
    return out


def _verificar_panel(exclusiones: dict, semilla: int) -> None:
    """``summarize`` promedia por archivo sobre las seis epocas y asume el mismo
    conjunto de pares en todas. Con exclusiones distintas por epoca ese promedio
    no esta definido, asi que se corta en vez de inventar una regla.
    """
    for test_set in v7ge.TEST_SETS:
        conjuntos = {ep: exclusiones[(test_set, ep)]["pares"] for ep in v7ge.EPOCHS}
        distintos = set(conjuntos.values())
        if len(distintos) > 1:
            detalle = ", ".join(f"ep{ep}={sorted(s) or '[]'}" for ep, s in conjuntos.items())
            raise PanelDesbalanceado(
                f"semilla {semilla}, {test_set}: la exclusion por NaN de la seccion 7 "
                f"es por epoca y deja un panel desbalanceado ({detalle}). "
                "summarize() esta congelada y promedia por archivo sobre las seis "
                "epocas, asi que ese caso necesita una decision declarada antes de "
                "calcular nada. No se resuelve solo."
            )


@contextmanager
def _ramas(semilla: int, exclusiones: dict | None = None):
    """Point ``v7ge._load`` at this seed's branch pair, applying the section 7
    exclusions. Restores the original loader on the way out, always.
    """
    original = v7ge._load

    def _load_semilla(rol: str, epoch: int, test_set: str) -> dict:
        data = original(_rama(rol, semilla), epoch, test_set)
        fuera = (exclusiones or {}).get((test_set, epoch), {}).get("pares", frozenset())
        if fuera:
            data = dict(data)
            data["all_pairs"] = [p for p in data["all_pairs"]
                                 if p["pair_id"] not in fuera]
        return data

    v7ge._load = _load_semilla
    try:
        yield
    finally:
        v7ge._load = original


def estimando_por_semilla(semilla: int) -> dict:
    """E(s) sobre los tres sellados, con el desglose del screening al lado."""
    faltantes, datos = _inventario(semilla)
    if faltantes:
        raise DatosFaltantes(
            f"semilla {semilla}: faltan {len(faltantes)} de "
            f"{len(ROLES) * len(v7ge.TEST_SETS) * len(v7ge.EPOCHS)} evaluaciones:\n  "
            + "\n  ".join(sorted(faltantes)))

    exclusiones = _exclusiones(datos)
    _verificar_panel(exclusiones, semilla)

    with _ramas(semilla, exclusiones):
        por_sellado = {ts: v7ge.summarize(ts) for ts in v7ge.TEST_SETS}
        buckets = {ts: v7ge.bucket_breakdown(ts) for ts in v7ge.TEST_SETS}
        mecanismo = {ts: v7ge.mechanism(ts) for ts in v7ge.TEST_SETS}

    epocas_invalidas = [{"test_set": ts, "epoch": ep, "n_excluidos": v["n"]}
                        for (ts, ep), v in sorted(exclusiones.items())
                        if v["epoca_invalida"]]
    n_invalidas_primario = sum(1 for e in epocas_invalidas
                               if e["test_set"] == SELLADO_PRIMARIO)

    return {
        "semilla": semilla,
        "ramas": {rol: _rama(rol, semilla) for rol in ROLES},
        "estimando": {ts: por_sellado[ts]["estimand"] for ts in v7ge.TEST_SETS},
        "por_sellado": por_sellado,
        "por_bucket": buckets,
        "mecanismo": mecanismo,
        "datos": {
            "excluidos_por_nan": {f"{ts}_ep{ep}": v["n"]
                                  for (ts, ep), v in sorted(exclusiones.items())
                                  if v["n"]},
            "total_excluidos": sum(v["n"] for v in exclusiones.values()),
            "epocas_invalidas": epocas_invalidas,
            # Seccion 7: una epoca invalida se reporta como desvio. NO se saca
            # del promedio, porque la seccion 3 fija el promedio sobre las seis
            # epocas y la seccion 7 no autoriza recortarlo.
            "semilla_no_computa": n_invalidas_primario > MAX_EPOCAS_INVALIDAS_POR_SEMILLA,
        },
    }


def agregar(por_semilla: dict) -> dict:
    """Media no ponderada de los estimandos por semilla (preregistro, seccion 3)."""
    computan = [s for s in SEMILLAS_CONFIRMATORIAS
                if not por_semilla[s]["datos"]["semilla_no_computa"]]
    descartadas = [s for s in SEMILLAS_CONFIRMATORIAS if s not in computan]

    agregado = {}
    for ts in v7ge.TEST_SETS:
        valores = np.array([por_semilla[s]["estimando"][ts] for s in computan])
        agregado[ts] = {
            "estimando_confirmatorio": float(valores.mean()),
            "por_semilla": {str(s): por_semilla[s]["estimando"][ts] for s in computan},
            "sd_entre_semillas": float(valores.std(ddof=1)) if len(valores) > 1 else None,
            "rango": [float(valores.min()), float(valores.max())],
            "n_semillas": len(computan),
            # Se reporta siempre, al lado y nunca adentro (seccion 3).
            "semilla_42_screening": por_semilla[SEMILLA_SCREENING]["estimando"][ts],
        }

    return {
        "definicion": "media no ponderada de los estimandos por semilla; NO pool "
                      "sobre archivos x epocas",
        "semillas_confirmatorias": list(computan),
        "semillas_descartadas_por_seccion_7": descartadas,
        "semilla_excluida_por_diseno": SEMILLA_SCREENING,
        "sellado_primario": SELLADO_PRIMARIO,
        "por_sellado": agregado,
    }


def estado() -> int:
    """Que hay y que falta, sin calcular ningun estimando."""
    total = len(ROLES) * len(v7ge.TEST_SETS) * len(v7ge.EPOCHS)
    listas = 0
    print(f"Barrido esperado: {total} evaluaciones por semilla "
          f"({len(ROLES)} ramas x {len(v7ge.TEST_SETS)} sellados x "
          f"{len(v7ge.EPOCHS)} epocas)\n")
    for s in (SEMILLA_SCREENING,) + SEMILLAS_CONFIRMATORIAS:
        faltantes, _ = _inventario(s)
        marca = "completa" if not faltantes else f"faltan {len(faltantes)}"
        print(f"  semilla {s:>2}  ({_rama('v7_gate', s)} / {_rama('v7_control', s)}): {marca}")
        if faltantes and len(faltantes) <= 6:
            for f in sorted(faltantes):
                print(f"        falta {f}")
        if not faltantes and s in SEMILLAS_CONFIRMATORIAS:
            listas += 1
    print(f"\nSemillas confirmatorias listas: {listas}/{len(SEMILLAS_CONFIRMATORIAS)}")
    if listas < len(SEMILLAS_CONFIRMATORIAS):
        print("Seccion 7: sin las tres no hay afirmacion confirmatoria. Se espera.")
    return 0


def verificar_s42() -> int:
    """Reproduce el E1 publicado del screening usando este envoltorio."""
    r = estimando_por_semilla(SEMILLA_SCREENING)
    obtenido = r["estimando"][SELLADO_PRIMARIO]
    esperado = E1_PUBLICADO_S42
    ok = round(obtenido, 4) == esperado

    print(f"Verificacion contra la semilla {SEMILLA_SCREENING} "
          f"(ramas {r['ramas']['v7_gate']} / {r['ramas']['v7_control']})")
    print(f"  E1 sobre {SELLADO_PRIMARIO}, media epocas {v7ge.EPOCHS[0]}-{v7ge.EPOCHS[-1]}")
    print(f"    obtenido : {obtenido:+.6f}  (redondeado {round(obtenido, 4):+.4f})")
    print(f"    publicado: {esperado:+.4f}  (docs/v7_compuerta_desde_cero.md, seccion 5)")
    print(f"    delta    : {obtenido - esperado:+.2e}")
    for ts in v7ge.TEST_SETS:
        if ts != SELLADO_PRIMARIO:
            print(f"  {ts}: {r['estimando'][ts]:+.6f}")
    print("  -> COINCIDE" if ok else "  -> NO COINCIDE: el script esta mal, no el documento")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--verificar-s42", action="store_true",
                        help="reproduce el E1 publicado del screening; no escribe nada")
    parser.add_argument("--estado", action="store_true",
                        help="dice que JSON hay y cuales faltan; no calcula nada")
    args = parser.parse_args()

    if args.estado:
        raise SystemExit(estado())
    if args.verificar_s42:
        raise SystemExit(verificar_s42())

    # Completitud: sin las tres semillas nuevas no hay numero confirmatorio
    # (preregistro, seccion 7). Falla con la lista, no devuelve un parcial.
    faltan_todas = {}
    for s in SEMILLAS_CONFIRMATORIAS:
        faltantes, _ = _inventario(s)
        if faltantes:
            faltan_todas[s] = faltantes
    if faltan_todas:
        detalle = "\n".join(
            f"  semilla {s}: faltan {len(f)} evaluaciones\n    "
            + "\n    ".join(sorted(f)[:4])
            + (f"\n    ... y {len(f) - 4} mas" if len(f) > 4 else "")
            for s, f in sorted(faltan_todas.items()))
        raise SystemExit(
            "No hay afirmacion confirmatoria: el barrido de semillas esta "
            f"incompleto.\n{detalle}\n\n"
            "El n esta fijado en tres y no se recorta (preregistro, seccion 7). "
            "Corre `--estado` para el detalle completo.")

    por_semilla = {s: estimando_por_semilla(s)
                   for s in (SEMILLA_SCREENING,) + SEMILLAS_CONFIRMATORIAS}
    agregado = agregar(por_semilla)

    payload = {
        "contraste": "v7_gate - v7_control, PESQ-NB, pareado por archivo",
        "epocas": v7ge.EPOCHS,
        "preregistro": "docs/preregistro_v7_semillas.sha256 (commit 0424c74)",
        "confirmatorio": agregado,
        "por_semilla": {str(s): por_semilla[s] for s in por_semilla},
        "criterios": "Los umbrales de decision viven en la seccion 4 del "
                     "preregistro, fuera del repo. Este script calcula el "
                     "estimando; no emite veredicto.",
    }
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== estimando confirmatorio de V7 — media no ponderada de 43, 44, 45 ===")
    for ts in v7ge.TEST_SETS:
        a = agregado["por_sellado"][ts]
        marca = "  <- primario (E1)" if ts == SELLADO_PRIMARIO else ""
        print(f"\n{ts}{marca}")
        for s, v in a["por_semilla"].items():
            print(f"  semilla {s}: {v:+.4f}")
        sd = a["sd_entre_semillas"]
        print(f"  confirmatorio (n={a['n_semillas']}): {a['estimando_confirmatorio']:+.4f}"
              + (f"   sd entre semillas {sd:.4f}" if sd is not None else "")
              + f"   rango [{a['rango'][0]:+.4f}, {a['rango'][1]:+.4f}]")
        print(f"  semilla 42 (screening, NO entra): {a['semilla_42_screening']:+.4f}")

    desvios = [(s, por_semilla[s]["datos"]) for s in SEMILLAS_CONFIRMATORIAS
               if por_semilla[s]["datos"]["total_excluidos"]]
    if desvios:
        print("\n=== desvios de datos (seccion 7) ===")
        for s, d in desvios:
            print(f"  semilla {s}: {d['total_excluidos']} pares excluidos por NaN "
                  f"{d['excluidos_por_nan']}")
            for e in d["epocas_invalidas"]:
                print(f"    epoca invalida: {e['test_set']} ep{e['epoch']} "
                      f"({e['n_excluidos']} > {MAX_EXCLUIDOS_POR_EPOCA})")
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
