""" # scripts/analyze_cv26_es.py
from pathlib import Path
import pandas as pd
import torchaudio


CV_ROOT = Path("data/raw/clean_speech/es/cv-corpus-26.0-2026-06-12/es/clips/")

# Cargar metadata validada
val_meta = pd.read_csv(CV_ROOT / "validated.tsv", sep="\t")
print(f"Clips validados totales: {len(val_meta)}")

# Verificar sample rate real en muestra
sample = val_meta.sample(20, random_state=42)
for _, row in sample.iterrows():
    audio_path = CV_ROOT / "clips" / row["path"]
    info = torchaudio.info(str(audio_path))
    print(f"  {row['path']}: {info.sample_rate} Hz, {info.num_channels} canal(es), "
          f"{info.num_frames/info.sample_rate:.2f}s")

# Distribución de duración
# (opcional: recorrer todo el dataset y calcular) """

#!/usr/bin/env python3
"""
scripts/analyze_cv26_es.py

Analiza cv-corpus-26.0-2026-06-12 (es) y genera manifiestos limpios por split.

Cambios respecto de la version original:
  - CV_ROOT apunta a es/ (no a es/clips/), que era el bug de rutas.
  - Usa train/dev/test oficiales en vez de validated.tsv, porque son
    disjuntos por hablante y por frase (verificado: 0 solapamiento).
  - Toma duraciones de clip_durations.tsv en vez de abrir cada mp3.
  - Excluye frases marcadas en reported.tsv.
  - Verifica sample rate real solo sobre una muestra chica.
"""

from pathlib import Path
import argparse
import pandas as pd

# --- Directorio raiz del idioma: contiene los .tsv y la carpeta clips/ ---
CV_ROOT = Path("data/raw/clean_speech/es/cv-corpus-26.0-2026-06-12/es")
CLIPS_DIR = CV_ROOT / "clips"
OUT_DIR = Path("data/interim/cv26_es")

# --- Filtros de calidad (ajustalos a tu caso) ---
MIN_DUR_S = 1.0
MAX_DUR_S = 12.0
MIN_UP_VOTES = 2
MAX_DOWN_VOTES = 0      # 13.1% de los validados tienen down_votes > 0
DROP_REPORTED = True

CLIP_COLS = [
    "client_id", "path", "sentence_id", "sentence",
    "up_votes", "down_votes", "age", "gender", "accents", "locale",
]


def read_tsv(path: Path, usecols=None) -> pd.DataFrame:
    """Lectura robusta: sin conversion automatica de NA (hay frases que son
    literalmente 'NA' y quedarian como NaN)."""
    return pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=str,
        keep_default_na=False, na_values=[], encoding="utf-8",
    )


def load_durations() -> pd.DataFrame:
    dur = read_tsv(CV_ROOT / "clip_durations.tsv")
    dur.columns = ["path", "duration_ms"]
    dur["duration_s"] = dur["duration_ms"].astype("int64") / 1000.0
    return dur[["path", "duration_s"]]


def load_reported_ids() -> set:
    rep = read_tsv(CV_ROOT / "reported.tsv", usecols=["sentence_id", "reason"])
    print("\n[reported.tsv] motivos principales:")
    print(rep["reason"].value_counts().head(5).to_string())
    return set(rep["sentence_id"])


def load_split(name: str, dur: pd.DataFrame, reported: set) -> pd.DataFrame:
    df = read_tsv(CV_ROOT / f"{name}.tsv", usecols=CLIP_COLS)
    n0 = len(df)

    df = df.merge(dur, on="path", how="left")
    missing = df["duration_s"].isna().sum()
    if missing:
        print(f"  [{name}] AVISO: {missing} clips sin duracion en clip_durations.tsv")

    df["up_votes"] = df["up_votes"].astype("int32")
    df["down_votes"] = df["down_votes"].astype("int32")

    keep = (
        df["duration_s"].between(MIN_DUR_S, MAX_DUR_S)
        & (df["up_votes"] >= MIN_UP_VOTES)
        & (df["down_votes"] <= MAX_DOWN_VOTES)
    )
    if DROP_REPORTED:
        keep &= ~df["sentence_id"].isin(reported)

    out = df[keep].copy()
    out["filepath"] = out["path"].map(lambda p: str(CLIPS_DIR / p))

    horas = out["duration_s"].sum() / 3600
    print(f"  [{name}] {n0:>7,} -> {len(out):>7,} clips  "
          f"({horas:6.1f} h, {out['client_id'].nunique():,} hablantes)")
    return out


def check_leakage(splits: dict) -> None:
    print("\n=== Chequeo de fuga entre splits ===")
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sp = len(set(splits[a]["client_id"]) & set(splits[b]["client_id"]))
            se = len(set(splits[a]["sentence_id"]) & set(splits[b]["sentence_id"]))
            flag = "OK" if sp == 0 and se == 0 else "!!"
            print(f"  {flag} {a} vs {b}: {sp} hablantes, {se} frases en comun")


def check_audio_sample(df: pd.DataFrame, n: int = 20, seed: int = 42) -> None:
    """Verifica sample rate / canales reales. Ojo: para MP3, torchaudio puede
    estimar num_frames a partir del bitrate, asi que la duracion confiable es
    la de clip_durations.tsv (columna duration_s)."""
    try:
        import torchaudio
    except ImportError:
        print("\n[audio] torchaudio no instalado, salteo la verificacion.")
        return

    print(f"\n=== Verificacion de formato ({n} clips al azar) ===")
    rates, chans = set(), set()
    for _, row in df.sample(min(n, len(df)), random_state=seed).iterrows():
        p = Path(row["filepath"])
        if not p.exists():
            print(f"  FALTA: {p}")
            continue
        info = torchaudio.info(str(p))
        rates.add(info.sample_rate)
        chans.add(info.num_channels)
        print(f"  {row['path']}: {info.sample_rate} Hz, {info.num_channels} ch, "
              f"{row['duration_s']:.2f}s (tsv) vs "
              f"{info.num_frames / info.sample_rate:.2f}s (torchaudio)")
    print(f"  sample rates observados: {sorted(rates)} | canales: {sorted(chans)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio-check", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dur = load_durations()
    print(f"[clip_durations] {len(dur):,} clips, "
          f"{dur['duration_s'].sum() / 3600:.1f} h en todo el corpus")

    reported = load_reported_ids() if DROP_REPORTED else set()

    print("\n=== Splits oficiales (disjuntos por hablante y por frase) ===")
    splits = {n: load_split(n, dur, reported) for n in ("train", "dev", "test")}

    check_leakage(splits)

    for name, df in splits.items():
        cols = ["filepath", "sentence", "duration_s", "client_id",
                "sentence_id", "age", "gender", "accents"]
        out = OUT_DIR / f"{name}_manifest.tsv"
        df[cols].to_csv(out, sep="\t", index=False)
        print(f"[write] {out}")

    print("\n=== Distribucion de acentos en train ===")
    acc = splits["train"]["accents"].replace("", "(sin declarar)")
    print((acc.value_counts(normalize=True).head(8) * 100).round(1).to_string())

    if not args.no_audio_check:
        check_audio_sample(splits["train"])


if __name__ == "__main__":
    main()