"""
scripts/select_by_val_pesq.py

Selecciona la época de una corrida por PESQ-NB sobre el set de VALIDACIÓN,
no por mínimo de `val_loss`.

POR QUÉ EXISTE. El criterio de `Trainer` es la val_loss mínima, y hay cuatro
evidencias propias de que no está alineado con PESQ:

  - V4b: el placebo en la época 2 superaba a V1 en las cuatro métricas pese a
    peor val_loss.
  - V5 réplica s43: best.pt en la época 12, pero la 18 daba +0.036 PESQ-NB.
  - V5 réplica s44: best.pt en la época 16, pero la 18 daba +0.022.
  - V6: los dos best.pt cayeron en la época 2, que resultó ser de las peores de
    las seis para las dos ramas a la vez.

Además la meseta de val_loss de V2 rebota ±0.02 época a época sin tendencia, así
que su mínimo es un golpe de suerte y no convergencia.

QUÉ NO HACE. No toca los test sets sellados. La selección sale del set de
validación de la propia corrida; usar el sellado para elegir una época sería
contaminar exactamente el endpoint que después se reporta.

TAMAÑO DE MUESTRA, medido y no estimado a ojo. Lo que importa no es la sd de
PESQ-NB (0.73 por par) sino la sd de la DIFERENCIA apareada entre checkpoints de
la misma corrida sobre los mismos archivos, que es 0.059 -- doce veces menor.
Con 300 pares el error estándar del contraste es 0.0034, seis veces menor que
las diferencias de ~0.02 que separan a las épocas entre sí. 300 es el default y
alcanza; con 100 ya se está en 0.0059, al límite.

LIMITACIÓN CONCEPTUAL, importante para las afirmaciones cross-lingual. Este
criterio optimiza calidad EN DOMINIO: elige el checkpoint que mejor rinde sobre
el idioma con el que se entrenó. Para una afirmación fuera de dominio, el mejor
checkpoint en dominio no tiene por qué ser el mejor fuera. Usar el idioma
objetivo para elegir sería usar el dominio objetivo, así que no se hace -- pero
hay que reportar que la selección es en dominio y que eso es una elección, no una
propiedad.

COSTO. Cero horas de GPU de entrenamiento: se aplica a posteriori sobre los
checkpoints `epoch_NN.pt` que ya se guardan con `save_every_n_epochs`.
Medido: 0.46 s por par y por checkpoint. 300 pares x 6 checkpoints = ~14 min.

USO:
    python -m scripts.select_by_val_pesq --checkpoint_dir checkpoints/v6_gate
    python -m scripts.select_by_val_pesq --checkpoint_dir checkpoints/v7_gate \
        --val_dir data/processed/val --n_pairs 300
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

from evaluation.metrics import compute_metrics
from models.crn import CRN
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
SUBSET_SEED = 42


def pick_pairs(val_dir: Path, n_pairs: int) -> list[Path]:
    """Fixed, seeded subset of validation pairs.

    The same subset for every checkpoint and every run, so the comparison is
    paired and the selection is not a lottery over which files were scored.
    """
    pairs = sorted(p for p in val_dir.iterdir() if p.is_dir())
    if n_pairs >= len(pairs):
        return pairs
    rng = np.random.default_rng(SUBSET_SEED)
    idx = rng.choice(len(pairs), size=n_pairs, replace=False)
    return [pairs[i] for i in sorted(idx)]


def load_model(ckpt_path: Path, device) -> CRN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    has_gate = any(k.startswith("conv_gate") for k in ckpt["model_state"])
    model = CRN(gate=has_gate).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_checkpoint(ckpt_path: Path, pairs: list[Path], stft, device) -> dict:
    model, ckpt = load_model(ckpt_path, device)
    scores, gates = [], []
    for pair in pairs:
        noisy, sr = torchaudio.load(str(pair / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair / "clean.wav"))
        assert sr == SAMPLE_RATE, f"SR incorrecto en {pair}"
        mag, phase = stft.to_spec(noisy.to(device))
        mag_est, g = model(mag, return_gate=True)
        est = stft.from_spec(mag_est, phase, length=noisy.shape[-1]).squeeze(0).cpu()
        m = compute_metrics(clean.squeeze(0), est)
        if not np.isnan(m["PESQ-NB"]):
            scores.append(m["PESQ-NB"])
        if g is not None:
            gates.append(float(g.mean()))
    return {
        "checkpoint": ckpt_path.name,
        "epoch": int(ckpt["epoch"]),
        "val_loss": float(ckpt["val_loss"]),
        "val_pesq_nb": float(np.mean(scores)),
        "n_scored": len(scores),
        "gate_mean": float(np.mean(gates)) if gates else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--val_dir", default=str(PROJECT_ROOT / "data" / "processed" / "val"))
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    epochs = sorted(ckpt_dir.glob("epoch_*.pt"),
                    key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)))
    if not epochs:
        raise SystemExit(f"No hay checkpoints epoch_*.pt en {ckpt_dir}. "
                         "La corrida necesita save_every_n_epochs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)
    pairs = pick_pairs(Path(args.val_dir), args.n_pairs)

    print(f"{'='*72}\nSELECCIÓN POR PESQ-NB SOBRE VALIDACIÓN\n{'='*72}")
    print(f"Corrida:    {ckpt_dir}")
    print(f"Validación: {args.val_dir}  ({len(pairs)} pares, semilla {SUBSET_SEED})")
    print(f"Checkpoints: {len(epochs)}\n")
    print(f"  {'época':>6s}{'val_loss':>12s}{'val PESQ-NB':>14s}{'media g':>10s}")

    rows = []
    for p in epochs:
        r = score_checkpoint(p, pairs, stft, device)
        rows.append(r)
        g = f"{r['gate_mean']:.4f}" if r["gate_mean"] is not None else "--"
        print(f"  {r['epoch']:6d}{r['val_loss']:+12.5f}{r['val_pesq_nb']:14.4f}{g:>10s}")

    by_pesq = max(rows, key=lambda r: r["val_pesq_nb"])
    by_loss = min(rows, key=lambda r: r["val_loss"])
    print(f"\n  Elegida por PESQ sobre validación : época {by_pesq['epoch']} "
          f"(PESQ {by_pesq['val_pesq_nb']:.4f})")
    print(f"  Elegida por mínimo de val_loss    : época {by_loss['epoch']} "
          f"(PESQ {by_loss['val_pesq_nb']:.4f})")
    delta = by_pesq["val_pesq_nb"] - by_loss["val_pesq_nb"]
    if by_pesq["epoch"] == by_loss["epoch"]:
        print("  Los dos criterios coinciden en esta corrida.")
    else:
        print(f"  El criterio viejo deja {delta:+.4f} PESQ-NB sobre la mesa.")

    out = Path(args.output) if args.output else ckpt_dir / "val_pesq_selection.json"
    with open(out, "w") as f:
        json.dump({"checkpoint_dir": str(ckpt_dir), "val_dir": args.val_dir,
                   "n_pairs": len(pairs), "subset_seed": SUBSET_SEED,
                   "selected_by_val_pesq": by_pesq["epoch"],
                   "selected_by_val_loss": by_loss["epoch"],
                   "epochs": rows}, f, indent=2)
    print(f"\n  Trayectoria guardada en {out}")


if __name__ == "__main__":
    main()
