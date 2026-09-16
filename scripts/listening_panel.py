"""
scripts/listening_panel.py

Genera el panel de escucha ciega y su planilla de valoración.

A diferencia de scripts/generate_listening_samples.py (chequeo informal rápido),
esto produce material para la validación subjetiva de la rama 8 de la EDT:

1. SELECCIÓN CURADA en tres grupos, no un muestreo al azar:
   - A (test_v2_es): el hallazgo. Buckets 0, 2 y 4 — el rango donde la ganancia
     en español decae con el SNR.
   - B (test_v1_en): el control de idioma. Los MISMOS pair_id que el grupo A, y
     como los dos sellados comparten snr_db par por par, cada clip de B tiene
     exactamente el mismo SNR que su contraparte de A.
   - C (test_v1_en): los archivos que las tres variantes de fine-tuning al
     español rompen a la vez (delta PESQ-NB < -0.2 contra V1 en v3, v3b y v3e).
     Ni el bucket ni la categoría de ruido los distinguen del resto, así que qué
     tienen en común solo puede salir de escucharlos.

2. CIEGO: dentro de cada clip las versiones se escriben con nombres codificados
   (A, B, C...) permutados con una semilla propia por clip. El mapeo va a
   key.csv, que no hay que abrir hasta terminar de valorar. Sin esto lo que se
   registra es la expectativa, no la percepción.

3. ESCALAS ITU-T P.835 (SIG, BAK, OVRL, 1-5), que es el estándar para evaluación
   subjetiva de supresión de ruido y es lo que estima DNSMOS. Usar las mismas
   escalas permite después correlacionar la escucha propia contra DNSMOS.

USO:
    python -m scripts.listening_panel --variants v1 v2 v3e
    python -m scripts.listening_panel --variants v1 v2 v3e v5 --cpu
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torchaudio

from models.crn import CRN
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
SEED = 42

OUT_DIR = PROJECT_ROOT / "data" / "estimates" / "listening_panel"

SETS = {
    "v1_en": {
        "dir": PROJECT_ROOT / "data" / "test_sealed" / "v1_en",
        "metadata": PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json",
    },
    "v2_es": {
        "dir": PROJECT_ROOT / "data" / "test_sealed" / "v2_es",
        "metadata": PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json",
    },
}

# Buckets del grupo A (y por reflejo del B): peor, medio y el de SNR alto donde
# aparece la degradación en español.
GROUP_A_BUCKETS = (0, 2, 4)
N_BROKEN = 3          # cuántos archivos rotos entran en el grupo C
BROKEN_THRESHOLD = -0.2
BROKEN_VARIANTS = ("v3", "v3b", "v3e")


def load_pairs(metadata_path: Path) -> dict[int, dict]:
    with open(metadata_path) as f:
        return {p["id"]: p for p in json.load(f)["pairs"]}


def pick_group_a(meta_es: dict[int, dict], rng: random.Random) -> list[int]:
    """One pair per target bucket, favouring noise-category variety."""
    chosen: list[int] = []
    used_categories: set[str] = set()
    for bucket in GROUP_A_BUCKETS:
        candidates = sorted(p["id"] for p in meta_es.values() if p["bucket_idx"] == bucket)
        fresh = [i for i in candidates if meta_es[i]["noise_category"] not in used_categories]
        pool = fresh or candidates
        pick = rng.choice(pool)
        used_categories.add(meta_es[pick]["noise_category"])
        chosen.append(pick)
    return chosen


def find_broken(n: int) -> list[int]:
    """pair_ids of test_v1_en that every ES fine-tune degrades beyond the threshold."""
    results_dir = PROJECT_ROOT / "results"
    base_path = results_dir / "v1_v1_en.json"
    if not base_path.exists():
        return []
    with open(base_path) as f:
        base = {r["pair_id"]: r["pesq_nb_est"] for r in json.load(f)["all_pairs"]}

    broken_sets = []
    for variant in BROKEN_VARIANTS:
        path = results_dir / f"{variant}_v1_en.json"
        if not path.exists():
            return []
        with open(path) as f:
            other = {r["pair_id"]: r["pesq_nb_est"] for r in json.load(f)["all_pairs"]}
        broken_sets.append({
            i for i in base if other[i] - base[i] < BROKEN_THRESHOLD
        })
    triple = sorted(set.intersection(*broken_sets))
    # Reparte por bucket para no llevarse tres del mismo régimen de SNR.
    meta = load_pairs(SETS["v1_en"]["metadata"])
    triple.sort(key=lambda i: (meta[i]["bucket_idx"], i))
    step = max(1, len(triple) // n)
    return triple[::step][:n]


def load_variant(variant: str, device: torch.device) -> CRN:
    ckpt_path = PROJECT_ROOT / "checkpoints" / variant / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint {ckpt_path}")
    model = CRN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def enhance(model: CRN, stft: STFTHelper, noisy: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    x = noisy.unsqueeze(0).to(device)
    mag, phase = stft.to_spec(x)
    est = model(mag)
    return stft.from_spec(est, phase, length=x.shape[-1]).squeeze(0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variantes a incluir, ej: v1 v2 v3e")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)

    meta = {name: load_pairs(cfg["metadata"]) for name, cfg in SETS.items()}
    rng = random.Random(SEED)

    group_a = pick_group_a(meta["v2_es"], rng)
    group_b = list(group_a)                    # mismos ids -> mismo SNR
    group_c = find_broken(N_BROKEN)

    clips: list[dict] = []
    for pid in group_a:
        clips.append({"grupo": "A", "set": "v2_es", "pair_id": pid,
                      "por_que": "el hallazgo: ganancia en español según SNR"})
    for pid in group_b:
        clips.append({"grupo": "B", "set": "v1_en", "pair_id": pid,
                      "por_que": "control de idioma: mismo SNR que su par del grupo A"})
    for pid in group_c:
        clips.append({"grupo": "C", "set": "v1_en", "pair_id": pid,
                      "por_que": "archivo que las tres variantes ES rompen"})

    if not group_c:
        print("AVISO: no se pudo armar el grupo C (faltan results/*_v1_en.json)")

    print(f"Clips: {len(clips)}  (A={len(group_a)} B={len(group_b)} C={len(group_c)})")
    print(f"Versiones por clip: {2 + len(args.variants)} (limpio, ruidoso, {', '.join(args.variants)})")
    print(f"Dispositivo: {device}")
    print()

    models = {v: load_variant(v, device) for v in args.variants}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_rows, sheet_rows = [], []

    for clip_idx, clip in enumerate(clips):
        pid = clip["pair_id"]
        info = meta[clip["set"]][pid]
        pair_dir = SETS[clip["set"]]["dir"] / f"pair_{pid:04d}"
        noisy, _ = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair_dir / "clean.wav"))
        noisy, clean = noisy.squeeze(0), clean.squeeze(0)

        versions = {"limpio": clean, "ruidoso": noisy}
        for name, model in models.items():
            versions[name] = enhance(model, stft, noisy, device)

        # Ciego: permutación propia por clip, reproducible.
        labels = sorted(versions)
        random.Random(SEED + clip_idx).shuffle(labels)

        clip_name = f"clip_{clip_idx:02d}_{clip['grupo']}"
        clip_dir = OUT_DIR / clip_name
        clip_dir.mkdir(exist_ok=True)

        for code_idx, real_name in enumerate(labels):
            code = chr(ord("A") + code_idx)
            torchaudio.save(str(clip_dir / f"{code}.wav"),
                            versions[real_name].unsqueeze(0), SAMPLE_RATE)
            key_rows.append({
                "clip": clip_name, "codigo": code, "version_real": real_name,
                "grupo": clip["grupo"], "test_set": clip["set"], "pair_id": pid,
            })
            sheet_rows.append({
                "clip": clip_name,
                "grupo": clip["grupo"],
                "test_set": clip["set"],
                "pair_id": pid,
                "snr_db": info["snr_db"],
                "bucket": info["bucket_idx"],
                "ruido": info["noise_category"],
                "codigo": code,
                "SIG_1a5": "",
                "BAK_1a5": "",
                "OVRL_1a5": "",
                "artefactos": "",
                "notas": "",
            })
        print(f"  {clip_name}  set={clip['set']} pair={pid:3d} "
              f"bucket={info['bucket_idx']} snr={info['snr_db']:+.1f} dB "
              f"ruido={info['noise_category']}  -> {len(labels)} versiones")

    with open(OUT_DIR / "planilla.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet_rows[0]))
        w.writeheader()
        w.writerows(sheet_rows)

    with open(OUT_DIR / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)

    with open(OUT_DIR / "INSTRUCCIONES.txt", "w", encoding="utf-8") as f:
        f.write(
            "PANEL DE ESCUCHA CIEGA — NoiseSuppressNet\n"
            "=========================================\n\n"
            "NO ABRAS key.csv hasta terminar de valorar todo. Ese archivo dice qué\n"
            "versión es cada letra; leerlo antes convierte la escucha en confirmación\n"
            "de expectativas.\n\n"
            "Para cada clip vas a encontrar una carpeta con A.wav, B.wav, ... Escuchá\n"
            "las versiones de un mismo clip una tras otra, cuantas veces haga falta, y\n"
            "completá una fila por letra en planilla.csv.\n\n"
            "Escalas ITU-T P.835 (1 a 5, enteros):\n\n"
            "  SIG  — calidad de la SEÑAL DE VOZ sola, ignorando el fondo.\n"
            "         1 muy degradada / distorsionada   5 nada degradada\n\n"
            "  BAK  — cuánto MOLESTA el ruido de fondo.\n"
            "         1 muy intrusivo   5 nada perceptible\n\n"
            "  OVRL — calidad GENERAL del audio, todo junto.\n"
            "         1 mala   5 excelente\n\n"
            "artefactos: texto libre. Palabras útiles: metálico, robótico, ruido\n"
            "  musical, cortes, chasquidos, voz apagada, siseo, eco, vocales raras.\n"
            "  Si no escuchás nada raro, escribí 'ninguno'.\n\n"
            "notas: cualquier cosa. Especialmente en el grupo C, donde la pregunta es\n"
            "  qué tienen en común esos archivos.\n\n"
            "Consejos: usá auriculares, volumen fijo durante toda la sesión, y hacé\n"
            "un descanso cada 15-20 minutos. Si dudás entre dos valores, poné el más\n"
            "bajo y aclaralo en notas.\n"
        )

    print()
    print(f"Audios:       {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Planilla:     {(OUT_DIR / 'planilla.csv').relative_to(PROJECT_ROOT)}")
    print(f"Clave (NO abrir todavía): {(OUT_DIR / 'key.csv').relative_to(PROJECT_ROOT)}")
    print(f"Instrucciones: {(OUT_DIR / 'INSTRUCCIONES.txt').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
