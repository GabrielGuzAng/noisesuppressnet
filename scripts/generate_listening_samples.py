"""
scripts/generate_listening_samples.py

Genera un puñado de audios noisy/clean/enhanced para escuchar a mano
(no para métricas — para eso está evaluation/evaluate_variant.py
--save_audio, que corre las 250 pares completas). Pensado para correr
en CPU rápido con unos pocos pares representativos, sin competir con
entrenamientos corriendo en GPU.

USO:
    python -m scripts.generate_listening_samples --variants v1 v2
    python -m scripts.generate_listening_samples --variants v1 v2 v3 --n_per_bucket 1
    python -m scripts.generate_listening_samples --variants v3 v3b v3e --lang es
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio

from models.crn import CRN
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

LANG_CONFIG = {
    "en": {
        "test_dir": PROJECT_ROOT / "data" / "test_sealed" / "v1_en",
        "metadata": PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json",
        "out_dir": PROJECT_ROOT / "data" / "estimates" / "listen",
    },
    "es": {
        "test_dir": PROJECT_ROOT / "data" / "test_sealed" / "v2_es",
        "metadata": PROJECT_ROOT / "seal_test_metadata" / "test_v2_metadata.json",
        "out_dir": PROJECT_ROOT / "data" / "estimates" / "listen_es",
    },
}


def pick_pairs(metadata_path, n_per_bucket=1):
    with open(metadata_path) as f:
        metadata = json.load(f)
    by_bucket = {}
    for p in metadata["pairs"]:
        by_bucket.setdefault(p["bucket_idx"], []).append(p)
    picked = []
    for b in sorted(by_bucket):
        picked.extend(by_bucket[b][:n_per_bucket])
    return picked


def load_variant(variant, device):
    ckpt_path = PROJECT_ROOT / "checkpoints" / variant / "best.pt"
    model = CRN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def enhance(model, stft, noisy, device):
    with torch.no_grad():
        x = noisy.unsqueeze(0).to(device)
        mag, phase = stft.to_spec(x)
        mag_est = model(mag)
        audio_est = stft.from_spec(mag_est, phase, length=x.shape[-1])
    return audio_est.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variantes a comparar, ej: v1 v2 v3")
    parser.add_argument("--n_per_bucket", type=int, default=1,
                        help="Pares por bucket de SNR (default 1 -> 5 pares totales)")
    parser.add_argument("--lang", choices=["en", "es"], default="en",
                        help="Test set a usar: en (test_v1_en) o es (test_v2_es)")
    parser.add_argument("--cpu", action="store_true",
                        help="Forzar CPU aunque haya GPU disponible (para no competir con un training corriendo)")
    args = parser.parse_args()

    cfg = LANG_CONFIG[args.lang]
    test_dir, metadata_path, out_dir = cfg["test_dir"], cfg["metadata"], cfg["out_dir"]

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Device: {device} | idioma: {args.lang} | test set: {test_dir.relative_to(PROJECT_ROOT)}")

    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)

    models = {v: load_variant(v, device) for v in args.variants}
    print(f"Modelos cargados: {list(models.keys())}")

    pairs = pick_pairs(metadata_path, args.n_per_bucket)
    print(f"Pares elegidos ({len(pairs)}): {[p['id'] for p in pairs]}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        pair_id = p["id"]
        pair_dir = test_dir / f"pair_{pair_id:04d}"
        noisy, _ = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair_dir / "clean.wav"))
        noisy = noisy.squeeze(0)
        clean = clean.squeeze(0)

        pair_out_dir = out_dir / f"pair_{pair_id:04d}_snr{p['snr_db']:.1f}dB"
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(pair_out_dir / "noisy.wav"), noisy.unsqueeze(0), SAMPLE_RATE)
        torchaudio.save(str(pair_out_dir / "clean.wav"), clean.unsqueeze(0), SAMPLE_RATE)

        for variant, model in models.items():
            enhanced = enhance(model, stft, noisy, device)
            torchaudio.save(str(pair_out_dir / f"{variant}_enhanced.wav"),
                             enhanced.unsqueeze(0), SAMPLE_RATE)

        print(f"  pair_{pair_id:04d} (SNR {p['snr_db']:.1f} dB, bucket {p['bucket_idx']}) -> {pair_out_dir.relative_to(PROJECT_ROOT)}")

    print(f"\nListo. Audios en: {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
