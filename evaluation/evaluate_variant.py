"""
evaluation/evaluate_variant.py

Evalúa una variante entrenada (V1, V2, ...) sobre el test set sellado.
Reporta métricas globales + por bucket de SNR + por categoría de ruido.

USO:
    # Solo métricas (rápido, no guarda audios)
    python -m evaluation.evaluate_variant --variant v1
    
    # Métricas + guardar audios estimados
    python -m evaluation.evaluate_variant --variant v1 --save_audio
"""
import argparse
import json
import torch
import numpy as np
import torchaudio
from pathlib import Path
from collections import defaultdict

from models.crn import CRN
from stft import STFTHelper
from evaluation.metrics import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

def _make_json_serializable(obj):
    """Convierte numpy types a Python nativos para JSON."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    return obj


def load_variant(variant_name, device):
    """Carga checkpoint de una variante entrenada."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / variant_name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CRN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"✓ {variant_name.upper()} cargado")
    print(f"  Época: {ckpt['epoch']}")
    print(f"  Val loss: {ckpt['val_loss']:.4f}")
    return model, ckpt


def infer_pair(model, stft, noisy_wav, device):
    """Aplica el modelo a un par noisy → enhanced."""
    with torch.no_grad():
        noisy = noisy_wav.to(device)
        if noisy.dim() == 1:
            noisy = noisy.unsqueeze(0)
        mag_noisy, phase_noisy = stft.to_spec(noisy)
        mag_est = model(mag_noisy)
        audio_est = stft.from_spec(mag_est, phase_noisy, length=noisy.shape[-1])
        return audio_est.squeeze(0).cpu()


def evaluate_variant(variant_name, test_dir, metadata_path, save_audio=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"EVALUACIÓN: {variant_name.upper()} sobre test set sellado")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Test set: {test_dir.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    
    # Cargar modelo
    model, ckpt = load_variant(variant_name, device)
    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)
    
    # Cargar metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    pair_meta = {p["id"]: p for p in metadata["pairs"]}
    print(f"\nPares a evaluar: {len(pair_meta)}")
    
    # Directorio de salida para audios estimados (opcional)
    estimates_dir = None
    if save_audio:
        estimates_dir = PROJECT_ROOT / "data" / "estimates" / variant_name
        estimates_dir.mkdir(parents=True, exist_ok=True)
        print(f"Guardando audios en: {estimates_dir.relative_to(PROJECT_ROOT)}")
    
    # Procesar cada par
    all_results = []
    print(f"\n{'-'*70}")
    print(f"Procesando pares...")
    print(f"{'-'*70}")
    
    for pair_id in sorted(pair_meta.keys()):
        pair_dir = test_dir / f"pair_{pair_id:04d}"
        if not pair_dir.exists():
            print(f"  ⚠ Falta pair_{pair_id:04d}, skip")
            continue
        
        # Cargar noisy y clean
        noisy, sr_n = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, sr_c = torchaudio.load(str(pair_dir / "clean.wav"))
        assert sr_n == sr_c == SAMPLE_RATE, f"SR incorrecto en pair_{pair_id}"
        
        # Inferencia
        enhanced = infer_pair(model, stft, noisy.squeeze(0), device)
        
        # Guardar audio si se pidió
        if estimates_dir is not None:
            out_pair = estimates_dir / f"pair_{pair_id:04d}"
            out_pair.mkdir(exist_ok=True)
            torchaudio.save(str(out_pair / "clean.wav"), clean, SAMPLE_RATE)
            torchaudio.save(str(out_pair / "noisy.wav"), noisy, SAMPLE_RATE)
            torchaudio.save(str(out_pair / "enhanced.wav"),
                            enhanced.unsqueeze(0), SAMPLE_RATE)
        
        # Métricas del enhanced (usa metrics.py — DRY)
        m_est = compute_metrics(clean.squeeze(0), enhanced)
        
        # Métricas de noisy sin procesar (baseline por par)
        m_noisy = compute_metrics(clean.squeeze(0), noisy.squeeze(0))
        
        result = {
            "pair_id": pair_id,
            "bucket_idx": pair_meta[pair_id]["bucket_idx"],
            "snr_db": pair_meta[pair_id]["snr_db"],
            "noise_category": pair_meta[pair_id]["noise_category"],
            "pesq_nb_noisy": m_noisy["PESQ-NB"],
            "pesq_nb_est": m_est["PESQ-NB"],
            "pesq_nb_delta": m_est["PESQ-NB"] - m_noisy["PESQ-NB"],
            "pesq_wb_noisy": m_noisy["PESQ-WB"],
            "pesq_wb_est": m_est["PESQ-WB"],
            "pesq_wb_delta": m_est["PESQ-WB"] - m_noisy["PESQ-WB"],
            "stoi_noisy": m_noisy["STOI"],
            "stoi_est": m_est["STOI"],
            "stoi_delta": m_est["STOI"] - m_noisy["STOI"],
            "sisdr_noisy": m_noisy["SI-SDR"],
            "sisdr_est": m_est["SI-SDR"],
            "sisdr_delta": m_est["SI-SDR"] - m_noisy["SI-SDR"],
        }
        all_results.append(result)
        
        if (pair_id + 1) % 25 == 0:
            print(f"  {pair_id + 1}/{len(pair_meta)} procesados")
    
    # ─── Análisis GLOBAL ───
    print(f"\n{'='*70}")
    print(f"RESULTADOS GLOBALES ({variant_name.upper()}, n={len(all_results)})")
    print(f"{'='*70}")
    
    def stats(key):
        vals = np.array([r[key] for r in all_results if not np.isnan(r[key])])
        return vals.mean(), vals.std()
    
    for metric in ["pesq_nb", "pesq_wb", "stoi", "sisdr"]:
        noisy_mean, noisy_std = stats(f"{metric}_noisy")
        est_mean, est_std = stats(f"{metric}_est")
        delta_mean, _ = stats(f"{metric}_delta")
        
        unit = " dB" if metric == "sisdr" else ""
        print(f"\n  {metric.upper().replace('_', '-')}")
        print(f"    Noisy:  {noisy_mean:.3f} ± {noisy_std:.3f}{unit}")
        print(f"    {variant_name.upper()}:  {est_mean:.3f} ± {est_std:.3f}{unit}")
        print(f"    Δ:      {delta_mean:+.3f}{unit}   {'✓' if delta_mean > 0 else '✗'}")
    
    # ─── Análisis POR BUCKET ───
    print(f"\n{'='*70}")
    print(f"RESULTADOS POR BUCKET DE SNR")
    print(f"{'='*70}")
    print(f"{'Bucket':<20} {'n':>4} {'PESQ-NB Δ':>10} {'STOI Δ':>10} {'SI-SDR Δ (dB)':>14}")
    print("-" * 70)
    
    by_bucket = defaultdict(list)
    for r in all_results:
        by_bucket[r["bucket_idx"]].append(r)
    
    for b_idx in sorted(by_bucket.keys()):
        b = metadata["snr_buckets"][b_idx]
        rs = by_bucket[b_idx]
        pesq_d = np.mean([r["pesq_nb_delta"] for r in rs])
        stoi_d = np.mean([r["stoi_delta"] for r in rs])
        sisdr_d = np.mean([r["sisdr_delta"] for r in rs])
        
        bucket_label = f"[{b[0]}, {b[1]}] dB"
        print(f"{bucket_label:<20} {len(rs):>4} "
              f"{pesq_d:>+10.3f} {stoi_d:>+10.3f} {sisdr_d:>+14.2f}")
    
    # ─── Análisis POR CATEGORÍA DE RUIDO ───
    print(f"\n{'='*70}")
    print(f"RESULTADOS POR CATEGORÍA DE RUIDO")
    print(f"{'='*70}")
    print(f"{'Categoría':<15} {'n':>4} {'PESQ-NB Δ':>10} {'STOI Δ':>10} {'SI-SDR Δ (dB)':>14}")
    print("-" * 60)
    
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r["noise_category"]].append(r)
    
    for cat in sorted(by_cat.keys()):
        rs = by_cat[cat]
        pesq_d = np.mean([r["pesq_nb_delta"] for r in rs])
        stoi_d = np.mean([r["stoi_delta"] for r in rs])
        sisdr_d = np.mean([r["sisdr_delta"] for r in rs])
        
        print(f"{cat:<15} {len(rs):>4} "
              f"{pesq_d:>+10.3f} {stoi_d:>+10.3f} {sisdr_d:>+14.2f}")
    
    # ─── Guardar resultados JSON ───
    output_json = PROJECT_ROOT / "results" / f"{variant_name}_test_sealed.json"
    output_json.parent.mkdir(exist_ok=True)
    
    global_stats = {}
    for metric in ["pesq_nb", "pesq_wb", "stoi", "sisdr"]:
        m_noisy, s_noisy = stats(f"{metric}_noisy")
        m_est, s_est = stats(f"{metric}_est")
        m_delta, _ = stats(f"{metric}_delta")
        global_stats[metric] = {
            "noisy_mean": float(m_noisy), "noisy_std": float(s_noisy),
            "est_mean": float(m_est), "est_std": float(s_est),
            "delta_mean": float(m_delta),
        }
    
    by_bucket_summary = []
    for b_idx in sorted(by_bucket.keys()):
        rs = by_bucket[b_idx]
        by_bucket_summary.append({
            "bucket_idx": b_idx,
            "snr_range_db": metadata["snr_buckets"][b_idx],
            "n_pairs": len(rs),
            "pesq_nb_delta_mean": float(np.mean([r["pesq_nb_delta"] for r in rs])),
            "stoi_delta_mean": float(np.mean([r["stoi_delta"] for r in rs])),
            "sisdr_delta_mean": float(np.mean([r["sisdr_delta"] for r in rs])),
        })
    
    by_cat_summary = []
    for cat in sorted(by_cat.keys()):
        rs = by_cat[cat]
        by_cat_summary.append({
            "category": cat,
            "n_pairs": len(rs),
            "pesq_nb_delta_mean": float(np.mean([r["pesq_nb_delta"] for r in rs])),
            "stoi_delta_mean": float(np.mean([r["stoi_delta"] for r in rs])),
            "sisdr_delta_mean": float(np.mean([r["sisdr_delta"] for r in rs])),
        })
    
    output_data = {
        "variant": variant_name,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": float(ckpt["val_loss"]),
        "test_set": str(test_dir.relative_to(PROJECT_ROOT)),
        "n_pairs_evaluated": len(all_results),
        "global": global_stats,
        "by_bucket": by_bucket_summary,
        "by_noise_category": by_cat_summary,
        "all_pairs": all_results,
    }
    
    with open(output_json, "w") as f:
        json.dump(_make_json_serializable(output_data), f, indent=2)

    
    print(f"\n{'='*70}")
    print(f"✅ Resultados guardados en: {output_json.relative_to(PROJECT_ROOT)}")
    print(f"{'='*70}\n")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True,
                        help="Nombre de la variante (v1, v2, v3, v4, v5)")
    parser.add_argument("--test_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "test_sealed" / "v1_en"),
                        help="Directorio del test set sellado")
    parser.add_argument("--metadata", type=str,
                        default=str(PROJECT_ROOT / "seal_test_metadata" / "test_v1_metadata.json"),
                        help="Archivo de metadata del test set")
    parser.add_argument("--save_audio", action="store_true",
                        help="Guardar audios estimados en data/estimates/<variant>/")
    args = parser.parse_args()
    
    evaluate_variant(
        variant_name=args.variant,
        test_dir=Path(args.test_dir),
        metadata_path=Path(args.metadata),
        save_audio=args.save_audio,
    )

