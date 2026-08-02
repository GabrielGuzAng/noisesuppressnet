# evaluation/metrics.py
import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from pathlib import Path
import pandas as pd

SAMPLE_RATE = 16000

def si_sdr(reference, estimation, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio en dB."""
    ref = reference - np.mean(reference)
    est = estimation - np.mean(estimation)
    
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    
    return 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + eps) + eps)

def compute_metrics(reference_audio, processed_audio, sr=SAMPLE_RATE):
    """Devuelve dict con PESQ-NB, PESQ-WB, STOI, SI-SDR."""
    ref = reference_audio.numpy() if torch.is_tensor(reference_audio) else reference_audio
    proc = processed_audio.numpy() if torch.is_tensor(processed_audio) else processed_audio
    
    # Asegurar misma longitud
    L = min(len(ref), len(proc))
    ref, proc = ref[:L], proc[:L]
    
    try:
        pesq_nb = pesq(sr, ref, proc, mode='nb')
    except Exception:
        pesq_nb = np.nan
    
    try:
        pesq_wb = pesq(sr, ref, proc, mode='wb')
    except Exception:
        pesq_wb = np.nan
    
    try:
        stoi_val = stoi(ref, proc, sr, extended=False)
    except Exception:
        stoi_val = np.nan
    
    sisdr_val = si_sdr(ref, proc)
    
    return {
        "PESQ-NB": pesq_nb,
        "PESQ-WB": pesq_wb,
        "STOI": stoi_val,
        "SI-SDR": sisdr_val,
    }


def evaluate_directory(estimates_dir, ref_filename="clean.wav", proc_filename="enhanced.wav"):
    """Evalúa todos los pares de un directorio."""
    results = []
    for pair_dir in sorted(Path(estimates_dir).iterdir()):
        if not pair_dir.is_dir():
            continue
        ref, _ = torchaudio.load(str(pair_dir / ref_filename))
        proc, _ = torchaudio.load(str(pair_dir / proc_filename))
        m = compute_metrics(ref.squeeze(0), proc.squeeze(0))
        m["pair"] = pair_dir.name
        results.append(m)
    
    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"Evaluation: {estimates_dir.name} ({proc_filename})")
    print(f"{'='*50}")
    print(f"PESQ-NB:  {df['PESQ-NB'].mean():.3f} ± {df['PESQ-NB'].std():.3f}")
    print(f"PESQ-WB:  {df['PESQ-WB'].mean():.3f} ± {df['PESQ-WB'].std():.3f}")
    print(f"STOI:     {df['STOI'].mean():.3f} ± {df['STOI'].std():.3f}")
    print(f"SI-SDR:   {df['SI-SDR'].mean():.2f} ± {df['SI-SDR'].std():.2f} dB")
    return df


