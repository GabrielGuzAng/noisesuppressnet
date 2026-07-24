# benchmarks/measure_rtf.py
import torch
import torchaudio
import time
from pathlib import Path

from models.crn import CRN
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def measure_rtf(model, stft, device, audio_duration_sec=4.0, n_warmup=5, n_runs=20):
    """Mide RTF de inferencia para un clip de audio_duration_sec."""
    n_samples = int(audio_duration_sec * 16000)
    x = torch.randn(1, n_samples).to(device)
    
    model.eval()
    
    # Warmup (importante: las primeras pasadas son más lentas)
    with torch.no_grad():
        for _ in range(n_warmup):
            mag, phase = stft.to_spec(x)
            est = model(mag)
            _ = stft.from_spec(est, phase, length=n_samples)
    
    # Sincronizar GPU si aplica
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Mediciones reales
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            mag, phase = stft.to_spec(x)
            est = model(mag)
            audio_out = stft.from_spec(est, phase, length=n_samples)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    
    times = sorted(times)
    median_time = times[len(times) // 2]
    p95_time = times[int(len(times) * 0.95)]
    
    rtf_median = median_time / audio_duration_sec
    rtf_p95 = p95_time / audio_duration_sec
    
    return {
        "device": str(device),
        "audio_duration_s": audio_duration_sec,
        "median_proc_time_s": median_time,
        "p95_proc_time_s": p95_time,
        "RTF_median": rtf_median,
        "RTF_p95": rtf_p95,
    }

if __name__ == "__main__":
    model = CRN()
    stft = STFTHelper(n_fft=320, hop_length=160)
    
    # Cargar checkpoint si querés mediciones del modelo entrenado
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "v0" / "best.pt")
    model.load_state_dict(ckpt["model_state"])
    
    print("=" * 60)
    print("RTF Benchmark — NoiseSuppressNet CRN v0")
    print("=" * 60)
    
    # Test en GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device)
        stft._window = torch.hamming_window(320).to(device)
        result_gpu = measure_rtf(model_gpu, stft, device)
        print(f"\nGPU ({torch.cuda.get_device_name(0)}):")
        print(f"  RTF median: {result_gpu['RTF_median']:.4f}")
        print(f"  RTF p95:    {result_gpu['RTF_p95']:.4f}")
    
    # Test en CPU (lo importante para tu spec)
    device = torch.device("cpu")
    model_cpu = model.to(device)
    stft._window = torch.hamming_window(320)
    
    torch.set_num_threads(1)  # 1 thread, como spec del proyecto
    result_cpu = measure_rtf(model_cpu, stft, device, n_runs=10)
    print(f"\nCPU (1 thread):")
    print(f"  RTF median: {result_cpu['RTF_median']:.4f}")
    print(f"  RTF p95:    {result_cpu['RTF_p95']:.4f}")
    print(f"  {'✓ Real-time capable' if result_cpu['RTF_p95'] < 1.0 else '✗ Too slow for real-time'}")
