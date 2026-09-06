"""Registra potencia y utilización GPU cada 10 segundos durante training."""
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

def log_gpu(output_path, interval_s=10):
    start = datetime.now()
    log = []
    try:
        while True:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            values = result.stdout.strip().split(", ")
            log.append({
                "timestamp": datetime.now().isoformat(),
                "seconds_since_start": (datetime.now() - start).total_seconds(),
                "power_w": float(values[0]),
                "gpu_util_pct": float(values[1]),
                "vram_mib": float(values[2]),
                "temp_c": float(values[3]),
            })
            time.sleep(interval_s)
    except KeyboardInterrupt:
        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n✓ Log guardado en {output_path}: {len(log)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    log_gpu(args.output, args.interval)
