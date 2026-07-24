# inference/infer.py
import torch
import torchaudio
from pathlib import Path

from models.crn import CRN
from stft import STFTHelper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT  = PROJECT_ROOT / "checkpoints" / "v0" / "best.pt"
VAL_DIR     = PROJECT_ROOT / "data" / "processed" / "val"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "estimates" / "v0"

SAMPLE_RATE = 16000

def load_model(checkpoint_path, device):
    model = CRN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model

def enhance_audio(model, stft, noisy_audio, device):
    """Procesa un único audio noisy y devuelve el enhanced."""
    with torch.no_grad():
        noisy = noisy_audio.unsqueeze(0).to(device)  # [1, T]
        mag_noisy, phase_noisy = stft.to_spec(noisy)
        mag_est = model(mag_noisy)
        audio_est = stft.from_spec(mag_est, phase_noisy, length=noisy.shape[-1])
    return audio_est.squeeze(0).cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(CHECKPOINT, device)
    stft = STFTHelper(n_fft=320, hop_length=160)
    stft._window = torch.hamming_window(320).to(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_pairs = sorted(VAL_DIR.iterdir())
    print(f"Processing {len(val_pairs)} pairs...")

    for pair_dir in val_pairs:
        noisy, _ = torchaudio.load(str(pair_dir / "noisy.wav"))
        clean, _ = torchaudio.load(str(pair_dir / "clean.wav"))
        noisy = noisy.squeeze(0)
        clean = clean.squeeze(0)

        enhanced = enhance_audio(model, stft, noisy, device)

        # Guardar los tres audios juntos para comparación
        out_subdir = OUTPUT_DIR / pair_dir.name
        out_subdir.mkdir(exist_ok=True)
        torchaudio.save(str(out_subdir / "noisy.wav"),    noisy.unsqueeze(0),    SAMPLE_RATE)
        torchaudio.save(str(out_subdir / "clean.wav"),    clean.unsqueeze(0),    SAMPLE_RATE)
        torchaudio.save(str(out_subdir / "enhanced.wav"), enhanced.unsqueeze(0), SAMPLE_RATE)

    print(f"Done. Outputs at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
