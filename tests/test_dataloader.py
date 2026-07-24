# tests/test_dataloader.py
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import NSDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"

ds = NSDataset(TRAIN_DIR, segment_samples=64000)
print(f"Dataset size: {len(ds)} pairs")
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
for batch_idx, (noisy, clean) in enumerate(loader):
  print(f"Batch {batch_idx}: noisy={tuple(noisy.shape)}, clean={tuple(clean.shape)}")
  print(f"  noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
  print(f"  clean range: [{clean.min():.3f}, {clean.max():.3f}]")
  if batch_idx >= 2:
      break

