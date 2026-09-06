"""
Verifica que el test set sellado no fue modificado.
Recalcula el hash SHA-256 y lo compara con el registrado.
"""
import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "test_sealed" / "v1_en"
HASH_FILE = PROJECT_ROOT / "seal_test_metadata" / "test_v1_hash.txt"


def compute_file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    if not TEST_DIR.exists():
        print(f"❌ Test set directory no existe: {TEST_DIR}")
        sys.exit(1)

    if not HASH_FILE.exists():
        print(f"❌ Hash file no existe: {HASH_FILE}")
        sys.exit(1)

    # Leer hash registrado (última línea no vacía sin '#')
    with open(HASH_FILE) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    registered_hash = lines[-1]

    # Recalcular
    pair_dirs = sorted([p for p in TEST_DIR.iterdir() if p.is_dir()])
    combined = hashlib.sha256()
    n_files = 0
    for pd in pair_dirs:
        for wav in sorted(pd.iterdir()):
            if wav.suffix == ".wav":
                combined.update(wav.name.encode())
                combined.update(compute_file_hash(wav).encode())
                n_files += 1

    current_hash = combined.hexdigest()

    print(f"Registrado: {registered_hash}")
    print(f"Actual:     {current_hash}")
    print(f"Archivos:   {n_files}")

    if current_hash == registered_hash:
        print("✅ INTACTO — el test set no fue modificado")
        sys.exit(0)
    else:
        print("❌ MODIFICADO — el test set cambió respecto al hash sellado")
        sys.exit(1)


if __name__ == "__main__":
    main()
