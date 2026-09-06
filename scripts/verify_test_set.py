"""
scripts/verify_test_set.py

Verifica que los test sets sellados no fueron modificados: recalcula el hash
SHA-256 acumulativo de cada uno y lo compara contra el registrado al sellar.

Es el mecanismo que hace cumplir la restricción dura del proyecto ("el test set
sellado NO se regenera"). Un set que no esté en SEALED_SETS queda sin proteger,
así que al sellar uno nuevo hay que agregarlo acá.

Uso:
    python -m scripts.verify_test_set              # verifica todos
    python -m scripts.verify_test_set --set v2_es  # uno solo

Sale con código 0 solo si todos los sets verificados están intactos.
"""
import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Registro de test sets sellados. Agregar acá cualquier set nuevo al sellarlo.
SEALED_SETS = {
    "v1_en": {
        "dir": PROJECT_ROOT / "data" / "test_sealed" / "v1_en",
        "hash_file": PROJECT_ROOT / "seal_test_metadata" / "test_v1_hash.txt",
        "description": "Inglés — LibriSpeech + MUSAN/ESC-50, canal audiolibro",
    },
    "v2_es": {
        "dir": PROJECT_ROOT / "data" / "test_sealed" / "v2_es",
        "hash_file": PROJECT_ROOT / "seal_test_metadata" / "test_v2_hash.txt",
        "description": "Español — Common Voice ES v26, canal crowdsourced",
    },
    "v3_mls_es": {
        "dir": PROJECT_ROOT / "data" / "test_sealed" / "v3_mls_es",
        "hash_file": PROJECT_ROOT / "seal_test_metadata" / "test_v3_mls_es_hash.txt",
        "description": "Español — MLS, canal audiolibro (control idioma/canal)",
    },
}


def compute_file_hash(path: Path) -> str:
    """SHA-256 of a single file, read in chunks."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dataset_hash(test_dir: Path) -> tuple[str, int]:
    """Cumulative SHA-256 over every .wav in the set, in sorted order.

    Must stay identical to the routine used by the sealing scripts: it hashes
    the file name followed by the file content, pair directory by pair
    directory, both sorted.
    """
    combined = hashlib.sha256()
    n_files = 0
    for pair_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for wav in sorted(f for f in pair_dir.iterdir() if f.suffix == ".wav"):
            combined.update(wav.name.encode())
            combined.update(compute_file_hash(wav).encode())
            n_files += 1
    return combined.hexdigest(), n_files


def read_registered_hash(hash_file: Path) -> str:
    """Read the recorded hash, tolerating both file formats in the repo.

    v1/v2 write a comment header plus a bare hash on the last line; earlier
    revisions of the v3 sealer wrote `<hash>  <n> archivos` on one line. Taking
    the first whitespace-separated token of the last non-comment line handles both.
    """
    lines = [
        ln.strip() for ln in hash_file.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{hash_file} no contiene ningún hash")
    return lines[-1].split()[0]


def verify_one(name: str, cfg: dict) -> bool:
    """Verify a single sealed set. Returns True if intact."""
    print(f"── {name} — {cfg['description']}")

    if not cfg["dir"].exists():
        print(f"   SIN VERIFICAR: no existe {cfg['dir'].relative_to(PROJECT_ROOT)}")
        print("   (el set no está generado en esta máquina)")
        return True  # ausente no es lo mismo que corrupto

    if not cfg["hash_file"].exists():
        print(f"   ERROR: falta el hash registrado ({cfg['hash_file'].name})")
        return False

    registered = read_registered_hash(cfg["hash_file"])
    current, n_files = compute_dataset_hash(cfg["dir"])

    print(f"   registrado: {registered}")
    print(f"   actual:     {current}")
    print(f"   archivos:   {n_files}")

    if current == registered:
        print("   INTACTO")
        return True
    print("   MODIFICADO — el set cambió respecto del hash sellado")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", dest="set_name", choices=sorted(SEALED_SETS),
        help="Verificar solo este set (por defecto: todos)",
    )
    args = parser.parse_args()

    targets = {args.set_name: SEALED_SETS[args.set_name]} if args.set_name else SEALED_SETS

    print("=" * 70)
    print("VERIFICACIÓN DE INTEGRIDAD DE TEST SETS SELLADOS")
    print("=" * 70)

    results = {name: verify_one(name, cfg) for name, cfg in targets.items()}

    print("=" * 70)
    ok = sum(results.values())
    print(f"Sets verificados: {len(results)}   intactos: {ok}   con problema: {len(results) - ok}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
