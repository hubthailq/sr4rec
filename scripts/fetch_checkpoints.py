"""Download the published recognizer checkpoints of a demo into checkpoints/<demo>/ and check SHA-256.

Usage: python scripts/fetch_checkpoints.py d1_earvn [--out checkpoints]
Then:  sr4rec reproduce d1_earvn --eval-only
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "checkpoints_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo")
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    a = ap.parse_args()
    demos = json.loads(MANIFEST.read_text())["demos"]
    if a.demo not in demos:
        sys.exit(f"No published checkpoints for demo {a.demo!r} in this version (available: {sorted(demos) or 'none'}).")
    dest = a.out / a.demo
    dest.mkdir(parents=True, exist_ok=True)
    for entry in demos[a.demo]:
        f = dest / entry["name"]
        if not (f.is_file() and sha256(f) == entry["sha256"]):
            print(f"Downloading {entry['name']}")
            urllib.request.urlretrieve(entry["url"], f)
        if sha256(f) != entry["sha256"]:
            sys.exit(f"SHA-256 mismatch for {f}; delete it and try again.")
    print(f"{len(demos[a.demo])} checkpoints ready in {dest}")


if __name__ == "__main__":
    main()
