"""Copy the recognizer checkpoints of a finished run into a folder ready for publication (Zenodo)
and print the manifest entries for scripts/checkpoints_manifest.json.

Usage: python scripts/export_checkpoints.py runs/reproduce_d1_earvn d1_earvn --out release/checkpoints
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("demo")
    ap.add_argument("--out", type=Path, default=Path("release/checkpoints"))
    ap.add_argument("--url-prefix", default="https://zenodo.org/records/<record>/files/")
    a = ap.parse_args()
    table = pd.read_csv(a.run / "checkpoints.csv")
    dest = a.out / a.demo
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in table.itertuples():
        name = f"{r.backbone}__{r.source}__seed{r.seed}.pt"
        shutil.copyfile(r.file, dest / name)
        digest = hashlib.sha256((dest / name).read_bytes()).hexdigest()
        entries.append({"name": name, "url": f"{a.url_prefix}{a.demo}__{name}", "sha256": digest})
    print(json.dumps({a.demo: entries}, indent=2))


if __name__ == "__main__":
    main()
