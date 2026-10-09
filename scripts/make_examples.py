"""Build the small sample datasets of examples/data/ from your own copies of the public datasets.

SR4Rec does not redistribute these images; this script rebuilds exactly the same subsets on every
machine. Each subset folder keeps a `selection.csv` (source file, SHA-256 of the source file,
path, label, split). When `selection.csv` exists, the script uses it and checks every SHA-256;
otherwise it applies the fixed selection rule below and writes `selection.csv` (maintainers commit
that file so that later builds are verified).

Subsets and rules:
  pets_mini  Oxford-IIIT Pet: 10 breeds (5 cats, 5 dogs); per breed the first 20 images of the
             official trainval list (split train) and the first 10 of the official test list
             (split test), in numeric order; images resized (Lanczos) so that the short side is
             256 px and saved as PNG (lossless, so the output does not depend on a JPEG encoder).
  earvn_mini EarVN1.0: the first 20 identity folders (sorted) that have at least 20 images; the
             first 20 images of each (sorted); original size; no split column (SR4Rec splits).
  lfw_mini   LFW: the 10 people with the most images; their first 30 images (sorted); original
             images copied unchanged; fixed split per person: images 1-21 train, 22-24 val,
             25-30 test.

Usage (from the repository root):
    python scripts/make_examples.py --pets ~/datasets/oxford-iiit-pet
    python scripts/make_examples.py --earvn ~/datasets/EarVN1.0 --lfw ~/datasets/lfw
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PET_BREEDS = ["Abyssinian", "Bengal", "Bombay", "British_Shorthair", "Persian",
              "beagle", "boxer", "pug", "samoyed", "shiba_inu"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def write_outputs(out: Path, rows: list[dict], has_split: bool) -> None:
    cols = ["source", "sha256", "path", "label", "split"] + (["output_sha256"] if "output_sha256" in rows[0] else [])
    sel = pd.DataFrame(rows, columns=cols)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "selection.csv").is_file():
        sel.to_csv(out / "selection.csv", index=False)
    cols = ["path", "label", "split"] if has_split else ["path", "label"]
    sel[cols].to_csv(out / "labels.csv", index=False)


def check_outputs(out: Path, rows: list[dict]) -> None:
    """Record the SHA-256 of every written image (column output_sha256) or, when recorded, compare.
    A difference means another Pillow version resampled the images differently: the subset is
    still usable, but results and cache keys will differ from the published ones."""
    changed = 0
    for r in rows:
        digest = sha256(out / "images" / r["path"])
        if r.get("output_sha256"):
            changed += digest != r["output_sha256"]
        else:
            r["output_sha256"] = digest
    if changed:
        print(f"Warning: {changed} resized images differ from the published ones (different Pillow version).")


def verify_or_select(out: Path, candidates: list[dict], source_root: Path) -> list[dict]:
    """Use selection.csv when present (and check hashes); otherwise hash the rule-based candidates."""
    sel_file = out / "selection.csv"
    if sel_file.is_file():
        sel = pd.read_csv(sel_file, dtype=str, keep_default_na=False).to_dict("records")
        bad = []
        for r in sel:
            f = source_root / r["source"]
            if not f.is_file():
                bad.append(f"missing: {r['source']}")
            elif r["sha256"] and sha256(f) != r["sha256"]:
                bad.append(f"different content: {r['source']}")
        if bad:
            sys.exit(f"{len(bad)} source images do not match {sel_file}:\n  " + "\n  ".join(bad[:20]))
        return sel
    for r in candidates:
        r["sha256"] = sha256(source_root / r["source"])
    return candidates


def build_pets(src: Path, out: Path) -> None:
    ann = src / "annotations"

    def read_list(name: str) -> list[str]:
        return [line.split()[0] for line in (ann / name).read_text().splitlines() if line and not line.startswith("#")]

    trainval, test = read_list("trainval.txt"), read_list("test.txt")
    cand = []
    for breed in PET_BREEDS:
        for names, split, k in ((trainval, "train", 20), (test, "test", 10)):
            chosen = sorted((n for n in names if n.rsplit("_", 1)[0] == breed), key=natural_key)[:k]
            if len(chosen) < k:
                sys.exit(f"Breed {breed}: only {len(chosen)} images in the {split} list")
            for n in chosen:
                cand.append({"source": f"images/{n}.jpg", "sha256": "", "path": f"{breed}/{n}.png", "label": breed,
                             "split": split})
    rows = verify_or_select(out, cand, src)
    for r in rows:
        dst = out / "images" / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src / r["source"]) as im:
            im = im.convert("RGB")
            f = 256 / min(im.size)
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            im.save(dst, format="PNG")
    check_outputs(out, rows)
    write_outputs(out, rows, has_split=True)
    print(f"pets_mini: {len(rows)} images written to {out}")


def build_earvn(src: Path, out: Path) -> None:
    base = src / "Images" if (src / "Images").is_dir() else src
    cand = []
    for folder in sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: natural_key(d.name)):
        files = sorted((f for f in folder.iterdir() if f.suffix.lower() in EXTS), key=lambda f: natural_key(f.name))
        if len(files) < 20:
            continue
        label = folder.name.split(".")[0]
        for f in files[:20]:
            cand.append({"source": f.relative_to(src).as_posix(), "sha256": "", "path": f"{label}/{f.name}",
                         "label": label, "split": ""})
        if len(cand) >= 400:
            break
    rows = verify_or_select(out, cand, src)
    for r in rows:
        dst = out / "images" / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / r["source"], dst)
    write_outputs(out, rows, has_split=False)
    print(f"earvn_mini: {len(rows)} images written to {out}")


def build_lfw(src: Path, out: Path) -> None:
    base = src / "lfw" if (src / "lfw").is_dir() else src
    people = [(d, sorted(f for f in d.iterdir() if f.suffix.lower() == ".jpg")) for d in base.iterdir() if d.is_dir()]
    people = sorted(people, key=lambda x: (-len(x[1]), x[0].name))[:10]
    cand = []
    for d, files in people:
        for i, f in enumerate(files[:30], 1):
            split = "train" if i <= 21 else ("val" if i <= 24 else "test")
            cand.append({"source": f.relative_to(src).as_posix(), "sha256": "", "path": f"{d.name}/{f.name}",
                         "label": d.name, "split": split})
    rows = verify_or_select(out, cand, src)
    for r in rows:
        dst = out / "images" / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / r["source"], dst)
    write_outputs(out, rows, has_split=True)
    print(f"lfw_mini: {len(rows)} images written to {out} (face images: keep them private)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pets", type=Path, help="Oxford-IIIT Pet folder (contains images/ and annotations/)")
    ap.add_argument("--earvn", type=Path, help="EarVN1.0 folder (one sub-folder per identity)")
    ap.add_argument("--lfw", type=Path, help="LFW folder (one sub-folder per person)")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "data", help="output folder (default: examples/data)")
    a = ap.parse_args()
    if not (a.pets or a.earvn or a.lfw):
        ap.error("give at least one of --pets, --earvn, --lfw")
    if a.pets:
        build_pets(a.pets, a.out / "pets_mini")
    if a.earvn:
        build_earvn(a.earvn, a.out / "earvn_mini")
    if a.lfw:
        build_lfw(a.lfw, a.out / "lfw_mini")


if __name__ == "__main__":
    main()
