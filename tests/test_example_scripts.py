"""Sample-data builders, dataset conversion scripts and make_sr_images.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from sr4rec.data.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    out = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=cwd)
    assert out.returncode == 0, out.stderr + out.stdout
    return out.stdout


def img(path: Path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(path)


def test_from_folder_per_label(tmp_path):
    for lab in ("a", "b"):
        for i in range(3):
            img(tmp_path / "images" / lab / f"{i}.png")
    run(ROOT / "examples/dataset_conversion/from_folder_per_label.py", tmp_path)
    ds = load_dataset(tmp_path)
    assert len(ds.table) == 6 and ds.classes == ["a", "b"]


def test_from_filename_pattern(tmp_path):
    for p in ("017_s1_03.png", "017_s2_01.png", "018_s1_01.png"):
        img(tmp_path / "images" / p)
    run(ROOT / "examples/dataset_conversion/from_filename_pattern.py", tmp_path)
    assert load_dataset(tmp_path).classes == ["017", "018"]


def test_from_annotation_csv(tmp_path):
    rows = []
    for i, sub in enumerate(["training", "validation", "testing"] * 2):
        img(tmp_path / "images" / f"{i}.png")
        rows.append({"file": f"{i}.png", "identity": f"id{i % 2}", "subset": sub})
    pd.DataFrame(rows).to_csv(tmp_path / "ann.csv", index=False)
    run(ROOT / "examples/dataset_conversion/from_annotation_csv.py", tmp_path, tmp_path / "ann.csv", "--path-col", "file",
        "--label-col", "identity", "--split-col", "subset", "--split-map", "training=train", "validation=val",
        "testing=test")
    ds = load_dataset(tmp_path)
    assert set(ds.table["split"]) == {"train", "val", "test"}


def _fake_pets(root: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_examples import PET_BREEDS

    lines_tv, lines_te = ["#Image CLASS-ID SPECIES BREED ID"], []
    for b in PET_BREEDS:
        for i in range(1, 36):
            img(root / "images" / f"{b}_{i}.jpg", (300, 260))
            (lines_tv if i <= 22 else lines_te).append(f"{b}_{i} 1 1 1")
    (root / "annotations").mkdir(parents=True)
    (root / "annotations/trainval.txt").write_text("\n".join(lines_tv))
    (root / "annotations/test.txt").write_text("\n".join(lines_te))


def test_make_examples_builders(tmp_path):
    _fake_pets(tmp_path / "pets")
    for p in range(12):
        for i in range(1, 33 if p < 10 else 5):
            img(tmp_path / "lfw" / f"Person_{p:02d}" / f"Person_{p:02d}_{i:04d}.jpg", (250, 250))
    for p in range(1, 24):
        for i in range(22 if p != 3 else 5):
            img(tmp_path / "earvn" / "Images" / f"{p:03d}.NAME" / f"{i:03d}.jpg", (20 + i, 30))
    out = tmp_path / "out"
    run(ROOT / "scripts/make_examples.py", "--pets", tmp_path / "pets", "--lfw", tmp_path / "lfw", "--earvn",
        tmp_path / "earvn", "--out", out)
    pets = load_dataset(out / "pets_mini")
    assert len(pets.table) == 300 and pets.short_sides().min() == 256
    assert (pets.table["split"] == "test").sum() == 100
    lfw = load_dataset(out / "lfw_mini")
    assert len(lfw.table) == 300 and len(lfw.classes) == 10
    ear = load_dataset(out / "earvn_mini")
    assert len(ear.table) == 400 and "003" not in ear.classes
    sel = pd.read_csv(out / "pets_mini" / "selection.csv")
    assert sel["sha256"].str.len().eq(64).all()
    run(ROOT / "scripts/make_examples.py", "--pets", tmp_path / "pets", "--out", out)  # verified rebuild
