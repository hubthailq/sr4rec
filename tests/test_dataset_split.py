"""Dataset format checks and the train/val/test split."""

import shutil

import pandas as pd
import pytest

from sr4rec.config import SplitConfig
from sr4rec.data.dataset import check_mode_constraints, load_dataset
from sr4rec.data.split import _ratio_counts, make_split, minimum_class_size
from sr4rec.errors import DatasetError
from tests.fixtures.synthetic import make_dataset


def cfg(**kw):
    return SplitConfig.model_validate(kw)


def test_ratio_split_every_class_everywhere(toy):
    ds = load_dataset(toy)
    res = make_split(ds, cfg(source="ratios"))
    t = res.table
    for label in ds.classes:
        assert set(t.loc[t.label == label, "split"]) == {"train", "val", "test"}
    assert t["path"].is_unique and len(t) == len(ds.table)


@pytest.mark.parametrize("ratios", [(0.7, 0.1, 0.2), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2)])
def test_ratio_counts_follow_rounding_rule(tmp_path, ratios):
    ds = load_dataset(make_dataset(tmp_path / "d", n_classes=3, per_class=17))
    tr, va, te = ratios
    res = make_split(ds, cfg(source="ratios", ratios={"train": tr, "val": va, "test": te}))
    n_train, n_val, n_test = _ratio_counts(17, va, te)
    c = res.table[res.table.label == "class_0"]["split"].value_counts()
    assert (c["train"], c["val"], c["test"]) == (n_train, n_val, n_test)
    assert res.counts() == {"train": 3 * n_train, "val": 3 * n_val, "test": 3 * n_test}
    assert f"train {3 * n_train}" in res.description


def test_too_small_class_is_listed(tmp_path):
    root = make_dataset(tmp_path / "d", n_classes=2, per_class=10)
    df = pd.read_csv(root / "labels.csv")
    df = pd.concat([df, pd.DataFrame([{"path": "c0/img_00.png", "label": "tiny"}])]).drop_duplicates("path", keep="last")
    df.loc[df.path == "c0/img_01.png", "label"] = "tiny"
    df.to_csv(root / "labels.csv", index=False)
    with pytest.raises(DatasetError, match="'tiny' has 2 image"):
        make_split(load_dataset(root), cfg(source="ratios"))
    assert minimum_class_size(0.1, 0.2) == 3


def test_same_seed_same_split_and_seed_changes_split(toy):
    ds = load_dataset(toy)
    a = make_split(ds, cfg(source="ratios", seed=0)).table
    b = make_split(ds, cfg(source="ratios", seed=0)).table
    c = make_split(ds, cfg(source="ratios", seed=1)).table
    assert a.equals(b) and not a["split"].equals(c["split"])


def test_split_column_used_and_val_from_train(tmp_path):
    root = make_dataset(tmp_path / "d", n_classes=3, per_class=12, with_split=True)
    ds = load_dataset(root)
    res = make_split(ds, cfg(source="column"))
    assert res.table["split"].tolist() == ds.table["split"].tolist()
    df = pd.read_csv(root / "labels.csv")
    df.loc[df.split == "val", "split"] = "train"
    df.to_csv(root / "labels.csv", index=False)
    ds = load_dataset(root)
    res = make_split(ds, cfg(source="auto", val_from_train=0.1))
    per_class_train = 9  # 12 images: 3 test, 9 train after removing val
    moved = max(1, int(per_class_train * 0.1 + 0.5))
    assert res.counts()["val"] == 3 * moved
    assert res.moved_to_val == 3 * moved


def test_column_source_without_column_fails(toy):
    with pytest.raises(DatasetError, match="no split column"):
        make_split(load_dataset(toy), cfg(source="column"))


def test_ratios_source_ignores_column(tmp_path):
    ds = load_dataset(make_dataset(tmp_path / "d", with_split=True))
    res = make_split(ds, cfg(source="ratios"))
    assert res.source == "ratios"


def test_class_missing_from_a_split(tmp_path):
    root = make_dataset(tmp_path / "d", n_classes=2, per_class=8, with_split=True)
    df = pd.read_csv(root / "labels.csv")
    df.loc[(df.label == "class_1") & (df.split == "test"), "split"] = "train"
    df.to_csv(root / "labels.csv", index=False)
    with pytest.raises(DatasetError, match="class_1' has no image in: test"):
        make_split(load_dataset(root), cfg(source="column"))


def test_duplicate_content_across_splits_is_detected(tmp_path):
    root = make_dataset(tmp_path / "d", n_classes=2, per_class=8, with_split=True)
    df = pd.read_csv(root / "labels.csv")
    train_path = df[(df.split == "train")].path.iloc[0]
    shutil.copy(root / "images" / train_path, root / "images" / "copy.png")
    df = pd.concat([df, pd.DataFrame([{"path": "copy.png", "label": df[df.path == train_path].label.iloc[0],
                                       "split": "test"}])])
    df.to_csv(root / "labels.csv", index=False)
    with pytest.raises(DatasetError, match="more than one split"):
        make_split(load_dataset(root), cfg(source="column"))


@pytest.mark.parametrize("edit,message", [
    (lambda df: df.drop(columns=["label"]), "missing: label"),
    (lambda df: df.assign(extra=1), "unknown columns"),
    (lambda df: pd.concat([df, df.iloc[:1]]), "listed more than once"),
    (lambda df: df.assign(path=df.path.where(df.index != 0, "c0/missing.png")), "file not found"),
    (lambda df: df.assign(path=df.path.where(df.index != 0, "c0/a.gif")), "unsupported extension"),
    (lambda df: df.assign(split="training"), "not allowed"),
    (lambda df: df.assign(label=df.label.where(df.index != 0, "")), "empty label"),
])
def test_format_errors_report_line_numbers(toy, edit, message):
    df = pd.read_csv(toy / "labels.csv")
    edit(df).to_csv(toy / "labels.csv", index=False)
    with pytest.raises(DatasetError, match=message):
        load_dataset(toy)


def test_unreadable_image(toy):
    (toy / "images" / "c0" / "img_00.png").write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="line 2: .*cannot be read"):
        load_dataset(toy)


def test_grayscale_conversion_is_noted(toy):
    from PIL import Image

    f = toy / "images" / "c0" / "img_00.png"
    Image.open(f).convert("L").save(f)
    assert "converted to 3-channel RGB" in load_dataset(toy).notes[0]


def test_synthetic_mode_rejects_tiny_images(tmp_path):
    ds = load_dataset(make_dataset(tmp_path / "d", size=(3, 10)))
    with pytest.raises(DatasetError, match="smaller than scale 4"):
        check_mode_constraints(ds, "synthetic", 4)
    check_mode_constraints(ds, "native-lr", 4)


def test_split_is_reused_from_cache_and_change_is_warned(toy, tmp_path):
    ds = load_dataset(toy)
    cache = tmp_path / "cache"
    a = make_split(ds, cfg(source="ratios"), cache)
    b = make_split(ds, cfg(source="ratios"), cache)
    assert a.table.equals(b.table) and not b.warnings
    c = make_split(ds, cfg(source="ratios", seed=5), cache)
    assert any("changed" in w for w in c.warnings)


def test_identical_images_are_kept_together_by_the_split(tmp_path):
    root = make_dataset(tmp_path / "d", n_classes=2, per_class=11)
    shutil.copy(root / "images/c0/img_01.png", root / "images/c0/img_02.png")
    ds = load_dataset(root)
    for seed in range(6):
        res = make_split(ds, cfg(source="ratios", seed=seed))  # never stops on its own duplicates
        t = res.table.set_index("path")
        assert t.loc["c0/img_01.png", "split"] == t.loc["c0/img_02.png", "split"]


def test_minimum_class_size_is_safe_for_every_larger_class():
    from sr4rec.data.split import _ratio_counts

    for val, test in [(0.1, 0.2), (0.45, 0.45), (0.3, 0.3)]:
        need = minimum_class_size(val, test)
        assert all(_ratio_counts(n, val, test)[0] >= 1 for n in range(need, 300))


def test_sixteen_bit_images_are_scaled(tmp_path):
    import numpy as np
    from PIL import Image

    from sr4rec.utils import load_rgb

    arr = (np.linspace(0, 65535, 64 * 64).reshape(64, 64)).astype(np.uint16)
    f = tmp_path / "x.png"
    Image.fromarray(arr).save(f)
    rgb = np.asarray(load_rgb(f))
    assert rgb.min() == 0 and rgb.max() == 255 and 100 < rgb.mean() < 155
