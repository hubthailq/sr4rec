"""End-to-end runs on a tiny generated dataset (CPU, random-init backbones).

Covers the fingerprint, --dry-run, protocol wiring, the oracle-SR sanity check (SR equals the HR
upper bound), a repeated run being cached and identical, and the output folder of section 2.1.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from sr4rec.config import parse_config
from sr4rec.errors import SRModelError
from sr4rec.pipeline import Run
from sr4rec.utils import mod_crop, png_name
from tests.fixtures.synthetic import make_dataset

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def base_cfg(**over):
    cfg = {
        "dataset": "data/toy", "mode": "synthetic",
        "recognizer": {"backbones": ["resnet18"], "pretrained": False, "input_size": 64, "seeds": [0, 1]},
        "training": {"epochs": 1, "batch_size": 16, "num_workers": 0},
        "latency": {"lr_size": 16, "threads": 1},
        "comparisons": {"count": 3},
        "runtime": {"device": "cpu"},
    }
    cfg.update(over)
    return cfg


def make_oracle(project: Path) -> None:
    """SR images that are exactly the (mod-cropped) HR images."""
    df = pd.read_csv(project / "data/toy/labels.csv")
    for p in df.path:
        im = mod_crop(Image.open(project / "data/toy/images" / p).convert("RGB"), 4)
        out = project / "sr_images/oracle" / png_name(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)


@pytest.fixture
def project(tmp_path):
    make_dataset(tmp_path / "data" / "toy", n_classes=3, per_class=10, size=(48, 40))
    shutil.copy(EXAMPLES / "sr_models" / "tiny_espcn.py", tmp_path / "tiny_espcn.py")
    make_oracle(tmp_path)
    return tmp_path


def logs():
    lines = []
    return lines, lines.append


@pytest.mark.slow
def test_matched_run_outputs_cache_and_oracle(project):
    cfg = base_cfg(sr=[{"name": "tiny", "module": "tiny_espcn.py:TinyESPCN"},
                       {"name": "oracle", "images": "sr_images/oracle"}])
    lines, log = logs()
    res = Run(parse_config(cfg, project), log=log).execute()
    d = res.run_dir
    for f in ["report.md", "plots/accuracy_by_size.png", "plots/cmc.png", "plots/quality_vs_accuracy.png", "cmc.csv",
              "metrics.csv", "stats.csv", "predictions.csv", "latency.csv", "table.tex", "split.csv",
              "comparisons/index.csv", "sr4rec.yaml", "fingerprint.yaml", "lr/test"]:
        assert (d / f).exists(), f
    assert (d / "table.tex").read_text(encoding="utf-8").isascii(), "table.tex must compile with plain pdflatex"
    fp = yaml.safe_load((d / "fingerprint.yaml").read_text(encoding="utf-8"))
    for k in ("config", "seeds", "environment", "dataset", "sr_models", "sr4rec_version"):
        assert k in fp
    assert fp["dataset"]["labels_csv_sha256"] and fp["dataset"]["split_csv_sha256"]
    assert {"python", "torch", "cpu"} <= set(fp["environment"]) | set(fp["environment"]["packages"])

    pred = pd.read_csv(d / "predictions.csv")
    assert set(pred.method) == {"bicubic", "tiny", "oracle", "hr"}
    a = pred[pred.method == "oracle"].sort_values(["seed", "path"])[["seed", "path", "pred"]].reset_index(drop=True)
    b = pred[pred.method == "hr"].sort_values(["seed", "path"])[["seed", "path", "pred"]].reset_index(drop=True)
    assert a.equals(b), "oracle SR must reproduce the HR upper bound exactly"
    met = pd.read_csv(d / "metrics.csv")
    assert met[(met.method == "oracle") & (met.size_bin == "all")].psnr.iloc[0] == float("inf")

    # the re-run is fully cached and gives identical numbers
    lines2, log2 = logs()
    res2 = Run(parse_config(cfg, project), log=log2).execute()
    assert not any("Training" in x for x in lines2) and not any(x.startswith("  SR tiny:") for x in lines2)
    assert pd.read_csv(res2.run_dir / "metrics.csv").equals(met)
    assert res2.run_dir != d

    # the copied configuration re-runs from the run folder
    from sr4rec.config import load_config

    loaded = load_config(d / "sr4rec.yaml")
    assert loaded.resolve(loaded.config.dataset) == (project / "data" / "toy").resolve()


@pytest.mark.slow
def test_native_fixed_recognizer_run(tmp_path):
    make_dataset(tmp_path / "data" / "toy", n_classes=3, per_class=10, sizes=[(20, 16), (40, 36), (70, 66)])
    cfg = base_cfg(mode="native-lr", protocol="fixed_recognizer", comparisons={"enabled": False},
                   latency={"enabled": False})
    lines, log = logs()
    res = Run(parse_config(cfg, tmp_path), log=log).execute()
    assert sum("Training" in x for x in lines) == 2  # one recognizer per backbone and seed
    text = (res.run_dir / "report.md").read_text(encoding="utf-8")
    assert "n/a (no HR reference in native-lr mode)" in text and "HR (upper bound) |" not in text
    assert "protocol fixed_recognizer" in text and not (res.run_dir / "lr").exists()


def test_protocol_wiring_and_dry_run(project):
    cfg = base_cfg(sr=[{"name": "tiny", "module": "tiny_espcn.py:TinyESPCN"}])
    run = Run(parse_config(cfg, project), log=None)
    run.prepare()
    run.load_sources()
    t = run.trainings()
    assert sorted({x.source for x in t}) == ["bicubic", "hr", "tiny"] and len(t) == 6
    item = run.image_item("tiny", run.paths["train"][0])
    assert "tiny_" in str(item.file.parent.parent) or "tiny_" in str(item.file)
    assert run.image_item("hr", run.paths["train"][0]).crop_scale == 4

    fixed = Run(parse_config({**cfg, "protocol": "fixed_recognizer"}, project), log=None)
    fixed.prepare()
    fixed.load_sources()
    assert {x.source for x in fixed.trainings()} == {"hr"} and len(fixed.trainings()) == 2
    assert fixed.needed_splits("tiny") == ("test",)

    text = Run(parse_config(cfg, project), dry_run=True, log=None).execute().dry_run_text
    assert "= 6 recognizer trainings" in text and "Dry run passed" in text and "tiny              OK  x4" in text


def test_dry_run_reports_every_bad_sr_model(project):
    (project / "sr_images/partial").mkdir(parents=True)
    cfg = base_cfg(sr=[{"name": "tiny", "module": "tiny_espcn.py:Missing"},
                       {"name": "partial", "images": "sr_images/partial"}])
    with pytest.raises(SRModelError) as err:
        Run(parse_config(cfg, project), dry_run=True, log=None).execute()
    text = str(err.value)
    assert "tiny              FAILED" in text and "partial           FAILED" in text


def test_matched_folder_source_needs_train_images(project):
    df = pd.read_csv(project / "data/toy/labels.csv")
    from sr4rec.config import SplitConfig
    from sr4rec.data.dataset import load_dataset
    from sr4rec.data.split import make_split

    split = make_split(load_dataset(project / "data/toy"), SplitConfig()).table
    for p in split[split.split == "train"].path[:2]:
        (project / "sr_images/oracle" / png_name(p)).unlink()
    cfg = base_cfg(sr=[{"name": "oracle", "images": "sr_images/oracle"}])
    with pytest.raises(SRModelError, match="missing"):
        Run(parse_config(cfg, project), dry_run=True, log=None).execute()
    fixed = base_cfg(protocol="fixed_recognizer", sr=[{"name": "oracle", "images": "sr_images/oracle"}])
    Run(parse_config(fixed, project), log=None).prepare()
    assert len(df) == 30
