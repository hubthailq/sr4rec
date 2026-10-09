"""CLI behaviour: init, [CHECK] refusal, exit codes, reproduce logic."""

import subprocess
import sys

import pandas as pd
import pytest
import yaml

from sr4rec.cli import main
from sr4rec.config import load_config
from sr4rec.reproduce import compare, list_demos, load_demo


def test_init_writes_config_and_refuses_overwrite(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "data/toy"]) == 0
    text = (tmp_path / "sr4rec.yaml").read_text(encoding="utf-8")
    assert "dataset: data/toy" in text and "[CHECK]" in text and "mode: native-lr" in text
    assert main(["init", "data/toy"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["init", "data/toy", "--force"]) == 0


def test_run_refuses_unconfirmed_config(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "data/toy"])
    assert main(["run", "sr4rec.yaml", "--dry-run"]) == 1
    assert "[CHECK] markers" in capsys.readouterr().err


def test_paths_are_relative_to_the_config_file(toy, tmp_path, monkeypatch):
    project = tmp_path
    (project / "cfg").mkdir()
    (project / "cfg" / "c.yaml").write_text("dataset: ../data/toy\nmode: synthetic\n")
    monkeypatch.chdir("/")
    loaded = load_config(project / "cfg" / "c.yaml")
    assert loaded.resolve(loaded.config.dataset) == (project / "data" / "toy").resolve()


def test_bad_config_exit_code(tmp_path, capsys):
    f = tmp_path / "c.yaml"
    f.write_text("dataset: x\nmode: native_lr\n")
    assert main(["run", str(f)]) == 1
    assert "'native_lr' is not allowed. Allowed values: synthetic, native-lr" in capsys.readouterr().err


def test_reproduce_unknown_demo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["reproduce", "nope"]) == 1
    assert "Unknown demo" in capsys.readouterr().err


def test_reproduce_missing_local_inputs(tmp_path, monkeypatch, capsys):
    # quickstart is frozen (expected results ship with the package); without a local
    # examples/data/pets_mini and the example SR weights it still cannot run.
    monkeypatch.chdir(tmp_path)
    assert main(["reproduce", "quickstart"]) == 3
    err = capsys.readouterr().err.replace("\\", "/")
    assert "Demo inputs are missing" in err and "examples/data/pets_mini" in err


def test_all_demo_configs_are_valid(tmp_path):
    from sr4rec.config import parse_config
    from sr4rec.reproduce import demo_config

    demos = list_demos()
    assert "quickstart" in demos and "d1_earvn" in demos and "d2_lfw" in demos
    for name in demos:
        demo, expected = load_demo(name)
        parse_config(demo_config(demo, None, None, "runs", "auto"), tmp_path)
        assert expected is not None and "frozen" in expected


def test_compare_detects_mismatch():
    metrics = pd.DataFrame([{"backbone": "resnet18", "method": "bicubic", "seed": s, "size_bin": "all",
                             "rank1": v} for s, v in enumerate([0.80, 0.82])])
    exp = {"metrics": [{"backbone": "resnet18", "method": "bicubic", "value_pct": 81.0, "tolerance_pp": 0.5}]}
    assert compare(metrics, exp, eval_only=False)[0]
    exp["metrics"][0]["value_pct"] = 83.0
    assert not compare(metrics, exp, eval_only=False)[0]


def test_module_entry_point():
    out = subprocess.run([sys.executable, "-m", "sr4rec.cli", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "SR4Rec" in out.stdout


@pytest.mark.parametrize("cmd", [["init", "--help"], ["run", "--help"], ["reproduce", "--help"]])
def test_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main(cmd)
    assert e.value.code == 0


def test_usage_errors_exit_1(capsys):
    with pytest.raises(SystemExit) as e:
        main(["reproduce", "quickstart", "--device", "gpu"])
    assert e.value.code == 1


def test_generated_yaml_quotes_awkward_dataset_names(tmp_path, monkeypatch):
    from tests.fixtures.synthetic import make_dataset

    make_dataset(tmp_path / "data" / "2024")
    monkeypatch.chdir(tmp_path)
    assert main(["init", "data/2024"]) == 0
    loaded = load_config(tmp_path / "sr4rec.yaml")
    assert loaded.config.dataset == "data/2024"
    from sr4rec.config_docs import format_value

    for s in ["my data #1", "2024", "yes", "a: b"]:
        assert yaml.safe_load(f"k: {format_value(s)}")["k"] == s
