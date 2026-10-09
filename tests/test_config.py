"""Marker detection, and the generated config and reference staying in sync."""

from pathlib import Path

import pytest
import yaml

from sr4rec.config import Config, parse_config, unconfirmed_lines
from sr4rec.config_docs import iter_fields, render_config, render_reference
from sr4rec.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
BASE = {"dataset": "data/x", "mode": "synthetic"}


def parse(extra: dict):
    return parse_config({**BASE, **extra}, Path("."))


@pytest.mark.parametrize("key,value", [("mode", "native_lr"), ("protocol", "fixed"), ("scale", 8)])
def test_enum_rejected_with_allowed_values(key, value):
    with pytest.raises(ConfigError) as err:
        parse({key: value})
    assert "Allowed values" in str(err.value) and key in str(err.value)


@pytest.mark.parametrize("extra", [
    {"split": {"source": "folders"}},
    {"runtime": {"device": "gpu"}},
    {"comparisons": {"selection": "best"}},
])
def test_nested_enum_rejected(extra):
    with pytest.raises(ConfigError, match="Allowed values"):
        parse(extra)


@pytest.mark.parametrize("extra", [
    {"training": {"epochs": 0}},
    {"training": {"batch_size": 2000}},
    {"training": {"learning_rate": 0}},
    {"recognizer": {"input_size": 100}},
    {"recognizer": {"seeds": []}},
    {"recognizer": {"seeds": [1, 1]}},
    {"latency": {"lr_size": 8}},
    {"comparisons": {"count": 0}},
    {"split": {"val_from_train": 0.9}},
])
def test_numeric_ranges(extra):
    with pytest.raises(ConfigError):
        parse(extra)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        parse({"trainig": {"epochs": 3}})


@pytest.mark.parametrize("entry", [
    {"name": "a", "weights": "w.pth", "images": "x/"},
    {"name": "a"},
    {"name": "a", "images": "x/", "tile": 128},
    {"name": "a", "module": "m.py:Net", "images": "x/"},
    {"name": "bicubic", "weights": "w.pth"},
    {"name": "bad name", "weights": "w.pth"},
    {"name": "a", "module": "m.py"},
])
def test_sr_entries_rejected(entry):
    with pytest.raises(ConfigError):
        parse({"sr": [entry]})


def test_duplicate_sr_names_and_comparison_backbone():
    with pytest.raises(ConfigError, match="unique"):
        parse({"sr": [{"name": "a", "weights": "w"}, {"name": "a", "weights": "v"}]})
    with pytest.raises(ConfigError, match="comparisons.backbone"):
        parse({"comparisons": {"backbone": "vit_base_patch16_224"}})


@pytest.mark.parametrize("ratios", [
    {"train": 0.7, "val": 0.1, "test": 0.3},
    {"train": 1.0, "val": 0.0, "test": 0.0},
    {"train": 0.0, "val": 0.5, "test": 0.5},
])
def test_ratios_rejected(ratios):
    with pytest.raises(ConfigError):
        parse({"split": {"source": "ratios", "ratios": ratios}})


def test_ratios_accepted():
    cfg = parse({"split": {"source": "ratios", "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}}}).config
    assert cfg.split.ratios.train == 0.8


def test_generated_config_documents_every_key_and_parses():
    text = render_config({"dataset": "data/x", "mode": "synthetic"}, {"mode": "[CHECK] suggested"})
    for path, _info in iter_fields():
        leaf = path.split(".")[-1]
        assert f"{leaf}:" in text, path
    for label in ("type:", "required:", "default:", "allowed:"):
        assert label in text
    data = yaml.safe_load(text)
    parse_config(data, Path("."))
    found = unconfirmed_lines(text)
    assert len(found) == 1 and found[0][1].startswith("mode:")


def test_every_field_has_complete_documentation():
    for path, info in iter_fields():
        ex = info.json_schema_extra
        for k in ("doc", "type", "allowed", "required"):
            assert k in ex, (path, k)


def test_reference_docs_are_up_to_date():
    committed = (ROOT / "docs" / "config_reference.md").read_text(encoding="utf-8")
    assert committed == render_reference(), "run: python scripts/make_config_docs.py"
    full = (ROOT / "examples" / "configs" / "full_reference.yaml").read_text(encoding="utf-8")
    assert full == render_config({"dataset": "../data/pets_mini", "mode": "synthetic"}), "run: python scripts/make_config_docs.py"


def test_comment_lines_do_not_count_as_markers():
    assert unconfirmed_lines("# lines marked [CHECK] were inferred\nmode: synthetic\n") == []


def test_defaults():
    cfg = Config.model_validate(BASE)
    assert cfg.protocol == "matched" and cfg.scale == 4 and cfg.recognizer.seeds == [0, 1, 2]
