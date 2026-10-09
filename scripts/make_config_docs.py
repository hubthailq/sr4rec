"""Regenerate docs/config_reference.md and examples/configs/full_reference.yaml from the schema.

Usage: python scripts/make_config_docs.py   (CI fails if the committed files are out of date)
"""

from pathlib import Path

from sr4rec.config_docs import render_config, render_reference

ROOT = Path(__file__).resolve().parents[1]

(ROOT / "docs" / "config_reference.md").write_text(render_reference(), encoding="utf-8")
(ROOT / "examples" / "configs" / "full_reference.yaml").write_text(
    render_config({"dataset": "../data/pets_mini", "mode": "synthetic"}), encoding="utf-8")
print("Wrote docs/config_reference.md and examples/configs/full_reference.yaml")
