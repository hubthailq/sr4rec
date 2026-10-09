"""``sr4rec reproduce <demo>``: run a published demo and compare with its expected results.

Demo definitions live in ``sr4rec/demos/<demo>.yaml`` and expected results in
``sr4rec/demos/expected/<demo>.yaml`` (both shipped with the package). A demo
file has the keys ``demo``, ``description``, ``defaults`` (``data_root``,
``weights_root``) and ``config`` (a normal SR4Rec configuration in which the
strings ``{data_root}`` and ``{weights_root}`` are replaced).
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml

from .config import parse_config
from .errors import MissingResource, ReproductionMismatch, SR4RecError

EVAL_ONLY_TOLERANCE_PP = 0.01  # 4 decimal places on the [0, 1] scale


def _demo_dir():
    return resources.files("sr4rec") / "demos"


def list_demos() -> dict[str, str]:
    out = {}
    for f in sorted(_demo_dir().iterdir(), key=lambda x: x.name):
        if f.name.endswith(".yaml"):
            d = yaml.safe_load(f.read_text(encoding="utf-8"))
            out[d["demo"]] = d.get("description", "")
    return out


def load_demo(name: str) -> tuple[dict, dict | None]:
    f = _demo_dir() / f"{name}.yaml"
    if not f.is_file():
        demos = ", ".join(list_demos())
        raise SR4RecError(f"Unknown demo {name!r}. Available demos: {demos}")
    demo = yaml.safe_load(f.read_text(encoding="utf-8"))
    e = _demo_dir() / "expected" / f"{name}.yaml"
    expected = yaml.safe_load(e.read_text(encoding="utf-8")) if e.is_file() else None
    return demo, expected


def _substitute(obj, mapping: dict[str, str]):
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("{" + k + "}", v)
        return obj
    if isinstance(obj, list):
        return [_substitute(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, mapping) for k, v in obj.items()}
    return obj


def demo_config(demo: dict, data_root: str | None, weights_root: str | None, output_dir: str, device: str) -> dict:
    d = demo.get("defaults", {})
    mapping = {"data_root": Path(data_root or d.get("data_root", "data")).as_posix(),
               "weights_root": Path(weights_root or d.get("weights_root", "weights")).as_posix()}
    cfg = _substitute(demo["config"], mapping)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["output_dir"] = output_dir
    cfg["runtime"]["device"] = device
    return cfg


def _absolutize(cfg: dict, base: Path) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    cfg["dataset"] = str((base / cfg["dataset"]).resolve())
    for e in cfg.get("sr", []):
        for k in ("weights", "images"):
            if e.get(k):
                e[k] = str((base / e[k]).resolve())
        if e.get("module"):
            f, c = e["module"].rsplit(":", 1)
            e["module"] = f"{(base / f).resolve()}:{c}"
    cfg["runtime"]["output_dir"] = str((base / cfg["runtime"]["output_dir"]).resolve())
    return cfg


def _missing_inputs(cfg: dict, base: Path) -> list[str]:
    missing = []
    ds = base / cfg["dataset"]
    if not (ds / "labels.csv").is_file():
        missing.append(f"dataset {ds} (prepare it as described in the demo notes)")
    for e in cfg.get("sr", []):
        for k in ("weights", "images"):
            if e.get(k) and not (base / e[k]).exists():
                missing.append(f"{k} of SR model {e['name']!r}: {base / e[k]}")
        if e.get("module"):
            f = e["module"].rsplit(":", 1)[0]
            if not (base / f).is_file():
                missing.append(f"module of SR model {e['name']!r}: {base / f}")
    return missing


def compare(metrics: pd.DataFrame, expected: dict, eval_only: bool) -> tuple[bool, list[str]]:
    overall = metrics[metrics["size_bin"] == "all"]
    lines = [f"{'backbone':<24}{'method':<22}{'expected':>10}{'got':>10}{'tolerance':>11}  result"]
    ok = True
    for row in expected["metrics"]:
        sub = overall[(overall["backbone"] == row["backbone"]) & (overall["method"] == row["method"])]
        got = 100 * float(sub[row.get("metric", "rank1")].mean()) if len(sub) else float("nan")
        tol = EVAL_ONLY_TOLERANCE_PP if eval_only else float(row["tolerance_pp"])
        exp = float(row["value_pct"])
        good = abs(got - exp) <= tol + 1e-9
        ok &= good
        lines.append(f"{row['backbone']:<24}{row['method']:<22}{exp:>10.2f}{got:>10.2f}{tol:>11.2f}  "
                     f"{'OK' if good else 'MISMATCH'}")
    return ok, lines


def reproduce(name: str, eval_only: bool = False, data_root: str | None = None, weights_root: str | None = None,
              checkpoints_root: str = "checkpoints", output_dir: str = "runs", device: str = "auto") -> int:
    from .pipeline import Run

    if name == "list":
        for k, v in list_demos().items():
            print(f"{k:<18} {v}")
        return 0
    demo, expected = load_demo(name)
    cfg = demo_config(demo, data_root, weights_root, output_dir, device)
    base = Path.cwd()
    if not expected or not expected.get("frozen"):
        dump = base / output_dir / f"{name}_config.yaml"
        dump.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Configuration of the SR4Rec demo {name!r}. Run it with: sr4rec run {dump.name}\n"
        dump.write_text(header + yaml.safe_dump(_absolutize(cfg, base), sort_keys=False), encoding="utf-8")
        raise MissingResource(
            f"The expected results of demo {name!r} have not been frozen in this version of SR4Rec, so there is "
            f"nothing to compare with. The demo configuration was written to {dump}; run it with "
            f"`sr4rec run {dump}`."
        )
    missing = _missing_inputs(cfg, base)
    if missing:
        notes = demo.get("notes", "")
        raise MissingResource("Demo inputs are missing:\n  - " + "\n  - ".join(missing) + (f"\n{notes}" if notes else ""))
    ckpt_dir = None
    if eval_only:
        ckpt_dir = base / checkpoints_root / name
        if not ckpt_dir.is_dir():
            raise MissingResource(f"Checkpoints for demo {name!r} not found in {ckpt_dir}. Download them with: "
                                  f"python scripts/fetch_checkpoints.py {name}")
    cfg["runtime"]["run_name"] = f"reproduce_{name}{'_eval_only' if eval_only else ''}"
    # A dedicated cache, emptied before a full re-training, so that level 2 really trains.
    cache = base / output_dir / ".cache_reproduce" / name
    if not eval_only and cache.exists():
        shutil.rmtree(cache / "checkpoints", ignore_errors=True)
    cfg["runtime"]["cache_dir"] = str(cache)
    loaded = parse_config(cfg, base)
    result = Run(loaded, checkpoints_dir=ckpt_dir, command=f"sr4rec reproduce {name}").execute()
    metrics = pd.read_csv(result.run_dir / "metrics.csv")
    ok, lines = compare(metrics, expected, eval_only)
    print("\n".join(lines))
    if not ok:
        raise ReproductionMismatch(f"Demo {name!r}: some metrics are outside the tolerance (see the table above).")
    print(f"Demo {name!r} reproduced within tolerance.")
    return 0
