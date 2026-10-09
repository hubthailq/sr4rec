"""Freeze the expected results of a demo from a finished run (maintainers, after a demo finishes).

Tolerance per backbone and method: max(0.2 pp, 3 x the standard deviation over seeds).

Usage: python scripts/freeze_expected.py runs/reproduce_d1_earvn d1_earvn
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED = Path(__file__).resolve().parents[1] / "src" / "sr4rec" / "demos" / "expected"


def to_native(obj):
    """Recursively convert numpy scalars to plain Python types so yaml.safe_dump can write them."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("demo")
    a = ap.parse_args()
    m = pd.read_csv(a.run / "metrics.csv")
    m = m[m["size_bin"] == "all"]
    rows = []
    for (b, meth), g in m.groupby(["backbone", "method"], sort=False):
        sd = float(100 * g["rank1"].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append({"backbone": str(b), "method": str(meth), "metric": "rank1",
                     "value_pct": float(round(100 * g["rank1"].mean(), 4)),
                     "tolerance_pp": float(round(max(0.2, 3 * sd), 2))})
    out = EXPECTED / f"{a.demo}.yaml"
    header = f"# Expected results of demo {a.demo}, frozen from {a.run.name}.\n"
    payload = to_native({"demo": a.demo, "frozen": True, "metrics": rows})
    out.write_text(header + yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {out} ({len(rows)} entries)")


if __name__ == "__main__":
    main()
