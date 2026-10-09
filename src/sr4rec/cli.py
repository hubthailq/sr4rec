"""Command-line interface: exactly three commands, ``init``, ``run`` and ``reproduce``.

Exit codes: 0 success, 1 error, 2 reproduced metrics outside tolerance, 3 missing
data, weights, checkpoints or expected results.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .errors import SR4RecError

EPILOG = "Documentation: README.md and docs/ in the SR4Rec repository."


class _Parser(argparse.ArgumentParser):
    """Usage errors exit with code 1 (exit code 2 is reserved for reproduction mismatches)."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _init(args: argparse.Namespace) -> int:
    from .config_docs import render_config
    from .data.dataset import load_dataset

    out = Path(args.output).expanduser().resolve()
    if out.exists() and not args.force:
        raise SR4RecError(f"{out} already exists. Use --force to overwrite it.")
    ds = load_dataset(args.dataset)
    from .config import SplitConfig
    from .data.split import make_split

    try:  # the default split block: checks class sizes and duplicates
        split, split_error = make_split(ds, SplitConfig()), None
    except SR4RecError as err:
        split, split_error = None, str(err)
    short = ds.short_sides()
    median = float(short.median())
    suggested = "synthetic" if median >= 112 else "native-lr"
    try:
        dataset_value = Path(os.path.relpath(ds.root, out.parent)).as_posix()
    except ValueError:  # different drives on Windows
        dataset_value = str(ds.root)
    note = (f"[CHECK] suggested from the median short side {median:.0f} px "
            f"({'>=' if median >= 112 else '<'} 112 px); confirm, then delete this marker")
    values = {"dataset": dataset_value, "mode": suggested}
    text = render_config(values, {"mode": note})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    split_note = "has a split column" if ds.has_split_column else "has no split column (SR4Rec will split by ratios)"
    print(f"Dataset OK: {len(ds.table):,} images, {len(ds.classes):,} classes; labels.csv {split_note}.")
    print(f"Image short side: min {short.min()} px, median {median:.0f} px, max {short.max()} px.")
    if split is not None:
        print(f"Default split ({split.source}): {split.description.split(': ', 1)[-1]}.")
    for n in ds.notes + ((split.warnings or []) if split is not None else []):
        print(f"Note: {n}")
    print(f"Wrote {out}")
    if split_error:
        print("Warning: the default split block does not work for this dataset; edit the `split` block "
              f"of the configuration before running:\n{split_error}")
    print("Next: confirm `mode` (marked [CHECK]), add your SR models under `sr:`, then run")
    print(f"  sr4rec run {Path(os.path.relpath(out)).as_posix()} --dry-run")
    return 0


def _run(args: argparse.Namespace) -> int:
    from .config import load_config
    from .pipeline import Run

    loaded = load_config(args.config)
    result = Run(loaded, dry_run=args.dry_run, command=" ".join(["sr4rec"] + sys.argv[1:])).execute()
    if result.dry_run_text:
        print(result.dry_run_text)
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from .reproduce import reproduce

    return reproduce(args.demo, eval_only=args.eval_only, data_root=args.data_root, weights_root=args.weights_root,
                     checkpoints_root=args.checkpoints_root, output_dir=args.output_dir, device=args.device)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="sr4rec", description="Measure whether super-resolution helps closed-set "
                                "image recognition, compared with bicubic upscaling.", epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"SR4Rec {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{init,run,reproduce}")

    pi = sub.add_parser("init", help="check a dataset and write a fully commented sr4rec.yaml",
                        description="Check that a dataset follows the required format (images/ + labels.csv) and "
                        "write a fully commented configuration file with a suggested `mode` marked [CHECK].")
    pi.add_argument("dataset", help="dataset folder, for example data/my_dataset")
    pi.add_argument("--output", default="sr4rec.yaml", help="configuration file to write (default: sr4rec.yaml)")
    pi.add_argument("--force", action="store_true", help="overwrite an existing configuration file")
    pi.set_defaults(func=_init)

    pr = sub.add_parser("run", help="run the whole workflow and write report.md",
                        description="Downsample (synthetic mode), run every SR model, train and evaluate the "
                        "recognizers, compute the statistics and write the report. Cached steps are skipped.")
    pr.add_argument("config", help="configuration file (sr4rec.yaml)")
    pr.add_argument("--dry-run", action="store_true", help="check the configuration, the dataset and every SR model "
                    "on 2 test images, print the plan and the estimated time, and stop without training")
    pr.set_defaults(func=_run)

    pp = sub.add_parser("reproduce", help="re-run a published demo and compare with the expected results",
                        description="Run a demo configuration shipped with SR4Rec and compare the metrics with the "
                        "published expected results. Exit code 0: within tolerance; 2: outside tolerance; 3: missing "
                        "data, weights, checkpoints or expected results.")
    pp.add_argument("demo", help="demo name (run `sr4rec reproduce list` to see them)")
    pp.add_argument("--eval-only", action="store_true", help="use the published recognizer checkpoints instead of "
                    "training (download them first with scripts/fetch_checkpoints.py)")
    pp.add_argument("--data-root", default=None, help="folder that contains the demo datasets (default: the demo's "
                    "own default, usually data/)")
    pp.add_argument("--weights-root", default=None, help="folder that contains the SR weight files (default: the "
                    "demo's own default, usually weights/)")
    pp.add_argument("--checkpoints-root", default="checkpoints", help="folder with published checkpoints, used "
                    "with --eval-only (default: checkpoints/)")
    pp.add_argument("--output-dir", default="runs", help="parent folder for the run (default: runs/)")
    pp.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="device: auto = CUDA, then Apple GPU (MPS), then CPU (default: auto)")
    pp.set_defaults(func=_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except SR4RecError as err:
        print(f"Error: {err}", file=sys.stderr)
        return err.exit_code
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
