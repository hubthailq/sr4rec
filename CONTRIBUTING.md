# Contributing to SR4Rec

Thank you for helping. Bug reports, documentation fixes and new validated models are welcome.

## Development setup

```bash
git clone https://github.com/hubthailq/sr4rec.git
cd sr4rec
pip install -e ".[dev]"
pre-commit install
pytest                      # all tests (the slow end-to-end tests take about 1 minute on a CPU)
pytest -m "not slow"        # quick tests
```

Before opening a pull request:

- `ruff check .` and `pytest` pass;
- `python scripts/make_config_docs.py` was run if you changed `src/sr4rec/config.py`;
- `CHANGELOG.md` describes user-visible changes.

## Rules that protect comparability

The methodological rules in `docs/methods.md` (split, protocols, statistics, fixed values) are not
configurable on purpose. A change to any of them is a new minor version and must be justified in the
pull request with a small experiment.

## Adding a validated SR model

1. The weights are publicly downloadable from the authors, with a license that allows research use.
2. spandrel loads them (method 1); otherwise provide a method-2 module under `examples/`.
3. The output of SR4Rec matches the authors' inference script on 20 images (mean absolute
   difference below 1/255); for bicubic-degradation models, PSNR on Set5 and Set14 (Y channel,
   border = scale) is within 0.05 dB of the paper.
4. Add the SHA-256, file name and degradation type to `src/sr4rec/sr/validated.py`, the README and
   `docs/adding_sr_models.md`.

## Adding a validated backbone

The timm model trains with the fixed recipe on `pets_mini` and one of the shipped demo datasets without changes,
its CPU latency is measured, and its license allows research use. Add it to
`VALIDATED_BACKBONES` in `src/sr4rec/config.py`.

## Maintenance

Issues are triaged at least monthly. Supported Python versions follow the scientific Python
ecosystem (currently 3.10-3.12). Releases are tagged on GitHub and archived on Zenodo.
