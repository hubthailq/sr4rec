# Changelog

SR4Rec follows [semantic versioning](https://semver.org/). Before 1.0, minor versions may change
the configuration format; every change is listed here, and old configurations fail with a clear
message instead of being read differently. Report numbers only change between versions when this
file says so.

## [0.1.0] - 2026-10-10

### Added
- Commands `sr4rec init`, `sr4rec run [--dry-run]` and `sr4rec reproduce <demo> [--eval-only]`.
- Dataset format `images/` + `labels.csv`, validation with line numbers, split by column or ratios,
  cross-split duplicate detection (SHA-256).
- Modes `synthetic` (MATLAB-style bicubic downsampling) and `native-lr`; scales 2, 3, 4.
- Three ways to add SR models: spandrel weights, Python module, pre-computed images; bicubic baseline.
- Protocols `matched` (default) and `fixed_recognizer`; timm backbones with a fixed recipe.
- Rank-1, Rank-5, CMC, macro precision/recall/F1, PSNR/SSIM (Y channel), size bins.
- Paired bootstrap CI, sign-flip permutation test, Holm correction, verdicts, same-sign counts.
- CPU latency benchmark; report.md, plots, LaTeX table, before/after-SR comparison panels,
  fingerprint.yaml; caching of LR/SR images, splits and recognizers.
- Commented configuration and reference documentation generated from one schema.
- Device priority CUDA > Apple GPU (MPS) > CPU; out-of-memory fallbacks (automatic SR tiling,
  gradient accumulation, smaller evaluation batches, CPU as a last resort) so that runs never stop
  because the GPU memory is full.
- `requirements.lock`, a pinned snapshot of one known-working dependency set, and real
  reference-machine values (CPU, OS, PyTorch/CUDA, GPU) in `docs/reference_environment.md`.
- Frozen expected results for every shipped demo, so `sr4rec reproduce <demo>` grades against real
  tolerances.
- Published recognizer checkpoints for `d1_earvn`, `d1_earvn_fixed`, `d2_lfw` and `d3_cub`,
  fetchable with `scripts/fetch_checkpoints.py <demo>`, so `sr4rec reproduce <demo> --eval-only`
  works without retraining.
