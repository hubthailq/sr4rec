# Output files

```
runs/<run_name>/
├── report.md             # human-readable report, read this first
├── plots/                # accuracy_by_size.png, cmc.png, quality_vs_accuracy.png (synthetic)
├── metrics.csv           # backbone, method, seed, size_bin, n_images, rank1, rank5, precision,
│                         #   recall, f1, psnr, ssim (fractions in [0, 1]; psnr in dB)
├── cmc.csv               # backbone, method, seed, rank1 ... rank10 (n/a when k > classes)
├── stats.csv             # backbone, size_bin, method, n_images, delta, ci_low, ci_high (pp),
│                         #   p_raw, p_holm, same_sign, verdict
├── predictions.csv       # backbone, method, seed, path, label, pred, true_rank, correct
├── quality.csv           # method, path, psnr, ssim per test image (synthetic)
├── latency.csv           # component (sr | recognizer | pipeline), sr, backbone, median_ms, p95_ms
├── checkpoints.csv       # recognizer checkpoints used (key, file, best epoch, best val Rank-1)
├── table.tex             # main results table (LaTeX, booktabs)
├── split.csv             # the frozen train/val/test split (path, label, split)
├── comparisons/          # one before/after-SR PNG per selected test image + index.csv
├── lr/                   # LR images of train/, val/, test/ (synthetic mode), for external SR
├── sr4rec.yaml           # the exact configuration used (paths relative to this folder)
└── fingerprint.yaml      # environment, versions, seeds, data and weight hashes
```

`method` is `bicubic`, the SR name from the configuration, or `hr` (HR upper bound, synthetic mode).
