# FAQ

**The GPU runs out of memory.**
The run does not stop. SR retries with smaller tiles (down to 64 px), then falls back to the CPU.
Recognizer training switches to gradient accumulation, then the CPU. Evaluation halves its batch.
Every fallback is listed under "Warnings and notes" in the report. To avoid the slowdown, set
`tile: 256` on large SR models or lower `batch_size`.

**`spandrel could not detect the architecture`.**
Use method 2 with the authors' model code, or method 3 with images produced by their script.

**`upscales x2, but the configuration uses scale 4`.**
All SR models of a run must have the configured scale. Run separate projects for other scales.

**My dataset is very small.**
Verdicts are computed over test images. With few test images the CIs are wide and most verdicts
will be `● no difference`. Size bins with fewer than 50 test images are not tested.

**Results differ between machines.**
Expected on a different GPU or backend; see [Determinism](reference_environment.md#determinism).
Compare `fingerprint.yaml` files to tell a real bug apart from hardware drift.

**The first run is slow, later runs are fast.**
SR outputs and recognizers are cached in `runs/.cache/`. Delete it to start from scratch.
