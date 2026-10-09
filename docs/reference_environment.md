# Reference environment

The expected results of the shipped demos (`src/sr4rec/demos/expected/`) are frozen on the machine
below. `fingerprint.yaml` of each run records the same fields, so any run can be compared with it.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 |
| CPU (latency) | 13th Gen Intel(R) Core(TM) i7-13700F |
| Operating system | Ubuntu 24.04.1 LTS |
| Python / PyTorch / CUDA | Python 3.12.3, PyTorch 2.11.0, CUDA 13.0 |
| SR4Rec | 0.1.0 |

Tolerances: full re-training `max(0.2 pp, 3 x sd over seeds)` per backbone and method;
`--eval-only` 0.01 pp (four decimals of the accuracy on the same type of hardware).

`requirements.lock` (repository root) pins one known-working set of exact package versions,
generated in a clean environment. It is not required to match the table above exactly - SR4Rec's
tolerances are designed to absorb this kind of drift between hardware and library versions across
time.

## Determinism

- **Same machine, same seed, `--eval-only`**: the evaluator and the recognizer are both
  deterministic given a fixed checkpoint, so results should match the published values closely
  (0.01 pp tolerance).
- **Same machine, same seed, full re-training**: training is not guaranteed bit-exact even on
  identical hardware (GPU kernel scheduling and library internals can vary run to run); the
  `max(0.2 pp, 3 x sd over seeds)` tolerance is sized for this.
- **Different GPU, GPU vendor or backend** (for example CUDA vs. Apple Silicon MPS vs. CPU):
  expect larger differences than the tolerance above, especially on a small test set. On a run
  with 100 test images, one flipped prediction alone moves Rank-1 by 1 pp, which already exceeds
  the `quickstart` demo's 0.2 pp tolerance; a mismatch there does not by itself mean the software
  is broken.
- When reporting an unexpected mismatch, attach the run's `fingerprint.yaml`
  (`runs/<run_name>/fingerprint.yaml`): it records the software versions, hardware and device used,
  so a genuine bug can be told apart from expected hardware drift.