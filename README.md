# SR4Rec

[![CI](https://github.com/hubthailq/sr4rec/actions/workflows/ci.yml/badge.svg)](https://github.com/hubthailq/sr4rec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)

**Does placing a super-resolution (SR) model in front of a recognizer improve closed-set
recognition, compared with plain bicubic upscaling?** SR4Rec answers this question for your own
dataset and your own SR models with one configuration file and three commands, under a fixed
protocol, and writes a report from the results.

## Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Preparing your dataset](#4-preparing-your-dataset)
5. [Adding SR models](#5-adding-sr-models)
6. [Examples](#6-examples)
7. [Configuration and reports](#7-configuration-and-reports)
8. [Reproducing the demos](#8-reproducing-the-demos)
9. [Limitations, roadmap, contributing, licenses](#9-limitations-roadmap-contributing-licenses)
10. [Citation and license](#10-citation-and-license)

## 1. Overview

SR4Rec is for researchers and engineers who work with low-resolution images of identities (faces,
ears, ...) or object classes (species, breeds, ...) and need to decide whether an SR model is worth
adding to a recognition system.

```
images/ + labels.csv ──> split ──> LR images ──> bicubic | SR model 1 | ... | SR model k
                        (every class           (synthetic: MATLAB bicubic ↓scale;
                         in train/val/test)      native-lr: images as given)
        ──> letterbox + ImageNet normalization ──> recognizer per backbone × seed (protocol matched or fixed_recognizer)
        ──> Rank-1/Rank-5/CMC, macro P/R/F1, PSNR/SSIM ──> paired statistics vs bicubic
        ──> report.md, CSV files, plots, LaTeX table, before/after-SR panels
```

Every run uses:

- the same test images and preprocessing for every method (letterbox, ImageNet normalization; see
  [Methods](docs/methods.md));
- a closed-set split: every class is in train, val and test; no image is in two splits;
- protocol `matched` by default: each SR model's recognizer trains on that model's own images;
- paired statistics per SR model and backbone (bootstrap CI, sign-flip permutation, Holm
  correction), also by image size;
- a fingerprint (versions, hardware, data and weight hashes) for every run.

Three commands only: `sr4rec init`, `sr4rec run`, `sr4rec reproduce`.

## 2. Installation

Requirements: Python 3.10-3.12, PyTorch 2.3 or newer. A CUDA GPU is recommended for real datasets;
Apple-silicon GPUs (MPS) are supported; the examples run on a CPU. With `runtime.device: auto`
SR4Rec uses a CUDA GPU if there is one, otherwise an Apple GPU, otherwise the CPU.

```bash
git clone https://github.com/hubthailq/sr4rec.git
cd sr4rec
pip install -e ".[dev]"
pytest -m "not slow"                    # quick self-test (about 30 s)
```

Install the PyTorch build for your GPU first if needed (<https://pytorch.org/get-started/>).

**Offline use.** The ImageNet-pretrained backbones are downloaded by timm from Hugging Face on
first use and cached in `~/.cache/huggingface/` (change with the `HF_HOME` environment variable).
On a machine without internet access, run once on a connected machine:

```bash
python -c "import timm; [timm.create_model(m, pretrained=True) for m in ['resnet18', 'mobilenetv3_small_100', 'convnext_tiny']]"
```

and copy the Hugging Face cache folder to the offline machine (same `HF_HOME`). SR4Rec never
downloads SR weights; you provide them.

## 3. Quick start

No dataset or SR weights yet? Build the small shipped demo once (needs your own copy of
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), the `images.tar.gz` and
`annotations.tar.gz` archives) and run it end to end in about 5 minutes on a laptop CPU (see "quickstart" in
[Reproducing the demos](#8-reproducing-the-demos)):

```bash
mkdir oxford_pet && cd oxford_pet
curl -LO https://thor.robots.ox.ac.uk/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/pets/annotations.tar.gz
tar xzf images.tar.gz && tar xzf annotations.tar.gz
cd ..

python scripts/make_examples.py --pets oxford_pet
python examples/sr_models/train_tiny_espcn.py
sr4rec reproduce quickstart
```

With your own dataset, keep everything for one study in one project folder and run every command
from it:

```
my_project/
├── sr4rec.yaml      # written by `sr4rec init`, edited by you
├── data/<name>/     # your dataset: images/ + labels.csv (see "Preparing your dataset")
├── weights/         # SR weight files you provide
└── runs/            # created by SR4Rec; report.md, CSV files, plots, ... per run
```

```bash
mkdir -p my_project/data/my_dataset && cd my_project
# copy your images/ and labels.csv into data/my_dataset/ (see "Preparing your dataset")
sr4rec init data/my_dataset/        # 1. validate the dataset, write ./sr4rec.yaml
# 2. edit sr4rec.yaml: put your SR weights in weights/ and list them under `sr:`,
#    set the split if needed, confirm every [CHECK] line (then delete the marker)
sr4rec run sr4rec.yaml --dry-run    #    check everything in under a minute
sr4rec run sr4rec.yaml              # 3. run; then open runs/<run_name>/report.md
```

Formatting your images and labels: [Preparing your dataset](#4-preparing-your-dataset). Providing
the SR weights that go under `sr:`: [Adding SR models](#5-adding-sr-models).

Every path in `sr4rec.yaml` is relative to the folder that contains it (or absolute), so the whole
project folder can be moved to another machine. `sr4rec run` refuses to start while a `[CHECK]`
marker remains. Running again reuses every cached step (SR outputs and trained recognizers); adding
an SR model trains only the recognizers of that model.

Protocol `matched` trains one recognizer per SR model, backbone and seed, so it gets expensive with
many SR models. For large studies, screen them first with `protocol: fixed_recognizer`, then run
`matched` on the promising ones; `--dry-run` prints the training count and an estimated time either
way.

Python API (identical results to the CLI):

```python
from sr4rec import Run, load_config

result = Run(load_config("sr4rec.yaml")).execute()
print(result.report)          # path of report.md
```

## 4. Preparing your dataset

```
data/<dataset_name>/
├── images/        # all images; any sub-folder structure is allowed
└── labels.csv     # columns: path,label[,split]
```

| Column | Required | Rule |
|---|---|---|
| `path` | yes | relative to `images/`, with `/`, unique |
| `label` | yes | class to recognize (identity, species, ...), any string |
| `split` | no | `train`, `val` or `test` |

`sr4rec init` checks the format (missing columns, missing or unreadable files, duplicate paths,
classes too small for the ratios, identical images in two splits) and reports errors by line
number of `labels.csv`. Split rules, ready-to-copy conversion snippets and common errors:
[docs/preparing_dataset.md](docs/preparing_dataset.md).

## 5. Adding SR models

| I have ... | Use | CPU latency measured? |
|---|---|---|
| a `.pth` / `.safetensors` file of an architecture spandrel supports | 1. `weights` | yes |
| the PyTorch code of the model (with or without weights) | 2. `module` | yes |
| only SR output images, or a model that is not PyTorch | 3. `images` | no |

```yaml
sr:
  - name: realesrgan                          # method 1: weights file (spandrel)
    weights: weights/RealESRGAN_x4plus.pth
  - name: my_sr                               # method 2: your PyTorch class
    module: sr_models/tiny_espcn.py:TinyESPCN
    weights: weights/tiny_espcn_x4.pth        # optional; passed to __init__(weights=...)
  - name: my_method                           # method 3: pre-computed SR images
    images: sr_images/my_method/
```

**Security:** method 2 runs the Python file you give it; only use code you trust.

Validated recognition backbones (timm): `resnet18`, `mobilenetv3_small_100`, `convnext_tiny`. Any
other timm model name runs with an "experimental" warning. Method interfaces, the four validated SR
models (with SHA-256 and download links) and the `--dry-run` check:
[docs/adding_sr_models.md](docs/adding_sr_models.md).

## 6. Examples

All examples use small subsets of **real** public datasets and run on a CPU. The subsets are
rebuilt from your own downloads (SR4Rec does not redistribute the images):

```bash
python scripts/make_examples.py --pets <path_to_oxford_iiit_pet>   # examples/data/pets_mini
python scripts/make_examples.py --earvn <path_to_EarVN1.0>         # examples/data/earvn_mini
python scripts/make_examples.py --lfw <path_to_lfw>                # examples/data/lfw_mini (faces)
python examples/sr_models/train_tiny_espcn.py                      # example SR weights
```

| Config | Shows | Time (laptop CPU) |
|---|---|---|
| `examples/configs/full_reference.yaml` | every key with full comments (= `sr4rec init` output) | n/a |
| `01_quickstart.yaml` | minimal workflow (= `sr4rec reproduce quickstart`) | ~5 min |
| `02_synthetic_pth.yaml` | method 1 with the 4 validated weight files | 1-2 h (minutes on a GPU) |
| `03_native_lr.yaml` | `mode: native-lr`, results by image size (ears) | ~5 min |
| `04_fixed_recognizer.yaml` | cheap screening protocol | ~3 min |
| `05_split_ratios.yaml` | your own split ratios | ~5 min |
| `06_sr_module.yaml` | method 2 (Python class) | ~5 min |
| `07_sr_images.yaml` | method 3 (pre-computed images) | ~5 min |
| `08_faces_lfw.yaml` | face identity recognition | ~5 min |

Run for example `sr4rec run examples/configs/01_quickstart.yaml` from the repository root. Faces
are biometric data: SR4Rec ships no face image; follow the original dataset's terms and your
data-protection rules. See [examples/README.md](examples/README.md).

## 7. Configuration and reports

`sr4rec init` writes a configuration in which every key documents its type, whether it is
required, its default and the complete list of accepted values; unknown keys and invalid values
are rejected with a message that names the key and the accepted values. Complete table (generated
from the schema): [docs/config_reference.md](docs/config_reference.md).

`report.md` starts with the question and a rule-generated answer, followed by tables, plots and a
fingerprint. The verdict of every SR model and backbone follows one rule:

| Verdict | Condition |
|---|---|
| `▲ significant gain` | Δ > 0, 95% CI excludes 0, p (Holm) < 0.05 |
| `▼ significant harm` | Δ < 0, 95% CI excludes 0, p (Holm) < 0.05 |
| `● no difference` | otherwise |
| `insufficient data` | size bin with fewer than 50 test images |

Report sections and glossary: [docs/report_guide.md](docs/report_guide.md); methods and statistics:
[docs/methods.md](docs/methods.md); every file written to `runs/<run_name>/`:
[docs/output_files.md](docs/output_files.md).

## 8. Reproducing the demos

Demo definitions and expected results are shipped with the package (`sr4rec reproduce list`). Run
every command below from the repository root; `data/`, `weights/` and `checkpoints/` are created
there.

| Demo | Dataset | Setting | What it demonstrates |
|---|---|---|---|
| `quickstart` | pets_mini | synthetic x4, CPU | a small end-to-end run, about 5 min |
| `d1_earvn` | [EarVN1.0](https://data.mendeley.com/datasets/yws3v3mwx3/4) | native-lr x4, matched, ResNet-18, 3 seeds | main result, results by image size |
| `d1_earvn_fixed` | EarVN1.0 | as `d1_earvn` with protocol `fixed_recognizer` | what the protocol changes |
| `d2_lfw` | [LFW](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) (>= 20 images per person) | synthetic x4, matched, ResNet-18, 3 seeds | does PSNR/SSIM predict recognition |
| `d3_cub` | [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) | synthetic x4, matched, ResNet-18, 1 seed | another domain (single run) |

```bash
python scripts/prepare_earvn.py --earvn <path> --out data/earvn
python scripts/prepare_lfw.py   --lfw   <path> --out data/lfw
python scripts/prepare_cub.py   --cub   <path> --out data/cub
# put the four validated SR weight files in weights/

python scripts/fetch_checkpoints.py d1_earvn     # level 1: published recognizers, no training
sr4rec reproduce d1_earvn --eval-only
sr4rec reproduce d1_earvn                        # level 2: full re-training on a GPU
```

`sr4rec reproduce <demo>` needs that demo's dataset and SR weights locally. Without them it exits
with code 3 and writes the demo configuration to `runs/<demo>_config.yaml`, which you can then run
with `sr4rec run`.

`sr4rec reproduce quickstart` needs no downloaded checkpoints, only the `pets_mini` dataset and
the example SR weights, built once ([Quick start](#3-quick-start)):
```bash
python scripts/make_examples.py --pets <path_to_oxford_iiit_pet>
python examples/sr_models/train_tiny_espcn.py
```
After that it retrains on the spot in about 5 minutes on a laptop CPU.

Exit codes: **0** within tolerance, **2** outside tolerance, **3** missing data/weights/checkpoints,
**1** any other error. Reference hardware: [docs/reference_environment.md](docs/reference_environment.md).

## 9. Limitations, roadmap, contributing, licenses

**Limitations of v0.1**: closed-set identification only, no verification or open-set protocols, no
attribute classification; one scale per run; a fixed training recipe; PSNR/SSIM only in synthetic
mode; CPU latency only for methods 1 and 2.

**Roadmap**: verification protocols, attribute tasks, multiple scales per run, more validated SR
models and backbones.

**Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md) (including the criteria for a validated SR
model or backbone) and [CHANGELOG.md](CHANGELOG.md) for the versioning policy. FAQ:
[docs/faq.md](docs/faq.md).

**Licenses**: sources and citations of every dataset, weight file and checkpoint are in
[DATA_AND_MODEL_LICENSES.md](DATA_AND_MODEL_LICENSES.md).

## 10. Citation and license

If you use SR4Rec, please cite the software (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{sr4rec,
  title   = {SR4Rec: A reproducible toolkit for measuring whether super-resolution helps closed-set image recognition},
  author  = {Le Quang, Thai and Lam Tran Tuan, Dzi and Truong Hoang, Vinh},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/hubthailq/sr4rec},
  doi     = {10.5281/zenodo.23401553}
}
```

SR4Rec is released under the [MIT License](LICENSE). It builds on PyTorch, timm, spandrel, NumPy,
SciPy, statsmodels, scikit-learn, pandas, Pillow, matplotlib, pydantic and Jinja2. SR weights,
backbone weights and datasets keep their own licenses (see DATA_AND_MODEL_LICENSES.md).
