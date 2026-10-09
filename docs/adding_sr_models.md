# Adding SR models

| I have ... | Use | CPU latency measured? |
|---|---|---|
| a `.pth` / `.safetensors` file of an architecture supported by spandrel | 1. `weights` | yes |
| the PyTorch code of the model (with or without weights) | 2. `module` | yes |
| only the SR output images, or a model that is not PyTorch | 3. `images` | no |

The bicubic baseline is always added; do not list it. Every SR model of a run must upscale by
exactly `scale`. Always check the configuration with `sr4rec run sr4rec.yaml --dry-run` before a real
run: it loads every model, runs it on 2 test images, checks the output size, prints the plan and the
estimated time, and stops without training.

## Method 1: a weights file (spandrel)

1. Download the weights from the authors' official page and put them in `weights/`.
2. Declare them:
   ```yaml
   sr:
     - name: swinir_classical
       weights: weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth
   ```
3. Run `--dry-run`. If the line of the model says `OK`, start the real run.
4. `spandrel could not detect the architecture`: use method 2 (you have the code) or method 3.

SR4Rec reads the architecture and the scale from the file, pads inputs that are smaller than the
architecture requires (reflection, cropped back afterwards) and records the SHA-256 of the file.
Use `tile: 256` (for example) if the GPU runs out of memory; tiles overlap by 16 px.

## Method 2: a Python class (`module`)

1. Put a file with the model class in `sr_models/` (or next to your configuration).
2. Declare it; `weights` is optional and is passed to `__init__(weights=...)`:
   ```yaml
   sr:
     - name: my_sr
       module: sr_models/tiny_espcn.py:TinyESPCN
       weights: weights/tiny_espcn_x4.pth
   ```
3. Run `--dry-run`. SR4Rec checks that the file and the class exist, that the class is a
   `torch.nn.Module` with an integer class attribute `scale` equal to the configured scale, and that
   a (2, 3, 16, 16) input gives a floating-point (2, 3, 16 x scale, 16 x scale) output without NaN.
4. If the class needs the authors' repository, install it (`pip install -e <repo>`) or add it to
   `sys.path` at the top of the file.

Required interface (see `examples/sr_models/tiny_espcn.py`):

- the class is a `torch.nn.Module` with an integer class attribute `scale`;
- `__init__` accepts one optional argument `weights` (a path or `None`);
- `forward(x)` receives a float32 RGB tensor in [0, 1] of shape (N, 3, H, W) and returns
  (N, 3, H x scale, W x scale);
- optional integer attribute `size_multiple`: inputs are padded to a multiple of it.

SR4Rec calls `model.eval()`, runs `forward()` under `torch.no_grad()`, moves the model to the
configured device and clamps the output to [0, 1].

**Security note:** method 2 executes the Python file. Only use code you trust. The SHA-256 of the
file and of the weights are written to `fingerprint.yaml`.

## Method 3: pre-computed SR images (`images`)

1. Run SR4Rec once (with other SR models, or with `sr: []`). The LR images SR4Rec gives to SR
   models are:
   - synthetic mode: `runs/<run_name>/lr/train/`, `lr/val/`, `lr/test/`;
   - native-lr mode: the dataset images themselves (the split of each image is in
     `runs/<run_name>/split.csv`).
   Protocol `matched` needs SR images of train, val and test; `fixed_recognizer` needs test only.
2. Run your SR method on those images and save PNG files under `sr_images/<name>/` with the same
   relative path and the extension `.png`, for example
   `lr/test/Abyssinian/Abyssinian_10.png` -> `sr_images/my_method/Abyssinian/Abyssinian_10.png`.
   All splits go into the same folder (paths are unique).
3. Declare it:
   ```yaml
   sr:
     - name: my_method
       images: sr_images/my_method/
   ```
4. Run `--dry-run`. SR4Rec reports missing files, files of the wrong size (each must be exactly
   `scale` x the LR size) and PNG files that match no dataset image (ignored).

`examples/sr_images/make_sr_images.py` writes such a folder with PIL Lanczos.

## Validated SR models

These weight files have been checked against the authors' own inference script and, for
bicubic-degradation models, against the PSNR reported in their paper (criteria and process in
`CONTRIBUTING.md`); SR4Rec labels their training degradation automatically by file hash. Any other
SR model still works (see the three methods above); it is shown as "unknown" instead.

| Model | File | Trained for | SHA-256 |
|---|---|---|---|
| Real-ESRGAN x4plus | `RealESRGAN_x4plus.pth` | real-world degradation | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| SwinIR-M x4 (classical) | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth` | bicubic degradation | `129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a` |
| SwinIR-M x4 (real-world, GAN) | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | real-world degradation | `b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6` |
| SPAN x4 (ch48) | `spanx4_ch48.pth` | bicubic degradation | `28fef8c6c845a0169afed9f9f566679990ef4c03fef9885298655fb5e4036402` |

Download pages: Real-ESRGAN <https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0>,
SwinIR <https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0>, SPAN
<https://github.com/hongyuanyu/SPAN>. The report shows the degradation each model was trained for,
and warns when a real-world model is run on clean bicubic LR images (synthetic mode).
