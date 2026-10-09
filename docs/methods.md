# Methods

This page describes what SR4Rec computes. The rules are fixed so that reports are comparable.

## Pipeline

1. **Dataset and split.** The dataset is checked (docs/preparing_dataset.md) and split so that
   every class is in train, val and test and no image content is in two splits.
2. **LR images.** `synthetic`: each HR image is cropped at the right and bottom to a multiple of
   `scale` (mod-crop) and downsampled by `scale` with MATLAB-style bicubic interpolation
   (`imresize`, cubic kernel a = -0.5, antialiasing), then stored as 8-bit PNG. `native-lr`: the
   dataset images are the LR images.
3. **SR.** Every SR model (and bicubic, which uses the same MATLAB-style routine to upscale)
   produces `scale` x the LR size. Outputs are clamped to [0, 1], stored as PNG and cached.
4. **Recognizer input.** Every image (bicubic, SR output or HR) is prepared identically:
   - *letterbox*: the longer side is resized to `input_size` (antialiased bicubic), the aspect ratio
     is kept, and the image is centred on a square canvas filled with the ImageNet mean colour
     (0.485, 0.456, 0.406); images are never stretched;
   - *ImageNet normalization*: `(x - mean) / std` with mean (0.485, 0.456, 0.406) and std
     (0.229, 0.224, 0.225), the same for every backbone; the padding is exactly 0 afterwards.
   Training augmentation crops square regions only, so it does not change the aspect ratio either.
5. **Protocol.**
   - `matched` (default): for each method (bicubic, every SR model and, in synthetic mode, HR) one
     recognizer per backbone and seed is trained on that method's train images, selected on that
     method's val images, and tested on that method's test images.
   - `fixed_recognizer`: one recognizer per backbone and seed is trained on HR images (synthetic)
     or bicubic images (native-lr); only the test images pass through each SR model. The report
     always explains that SR outputs are then out of the recognizer's training domain.
6. **Training recipe.** timm backbone (ImageNet-pretrained by default) with a new classifier;
   AdamW, cosine schedule with one warm-up epoch, cross-entropy, mixed precision on CUDA;
   RandomResizedCrop (scale 0.8-1.0, square crops only, so the aspect ratio is never changed), small ColorJitter, optional horizontal flip; the checkpoint
   with the best val Rank-1 is kept (earliest epoch on ties). The test split is never used for
   training or checkpoint selection. Deterministic algorithms are requested and every data-loader
   worker is seeded. `num_workers` changes the augmentation random streams, so it is part of the
   checkpoint cache key.

## Devices and memory

`runtime.device: auto` uses a CUDA GPU, otherwise an Apple GPU (MPS), otherwise the CPU; a device
that was requested but is not available falls back along the same order with a warning. Mixed
precision is used on CUDA only. Out-of-memory errors never stop a run:

- **SR:** the image is retried with tiled inference, halving the tile size down to 64 px (16 px
  overlap, averaged), then the model continues on the CPU for the remaining images.
- **Training:** the recognizer is retrained from the beginning with twice the gradient accumulation
  (micro-batches of `batch_size / k` images, one optimizer step per `batch_size` images, same sample
  order and schedule) until the micro-batch is one image, then on the CPU. BatchNorm statistics
  then come from the smaller micro-batches.
- **Evaluation:** the batch is halved (eval-mode results do not depend on it), then the CPU is used.

Each fallback is written to the report (Warnings and notes) and, for training, to the checkpoint
metadata (`accumulation`, `device`).

## Metrics

- **Rank-1 accuracy** (primary), **Rank-5**, **CMC** Rank-1 to Rank-10 (n/a when k exceeds the
  number of classes). Scores are computed in float32; the rank of the true class is 1 + the number
  of classes with a higher score, where equal scores are ordered by class index as in argmax, so
  Rank-1 correct means exactly "predicted label = true label".
- **Macro precision, recall, F1** over all dataset classes (scikit-learn, `zero_division=0`); in the
  size-bin rows of `metrics.csv`, over the classes present in that bin.
- **PSNR and SSIM** (synthetic mode only) between each SR output and the mod-cropped HR image, on
  the Y channel of MATLAB `rgb2ycbcr`, with a border of `scale` pixels removed. SSIM follows Wang et
  al. (2004): Gaussian window 11 x 11, sigma 1.5, K1 = 0.01, K2 = 0.03, L = 255, valid filtering.
- **Size bins** by the short side of the LR image given to SR: <32, 32-63, 64-127, >=128 px.

## Statistics

- For each backbone, method and test image, Rank-1 correctness is averaged over the seeds.
- Delta = mean per-image score of the SR model minus that of bicubic, in percentage points.
- 95% CI: percentile bootstrap over test images, 10,000 resamples, seed 0.
- p: two-sided paired sign-flip permutation test on the per-image differences, 10,000 random sign
  assignments, seed 0, p = (1 + count) / (1 + 10,000).
- Holm correction over the SR-vs-bicubic comparisons of one backbone (and one size bin). Each
  backbone is treated as its own question, so there is no correction across backbones.
- Verdict: `▲ significant gain` if Delta > 0, the CI excludes 0 and Holm p < 0.05;
  `▼ significant harm` if Delta < 0 under the same conditions; `● no difference` otherwise;
  `insufficient data` for size bins with fewer than 50 test images (no test is run).
- Same sign: the number of seeds whose own Delta has the sign of the mean Delta.
- Only Rank-1 is tested; the other metrics are descriptive (mean ± sd over seeds).

## CPU latency

Batch 1, square LR input of `latency.lr_size`, `latency.threads` CPU threads, 10 warm-up runs and
100 timed runs; median and 95th percentile in milliseconds, for each SR model alone, each
recognizer alone and the full pipeline (SR, letterbox, recognizer). Not measured for method 3.

## Fixed values

Size bins, 10,000 resamples, alpha 0.05, the 50-image threshold, the MATLAB bicubic kernel and the
number of latency runs are not configurable; they are printed in the Setup section of every report.
