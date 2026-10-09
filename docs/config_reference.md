# Configuration reference

This file is generated from the configuration schema (`src/sr4rec/config.py`) by
`python scripts/make_config_docs.py`. Do not edit it by hand.

Every key of `sr4rec.yaml` is listed with its type, whether it is required, its default
value and the complete set of accepted values. Unknown keys are rejected.

| Key | Type | Required | Default | Allowed | Description |
|---|---|---|---|---|---|
| `dataset` | path | yes | `none` | an existing folder | Dataset folder. It must contain `images/` and `labels.csv` (see docs/preparing_dataset.md). A relative path is resolved from the folder that contains this file. Recommended location: data/<dataset_name>. |
| `mode` | enum | yes | `none` | synthetic : images are high resolution (HR). SR4Rec downsamples them by `scale`; PSNR/SSIM and an HR upper bound are reported.<br>native-lr : images are already low resolution. No downsampling; PSNR/SSIM are not available (there is no HR reference). | Resolution of the images in `images/`. |
| `scale` | integer (enum) | no | `4` | 2 \| 3 \| 4 | Upscaling factor. In `synthetic` mode it is also the downsampling factor. Every SR model below must have exactly this scale, otherwise the run stops. |
| `protocol` | enum | no | `matched` | matched          : for every SR model (and bicubic), the train, val and test images all pass through that SR model; one recognizer is trained per SR model, backbone and seed. Answers: "does a system built with this SR model recognize better?"<br>fixed_recognizer : one recognizer per backbone and seed, trained on HR images (synthetic) or bicubic-upscaled images (native-lr); only the test images pass through SR. Answers: "does adding SR in front of an existing recognizer help?" Fewer recognizer trainings. | How the recognizer is trained with respect to SR. |
| `split.source` | enum | no | `auto` | auto   : use the `split` column of labels.csv if it exists, otherwise split by `ratios`<br>column : use the `split` column; stop with an error if it is missing<br>ratios : ignore any `split` column and split by `ratios` | Where the train/val/test assignment comes from. |
| `split.ratios` | mapping with keys train, val, test | no | `{train: 0.7, val: 0.1, test: 0.2}` | each value greater than 0 and less than 1; the three values must sum to 1.0 | Fractions of images for each split, used when SR4Rec creates the split. Applied inside every class (each class keeps at least one image in train, val and test). The report prints the actual counts. |
| `split.ratios.train` | float | no | `0.7` | greater than 0 and less than 1 | Fraction of images in the train split. |
| `split.ratios.val` | float | no | `0.1` | greater than 0 and less than 1 | Fraction of images in the val split. |
| `split.ratios.test` | float | no | `0.2` | greater than 0 and less than 1 | Fraction of images in the test split. |
| `split.val_from_train` | float | no | `0.1` | greater than 0, at most 0.5 | Fraction of the train images moved to val when the `split` column has train and test rows but no val rows. |
| `split.seed` | integer | no | `0` | 0 to 2147483647 | Seed used to create the split. It is independent of the recognizer seeds, so all recognizer seeds share the same split. |
| `sr[].name` | string | yes | `none` | letters, digits, "_" and "-"; must be unique; "bicubic" and "hr" are reserved | Name used in reports. |
| `sr[].weights` | path | conditional (method 1; optional in method 2) | `null` | any existing file | Method 1: a .pth or .safetensors file whose architecture spandrel can detect (its scale must equal `scale`). Method 2: optional file passed to the class as __init__(weights=<path>). Recommended location: weights/. |
| `sr[].module` | string "<file.py>:<ClassName>" | conditional (method 2) | `null` | an existing Python file and a class defined in it | Method 2: a Python file and a torch.nn.Module class with an integer attribute `scale` equal to `scale` and forward(x) mapping (N, 3, H, W) in [0, 1] to (N, 3, H*scale, W*scale). Recommended location: sr_models/. |
| `sr[].images` | path | conditional (method 3; no other source key) | `null` | an existing folder | Method 3: a folder of PNG images with the same relative paths as the dataset images (train, val and test images for protocol `matched`; test images only for `fixed_recognizer`). Recommended location: sr_images/<name>/. CPU latency is not measured for this source. |
| `sr[].tile` | integer or null | no | `null` | null (process the whole image) \| 64 to 1024 | LR tile size in pixels for tiled inference (tiles overlap by 16 px). Use it only if the GPU runs out of memory. Valid for methods 1 and 2 only. |
| `recognizer.backbones` | list of strings | no | `[resnet18, mobilenetv3_small_100, convnext_tiny]` | validated by SR4Rec: resnet18 \| mobilenetv3_small_100 \| convnext_tiny<br>any other timm model name runs with an "experimental" warning | Recognition backbones (timm model names, fine-tuned for the dataset classes). |
| `recognizer.pretrained` | boolean | no | `true` | true \| false | Start from ImageNet-pretrained weights downloaded by timm (needs internet access or a filled HF_HOME cache). |
| `recognizer.input_size` | integer | no | `224` | 64 to 512, multiple of 32 | Side length of the square recognizer input. Every image is letterboxed (longer side resized to this value, aspect ratio kept, ImageNet-mean padding; never stretched) and ImageNet-normalized. |
| `recognizer.seeds` | list of integers | no | `[0, 1, 2]` | 1 to 10 distinct integers from 0 to 2147483647 (3 or more recommended) | Random seeds. One recognizer is trained per image pipeline, backbone and seed. |
| `training.epochs` | integer | no | `30` | 1 to 500 | Number of training epochs. The checkpoint with the best val Rank-1 accuracy is kept. |
| `training.batch_size` | integer | no | `64` | 1 to 1024 | Mini-batch size. |
| `training.learning_rate` | float | no | `0.0003` | greater than 0, at most 1.0 | AdamW learning rate (cosine schedule, 1 warm-up epoch). |
| `training.weight_decay` | float | no | `0.05` | 0.0 to 1.0 | AdamW weight decay. |
| `training.horizontal_flip` | boolean | no | `false` | true \| false | Random horizontal flip during training. Off by default because flipping can remove identity cues (e.g. left vs right ear). |
| `training.num_workers` | integer | no | `4` | 0 to 64 | Data-loading worker processes. |
| `latency.enabled` | boolean | no | `true` | true \| false | Measure CPU latency of each SR model, each backbone and the full pipeline. |
| `latency.lr_size` | integer | no | `64` | 16 to 512 | Side length of the square LR input used for timing. |
| `latency.threads` | integer | no | `4` | 1 to 64 | Number of CPU threads used for timing. |
| `comparisons.enabled` | boolean | no | `true` | true \| false | Export one PNG per selected test image that combines the LR input, bicubic, every SR model and (synthetic mode) the HR image, each with an English caption: method name, predicted label with a correct/wrong mark, and PSNR/SSIM (synthetic mode only). |
| `comparisons.count` | integer | no | `24` | 1 to 1000 | Number of test images to export. |
| `comparisons.selection` | enum | no | `mixed` | mixed  : one third corrected by at least one SR model (bicubic wrong, SR right), one third degraded by at least one SR model (bicubic right, SR wrong), one third random; fixed selection seed 0<br>random : uniformly at random from the test set; fixed selection seed 0<br>all    : every test image (ignores `count`; can produce many files) | How the test images are chosen. |
| `comparisons.backbone` | string or null | no | `null` | null (first backbone in `recognizer.backbones`) \| any name listed in `recognizer.backbones` | Backbone whose predictions (first seed) are written in the captions. |
| `runtime.device` | enum | no | `auto` | auto : CUDA GPU if available, otherwise Apple GPU (MPS), otherwise the CPU<br>cuda : CUDA GPU; if none is found, fall back to MPS, then the CPU, with a warning<br>mps  : Apple GPU (Apple silicon); if unavailable, fall back to the CPU with a warning<br>cpu  : always the CPU (slow; intended for small datasets and tests) | Device for SR inference and recognizer training. If the GPU runs out of memory, the run never stops: SR switches to smaller tiles, training to gradient accumulation (same effective batch size) and evaluation to smaller batches, and as a last resort the step continues on the CPU; the report lists every such fallback. |
| `runtime.output_dir` | path | no | `runs` | any folder path | Parent folder for all runs. |
| `runtime.cache_dir` | path or null | no | `null` | null (use <output_dir>/.cache) \| any folder path | Folder for cached LR/SR images and trained recognizer checkpoints, shared by all runs so that repeated runs are fast. Deleting it is always safe. |
| `runtime.run_name` | string or null | no | `null` | null (automatic: <dataset>_<YYYY-MM-DD>_<HHMM>) \| letters, digits, "_" and "-" | Name of this run's folder inside `output_dir`. |
