# Preparing your dataset

SR4Rec accepts exactly one dataset format. It does not guess folder structures; convert your
data once with a few lines of code (examples below and in `examples/dataset_conversion/`).

## Format

```
data/<dataset_name>/
|-- images/        all images; any sub-folder structure is allowed
`-- labels.csv     exactly one row per image
```

| Column | Required | Rule |
|---|---|---|
| `path` | yes | path relative to `images/`, with `/`, unique |
| `label` | yes | the class to recognize (an identity, a species, ...), any string |
| `split` | no | `train`, `val` or `test` |

- Image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. Grayscale and other modes are converted to RGB
  (the report says how many).
- SR4Rec stores processed images as `<path without extension>.png`, so two paths that differ only
  by extension or letter case are rejected.
- Closed-set recognition: every class must appear in train, val and test.
- `mode: synthetic` needs high-resolution images (SR4Rec downsamples them by `scale`); use
  `mode: native-lr` for images that are already small. `sr4rec init` suggests a mode from the median
  short side (at least 112 px suggests synthetic) and marks it `[CHECK]`.

Examples: `examples/data/*/labels.csv` after running `scripts/make_examples.py`.

## Declaring the split

The `split` block of `sr4rec.yaml` chooses where the split comes from:

- `source: auto` (default): the `split` column if labels.csv has one, otherwise `ratios`.
- `source: column`: the `split` column is required. It needs `train` and `test` rows; if there is no
  `val` row, `val_from_train` (default 0.1) of each class's train images is moved to val.
- `source: ratios`: any `split` column is ignored; SR4Rec splits every class by `ratios`
  (default train 0.7 / val 0.1 / test 0.2, each strictly between 0 and 1, summing to 1).

Rounding, per class of n images: test = max(1, floor(n x test + 0.5)), val = max(1, floor(n x val +
0.5)), train = the rest, which must be at least 1. Classes that are too small are listed with the
minimum number of images they need (3 with the default ratios).

The split uses `split.seed`, independent of the recognizer seeds. It is stored as `split.csv` in
every run folder and reused as long as the data and the `split` block do not change.

When SR4Rec creates or completes the split, identical images (same SHA-256, even under another
file name) are always placed in the same split. When the split comes from your `split` column,
SR4Rec stops if the same image content appears in two splits. `sr4rec init` already runs these
checks with the default `split` block.

## Conversion snippets

```python
# Folder-per-label layout (images/<label>/*.jpg) -> labels.csv
from pathlib import Path
import pandas as pd

root = Path("data/my_dataset/images")
exts = {".jpg", ".jpeg", ".png", ".bmp"}
rows = [{"path": p.relative_to(root).as_posix(), "label": p.parent.name}
        for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
pd.DataFrame(rows).to_csv("data/my_dataset/labels.csv", index=False)
```

```python
# Identity encoded in file names (e.g. 017_s1_03.jpg -> person 017) -> labels.csv
from pathlib import Path
import pandas as pd

root = Path("data/my_dataset/images")
rows = []
for p in sorted(root.glob("*.jpg")):
    person = p.stem.split("_")[0]
    rows.append({"path": p.name, "label": person})
pd.DataFrame(rows).to_csv("data/my_dataset/labels.csv", index=False)
```

Ready-made scripts: `examples/dataset_conversion/from_folder_per_label.py`,
`from_filename_pattern.py` and `from_annotation_csv.py`.

## Natively low-resolution faces (TinyFace)

TinyFace contains natural low-resolution faces (about 20 x 16 px on average) and fits
`mode: native-lr`. Its official page does not state terms of use; contact the dataset authors
before using it. To convert it, give each identity its own folder under `images/` and run
`from_folder_per_label.py`. SR4Rec ships no configuration or subset for TinyFace.

## Common `sr4rec init` errors

| Message | Fix |
|---|---|
| `must contain images/ and labels.csv` | put the images in `images/` and write `labels.csv` next to it |
| `missing: label` / `unknown columns` | the header must be `path,label` or `path,label,split` |
| `line N: file not found` | `path` is relative to `images/` and case-sensitive |
| `listed more than once` | one row per image |
| `split value ... is not allowed` | use `train`, `val`, `test` (lower case is applied automatically) |
| `class ... has k image(s); at least m are needed` | add images, merge or drop the class, or change `ratios` |
| `identical images in different splits` | remove the duplicate or keep all copies in one split |
| `smaller than scale` (synthetic) | use `mode: native-lr` for low-resolution images |
