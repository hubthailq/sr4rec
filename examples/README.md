# SR4Rec examples

Every example runs on a CPU. Run all commands from the repository root.

| I want to ... | Open |
|---|---|
| see every configuration key with its documentation | `configs/full_reference.yaml` |
| run the minimal workflow | `configs/01_quickstart.yaml` |
| use SR weight files (`.pth`, spandrel) | `configs/02_synthetic_pth.yaml` |
| evaluate natively low-resolution images | `configs/03_native_lr.yaml` |
| screen SR models cheaply (`fixed_recognizer`) | `configs/04_fixed_recognizer.yaml` |
| set my own train/val/test ratios | `configs/05_split_ratios.yaml` |
| use my own PyTorch SR class | `configs/06_sr_module.yaml`, `sr_models/tiny_espcn.py` |
| use SR images made by another tool | `configs/07_sr_images.yaml`, `sr_images/make_sr_images.py` |
| try face identity recognition | `configs/08_faces_lfw.yaml` (read the privacy note) |
| convert my dataset to `images/` + `labels.csv` | `dataset_conversion/` |

## Sample data

The samples are small subsets of real public datasets. They are rebuilt from your own downloads,
so that SR4Rec never redistributes images whose license does not allow it:

```bash
python scripts/make_examples.py --pets <path_to_oxford_iiit_pet>   # data/pets_mini   (~2 min)
python scripts/make_examples.py --earvn <path_to_EarVN1.0>         # data/earvn_mini
python scripts/make_examples.py --lfw <path_to_lfw>                # data/lfw_mini (faces)
python examples/sr_models/train_tiny_espcn.py                      # example SR weights (~3 min)
```

Each subset folder has a README with its source, license and citation.

## Dataset conversion scripts

| Script | Input layout |
|---|---|
| `from_folder_per_label.py` | `images/<label>/*.jpg` |
| `from_filename_pattern.py` | label inside the file name, for example `017_s1_03.jpg` |
| `from_annotation_csv.py` | an annotation CSV with other column names and split values |
