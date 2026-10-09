# earvn_mini (EarVN1.0 subset)

Identity recognition from natural low-resolution ear images: 20 identities x 20 images at their
original size (many are smaller than 64 px), `labels.csv` with `path,label` only (SR4Rec creates the
split). Used by example 03 (`mode: native-lr`).

EarVN1.0 may not be redistributed without the permission of its authors, so no image is stored in
the repository. Download the dataset from its official page (Mendeley Data) and build the subset:

```bash
python scripts/make_examples.py --earvn <path_to_EarVN1.0>
```

Terms: see `LICENSE.txt` in this folder. Cite: V. T. Hoang, "EarVN1.0: A new large-scale ear images
dataset in the wild", Data in Brief, 2019.
