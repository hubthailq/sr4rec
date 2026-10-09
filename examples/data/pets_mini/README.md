# pets_mini (Oxford-IIIT Pet subset)

Closed-set recognition of 10 breeds (5 cats, 5 dogs), 30 images per breed, HR images
(short side 256 px, PNG), `labels.csv` with the official train/test split (`split` column; SR4Rec moves
10% of train to val). Used by examples 01, 02, 04-07 and by `sr4rec reproduce quickstart`.

Build it from your copy of the dataset (<https://www.robots.ox.ac.uk/~vgg/data/pets/>):

```bash
python scripts/make_examples.py --pets <path_to_oxford_iiit_pet>
```

The selection rule is documented in `scripts/make_examples.py`; `selection.csv` records the source
file and its SHA-256 for every image. Licence and attribution: see `LICENSE.txt` in this folder and
`DATA_AND_MODEL_LICENSES.md`. Cite: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
"Cats and Dogs", CVPR 2012.
