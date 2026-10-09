# lfw_mini (Labeled Faces in the Wild subset)

Face identity recognition in synthetic mode: the 10 people with the most images in LFW, 30 images
each (250 x 250 px), with a fixed split in `labels.csv` (21 train / 3 val / 6 test per person).
Used by example 08.

**No face image is stored in the repository.** Face images are biometric data of real people:
follow the terms of the original dataset and the data-protection rules that apply where you work,
and use this example for research only. Do not use SR4Rec to identify people outside research.

Download LFW from Kaggle (<https://www.kaggle.com/datasets/atulanandjha/lfwpeople>) and build the
subset:

```bash
python scripts/make_examples.py --lfw <path_to_lfw>
```

Cite: G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller, "Labeled Faces in the Wild: A Database for
Studying Face Recognition in Unconstrained Environments", University of Massachusetts, Amherst,
Technical Report 07-49, 2007.
