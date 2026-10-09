# Data and model licenses

The SR4Rec source code is MIT-licensed. Datasets, SR weights, backbone weights and recognizer
checkpoints keep their own terms, listed here. SR4Rec does not redistribute third-party images or
SR weights: sample subsets are rebuilt from your own downloads and weights are downloaded by you.
Check each official page for the current terms before use.

## Datasets

| Dataset | Used for | Terms (as stated by the source; verify) | Citation |
|---|---|---|---|
| Oxford-IIIT Pet | `pets_mini`, quickstart | CC BY-SA 4.0 | Parkhi et al., CVPR 2012 |
| EarVN1.0 | `earvn_mini`, `d1_earvn`, `d1_earvn_fixed` | all rights reserved; no commercial use or redistribution | Hoang, Data in Brief 2019 |
| LFW | `lfw_mini`, `d2_lfw` | no license stated on the official page; biometric data | Huang et al., UMass TR 07-49, 2007 |
| CUB-200-2011 | `d3_cub` | research use; images belong to their photographers (Flickr) | Wah et al., Caltech TR CNS-TR-2011-001 |
| TinyFace (documentation only) | none shipped | no terms stated; contact the authors | Cheng et al., ACCV 2018 |

Face images are biometric data: use them for research only and follow the data-protection rules
that apply to you. No face image is included in this repository.

## SR weights (validated)

| Weights | Licence of the release | Citation |
|---|---|---|
| Real-ESRGAN x4plus | BSD-3-Clause (Real-ESRGAN repository) | Wang et al., ICCVW 2021 |
| SwinIR-M x4 classical / real-world | Apache-2.0 (SwinIR repository) | Liang et al., ICCVW 2021 |
| SPAN x4 | see the SPAN repository | Wan et al., CVPRW 2024 |

## Backbone weights

ImageNet-pretrained weights are downloaded by timm from Hugging Face; each model card states its
license (ResNet-18, MobileNetV3-Small and ConvNeXt-Tiny are released under Apache-2.0 in timm).

## Recognizer checkpoints (published for `--eval-only`)

Checkpoints trained on a dataset inherit the terms of that dataset. They are published only for
datasets whose terms allow it; `scripts/checkpoints_manifest.json` lists what is available.

## Code adapted from other projects

`src/sr4rec/lr/resize.py` is an independent implementation of MATLAB `imresize` (bicubic); it is
validated against the BasicSR (Apache-2.0) port of the same routine, which is not included.
