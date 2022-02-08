# Google Landmark Recognition / Retrieval 2021 1st place

This repository contains code used achieve 1st place in the Google Landmark Recognition/ Retrieval 2021 competition which was hosted on kaggle (https://www.kaggle.com/c/landmark-recognition-2021 , https://www.kaggle.com/c/landmark-retrieval-2021)


## Models

To derive the solution the following models were trained using 8xV100 NVIDIA GPU with distributed data parallel (DDP). The current repository only consits of dataset and model architecture as well as hyperparameters in form of config files, but lacks precise training and inference routine. 

| model                         | image size | stride | data   | private score recognition | public score recognition | private score retrieval | public score retrieval |
|-------------------------------|------------|--------|--------|---------------------------|--------------------------|-------------------------|------------------------|
| DOLG-EfficientNet-B5          | 768        | 2      | GLDv2x | 0.476                     | 0.497                    | 0.478                   | 0.464                  |
| DOLG-EfficientNet-B6          | 768        | 2      | GLDv2x | 0.476                     | 0.479                    | 0.474                   | 0.454                  |
| DOLG-EfficientNet-B7          | 448        | 1      | GLDv2x | 0.465                     | 0.484                    | 0.470                   | 0.458                  |
| EfficientNet-B3-Swin-Base-224 | 896        | 2      | GLDv2x | 0.462                     | 0.487                    | 0.481                   | 0.454                  |
| EfficientNet-B5-Swin-Base-224 | 448        | 1      | GLDv2x | 0.462                     | 0.482                    | 0.476                   | 0.443                  |
| EfficientNet-B6-Swin-Base-384 | 384        | 1      | GLDv2x | 0.467                     | 0.492                    | 0.487                   | 0.462                  |
| EfficientNet-B3               | 768        | 2      | GLDv2  | 0.463                     | 0.487                    |                         |                        |
| EfficientNet-B6               | 512        | 2      | GLDv2  | 0.470                     | 0.484                    | 0.454                   | 0.441                  |
| EfficientNet-B5               | 704        | 2      | GLDv2x |                           |                          | 0.459                   | 0.428                  |
| Ensemble Recognition          |            |        |        | 0.513                     | 0.534                    |                         |                        |
| Ensemble Retrieval            |            |        |        |                           |                          | 0.537                   | 0.518                  |


### DOLG models

#### DOLG-EfficientNet-B5
#### DOLG-EfficientNet-B6
#### DOLG-EfficientNet-B7

### Hybrid-Swin-Transformers
#### EfficientNet-B3-Swin-Base-224
#### EfficientNet-B5-Swin-Base-224
#### EfficientNet-B6-Swin-Base-384

### Last years solutions
#### EfficientNet-B3 & EfficientNet-B6
refer to https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution

#### EfficientNet-B5
refer to https://github.com/bestfitting/instance_level_recognition


## Paper

The solution is summarized in the paper `Efficient large-scale image retrieval with deep feature orthogonality and Hybrid-Swin-Transformers` which is available under https://arxiv.org/abs/2110.03786

## Citing

### BibTeX

```bibtex
@misc{henkel2021efficient,
      title={Efficient large-scale image retrieval with deep feature orthogonality and Hybrid-Swin-Transformers}, 
      author={Christof Henkel},
      year={2021},
      eprint={2110.03786},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## ToDos

include training/ inference script
