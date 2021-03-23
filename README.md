# Multi-speaker DGP
This repository provides official implementation of deep Gaussian process (DGP)-based multi-speaker speech synthesis with PyTorch.

Our paper: Deep Gaussian Process Based Multi-speaker Speech Synthesis with Latent Speaker Representation

## Test environment
This repository is tested in the following environment.
- Ubuntu 18.04
- NVIDIA GeForce RTX 2080 Ti
- Python 3.7.3
- CUDA 11.1
- cuDNN 8.1.1

## Setup
You can complete setup by simply executing `setup.sh`.
```
$ . ./setup.sh
```
*Please make sure that installed PyTorch is compatible with CUDA (see https://pytorch.org/ for more info).
Otherwise, CUDA error will occur during training.

## How to use
This repository is designed according to [Kaldi](https://github.com/kaldi-asr/kaldi)-style recipe.
To run the scripts, please follow the below instruction.
JVS corpus [Takamichi et al., 2020] can be downloaded from [here](https://drive.google.com/file/d/19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt/view). 
```
# Move to the recipe directory
$ cd egs/jvs

# Download the corpus to be used. The directory structure will be as follows:

├── conf/     # directory containing YAML format configuration files
├── jvs_ver1/ # downloaded data
├── local/    # directory containing corpus-dependent scripts
└── run.sh    # main scripts

# Run the recipe from scratch
$ ./run.sh

# Or you can run the recipe step by step
$ ./run.sh --stage 0 --stop-stage 0  # train/dev/eval split
$ ./run.sh --stage 1 --stop-stage 1  # preprocessing
$ ./run.sh --stage 2 --stop-stage 2  # train phoneme duration model
$ ./run.sh --stage 3 --stop-stage 3  # train acoustic model
$ ./run.sh --stage 4 --stop-stage 4  # decoding

# During stage 2 & 3, you can monitor logs using Tensorboard
# for example:
$ tensorboard --logdir exp/dgp
```

## How to customize
`conf/*.yaml` include all settings for data preparation, preprocessing, training, and decoding.
We have prepared two configuration files, `dgp.yaml` and `dgplvm.yaml`.
You can change experimental conditions by editing these files.
