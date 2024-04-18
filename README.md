# KNN-project
Project about image generating and inpainting using diffusion models.

## Sections:
Project is divided into 2 parts:

1. Image generation
2. Inpainting of images

### Image generation

Image generation is achieved using Denoising Diffusion model. Unconditional on Landscape dataset and conditional on CIFAR10 dataset. Solution is based on
[paper](https://arxiv.org/pdf/2006.11239.pdf) and [baseline solution](https://github.com/dome272/Diffusion-Models-pytorch). Improvements to make:
  - [x] More user-friendly to use
  - [ ] Automatically find best batch-size for training
  - [ ] Bigger image size 128x128
  - [ ] Image size 256x256 on MNIST dataset
  - [x] Add option to continue training from checkpoint

Training can be launched with following command:

    $ python3 train.py <path to dataset> <epochs> <name of run> --continue_training <starting epoch> <checkpoint file> --batch_size <images in batch> --cuda

Sampling with following command:

    $ python3 sample.py <path to model> <number of images> --cuda --save <name of saved images>

- cuda argument sets device to cuda for GPU acceleration, default is true

### Inpainting

