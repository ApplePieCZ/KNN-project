# KNN-project
Project about image generating and inpainting using diffusion models.

## Image generation

Image generation is achieved using Denoising Diffusion model. Unconditional model was trained on Landscape dataset and conditional on CIFAR10 dataset. Solution is based on
[paper](https://arxiv.org/pdf/2006.11239.pdf) and [baseline solution](https://github.com/dome272/Diffusion-Models-pytorch). Improvements to make:
  - [x] More user-friendly to use
  - [ ] Simple GUI
  - [x] Add option to change image resolution
  - [x] Add option to continue training from checkpoint
  - [x] Unconditional inpainting
  - [ ] Conditional inpainting

### Setup

Training can be launched with following command:

    $ python3 train.py <path to dataset> <epochs> <name of run> --continue_training <starting epoch> <checkpoint file> --batch_size <images in batch> --cuda

Sampling with following command:

    $ python3 sample.py <path to model> <number of images> --cuda --save <name of saved images>

- Cuda argument sets device to cuda for GPU acceleration, default is cpu

### Results

## Inpainting

We implemented inpainting post condition inpainting solution based on this 
[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Corneanu_LatentPaint_Image_Inpainting_in_Latent_Space_With_Diffusion_Models_WACV_2024_paper.pdf).

Inpainting was implemented on:
- [x] Our Diffusion model
- [x] [DiT](https://github.com/facebookresearch/DiT)
- [ ] Stable diffusion

Inpainting on our model can be run with following command:

    $ python3 sample.py <path to model> <number of images> --cuda --save <name of saved images> --inpainting <path to image> <path to mask>

Mask doesn't have to be the same size as image, but it should have the same aspect ratio for it to work properly.

### Results