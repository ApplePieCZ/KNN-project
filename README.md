# KNN-project
Project about image generating and inpainting using diffusion models.

## Image generation

Image generation is achieved using Denoising Diffusion model. Unconditional model was trained on Landscape dataset and conditional on CIFAR10 dataset. Solution is based on
[paper](https://arxiv.org/pdf/2006.11239.pdf) and [baseline solution](https://github.com/dome272/Diffusion-Models-pytorch). Improvements to make:
  - [x] More user-friendly to use
  - [x] Add option to change image resolution
  - [x] Add option to continue training from checkpoint
  - [x] Unconditional inpainting
  - [x] Conditional inpainting
  - [x] Implement resampling

### Training
Training saves model every epoch as **checkpoint.pt**. It is also possible to save and sample each n-th epoch, results are saved as **sample{epoch}.jpg** and **checkpoint{epoch}.pt**.
All checkpoints are saved in corresponding folders - models and results under name of run.

Training can be launched with following command:

    $ python3 train.py <path to dataset> <epochs> <name of run> 

Possible additional arguments:
- --continue_training <starting epoch\> <checkpoint file\> - From which epoch you want to train (it only looks better) and checkpoint from which you want.
- --image_size <resolution in px\> - Resolution of images for training, must be powers of 2 (32, 64, 128...). **Default is 64**
- --batch_size <images in batch\> - Number of images in batch. For GPUs with more memory bigger batch is better. **Default is 6** (for 10GB and 64x64)
- --save_epoch <n\> - Each n-th epoch it will save checkpoints as mentioned previously. **Default is 25**
- --cuda - Use GPU instead CPU. **Default is CPU**. Only use when CUDA is available!

For training conditional model add following argument:
- --conditional <number of classes\>

### Sampling

Unconditional Sampling with following command:

    $ python3 sample.py <path to model> 

Possible additional arguments:
- --images <n\> - number of images to be sampled. **Default is 8**
- --image_size <resolution in px\> - Resolution of images for sampling, can be bigger size that what model was trained on, but results will be undefined. **Default is 64**
- --cuda - Use GPU instead CPU. **Default is CPU**. Only use when CUDA is available!
- --save <name.jpg\> - To save generated images to name.jpg.

Conditional Sampling with following command:

    $ python3 sample_cond.py <path to model> <number of classes>

Possible additional arguments:
- --images <n\> <c\> - number of **n** images to be sampled for class **c**.
- --image_size <resolution in px\> - Resolution of images for sampling, can be bigger size that what model was trained on, but results will be undefined. **Default is 32**
- --cuda - Use GPU instead CPU. **Default is CPU**. Only use when CUDA is available!
- --save <name.jpg\> - To save generated images to name.jpg.

### Results

## Inpainting

We implemented inpainting post condition inpainting solution with resampling based on this 
[paper](https://arxiv.org/pdf/2201.09865.pdf).

Inpainting with resampling was implemented on:
- [x] Our Denoising Diffusion model
- [x] [DiT](https://github.com/facebookresearch/DiT)

Inpainting on our model can be run with following command:

    $ python3 sample.py <path to model> --inpainting <path to image> <path to mask>

All other arguments are same as before.

For inpainting on conditional model:

    $ python3 sample_cond.py <path to model> --inpainting <path to image> <path to mask>

All other arguments are same as before.

Mask doesn't have to be the same size as image, but it should have the same aspect ratio for it to work properly.

### Results