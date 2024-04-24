# KNN project - Diffusion model
# Script with Diffusion module, training, sampling and inpainting function
# Modified by Lukas Marek and Tomas Krsicka
# 24.4.2024

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from tqdm import tqdm
from plot import save_images, plot_images, get_data, setup_logging
from model import UNet, UNetConditional, EMA
from PIL import Image


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        Prepare array of betas for all noise steps
        :return: Tensor of 1D array
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Take given image (x) and apply noise equivalent to t number of steps
        :param x: Input image
        :param t: Timesteps
        :return: Noised image and noise that was used
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        random_noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * random_noise, random_noise

    def noise_step(self, x, t):
        """
        Take given image (x) and applies noise equivalent to one step
        and output is x(t)
        :param x: Image to be noised
        :param t: Timestep of noise that it should apply
        :return: Noised image
        """
        sqrt_beta = torch.sqrt(1. - self.beta[t])
        random_noise = torch.randn_like(x)
        return sqrt_beta * x + self.beta[t] * random_noise

    def sample_time_steps(self, n):
        """
        Generate random numbers from 1 to noise_steps for each image to be sampled
        :param n: Number of images to be sampled
        :return: Tensor with array of random numbers
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        Sample n images with chosen model, unconditional model is expected
        :param model: Model that will be used for sampling
        :param n: Number of images to be sampled
        :return: Returns sampled images as tensors
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=1000, colour="green"):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)\
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_conditional(self, model, n, labels, cfg_scale=3):
        """
        Sample n images with chosen model, conditional model is expected
        :param model: Model that will be used for sampling
        :param n: Number of images to be sampled
        :param labels: Number of labels
        :param cfg_scale: Sets how strong CFG is
        :return: Returns sampled images as tensors
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=1000, position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)\
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def inpainting(self, model, n, image_path, mask_path):
        """
        Inpaint selected image with specified mask
        :param model: Model used for sampling
        :param n: Number of inpainted images
        :param image_path: Path to image that will be inpainted
        :param mask_path: Path to inpainting mask
        :return: Inpainted images as tensors
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
        ])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image_tensor = transforms(image).unsqueeze(0).to(self.device)
        mask_tensor = transform(mask).unsqueeze(0).to(self.device)
        mask_tensor = (1 - mask_tensor)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=1000, colour="green"):
                noised = self.noise_images(image_tensor, torch.tensor([i]))[0]
                x = x * mask_tensor + noised.to(self.device) * (1 - mask_tensor)
                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                for _ in range(10):
                    predicted_noise = model(x, t)

                    if i > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                        + torch.sqrt(beta) * noise

                    x = self.noise_step(x, torch.tensor([t[0]]))

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def inpaint_conditional(self, model, n, labels, image_path, mask_path, cfg_scale=3):
        """
        Inpaint selected image with specified mask
        :param model: Model used for sampling
        :param n: Number of inpainted images
        :param labels: Number of labels
        :param image_path: Path to image that will be inpainted
        :param mask_path: Path to inpainting mask
        :param cfg_scale: Sets how strong CFG is
        :return: Inpainted images as tensors
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image_tensor = transforms(image).unsqueeze(0).to(self.device)
        mask_tensor = transform(mask).unsqueeze(0).to(self.device)
        mask_tensor = (1 - mask_tensor)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=1000, position=0):
                noised = self.noise_images(image_tensor, torch.tensor([i]))[0]
                x = x * mask_tensor + noised.to(self.device) * (1 - mask_tensor)

                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)\
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    """
    Train model on selected dataset and save learned model every epoch
    :param args: Input arguments for training (all are described in train.py)
    """
    setup_logging(args.run_name)
    dataloader = get_data(args)
    model = UNet(image_size=args.image_size).to(args.device)
    starting_epoch = 0

    if args.training_continue:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)
        starting_epoch = int(args.epoch_continue)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=args.image_size, device=args.device)

    for epoch in range(starting_epoch, starting_epoch + args.epochs, 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", colour="green")
        for i, (images, _) in enumerate(pbar):
            images = images.to(args.device)
            t = diffusion.sample_time_steps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        if epoch % args.save_epoch == 0 and epoch != 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"epoch{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint.pt"))


def train_conditional(args):
    """
    Train model on selected dataset and save learned model every epoch
    :param args: Input arguments for training (all are described in train.py)
    """
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNetConditional(num_classes=args.conditional, image_size=args.image_size).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    starting_epoch = 0

    if args.training_continue:
        ckpt = torch.load(args.checkpoint)
        ema_ckpt = torch.load(args.ema)
        model.load_state_dict(ckpt)
        ema_ckpt = torch.load(ema_ckpt)
        ema_model.load_state_dict(ema_ckpt)
        starting_epoch = int(args.epoch_continue)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=args.image_size, device=device)
    ema = EMA(0.995)

    for epoch in range(starting_epoch, starting_epoch + args.epochs, 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", colour="green")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

        if epoch % args.save_epoch == 0 and epoch != 0:
            labels = torch.arange(args.conditional).long().to(device)
            sampled_images = diffusion.sample_conditional(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample_conditional(ema_model, n=len(labels), labels=labels)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ckpt{epoch}_ema.pt"))
            # torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim{epoch}.pt"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))


def sample(model_dict, n, device, save, image_size):
    """
    Preparee all necessities for sampling, launch sampling and plot results
    :param model_dict: Model dictionary to be used for sampling
    :param n: Number of images for sampling
    :param device: Device where sampling will run (either CUDA or CPU)
    :param save: If images are meant to be saved (name of saved image)
    :param image_size: Size of final images, should be same as model was trained on
    """
    model = UNet(image_size=image_size).to(device)
    ckpt = torch.load(model_dict)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=image_size, device=device)
    sampled_images = diffusion.sample(model, n)
    if save != "":
        save_images(sampled_images, save)
    plot_images(sampled_images)


def sample_conditional(arguments):
    """
    Prepare all necessities for conditional sampling, launch it and plot results
    :param arguments: All necessary arguments are described in sample_cond.py
    """
    model = UNetConditional(num_classes=arguments.classes, image_size=arguments.image_size).to(arguments.device)
    ckpt = torch.load(arguments.path)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=arguments.image_size, device=arguments.device)
    if arguments.images:
        labels = torch.Tensor([arguments.images[1]] * arguments.images[0]).long().to(arguments.device)
        x = diffusion.sample_conditional(model, n=arguments.images[0], labels=labels, cfg_scale=3)
    else:
        labels = torch.arange(arguments.classes).long().to(arguments.device)
        x = diffusion.sample_conditional(model, n=len(labels), labels=labels, cfg_scale=3)
    if arguments.save != "":
        save_images(x, arguments.save)
    plot_images(x)


def inpaint(model_dict, n, device, save, image_size, image, mask):
    """
    Prepare all necessities for inpainting, launch it and plot results
    :param model_dict: Model dictionary to be used for sampling
    :param n: Number of images for sampling
    :param device: Device where sampling will run (either CUDA or CPU)
    :param save: If images are meant to be saved (name of saved image)
    :param image_size: Size of final images, should be same as model was trained on
    :param image: Image to be inpainted
    :param mask: Mask to cover desired area for inpainting
    """
    model = UNet(image_size=image_size).to(device)
    ckpt = torch.load(model_dict)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=image_size, device=device)
    inpainted_images = diffusion.inpainting(model, n, image, mask)
    if save != "":
        save_images(inpainted_images, save)
    plot_images(inpainted_images)


def inpaint_conditional(arguments):
    """
    Prepare all necessities for conditional inpainting, launch it and plot results
    :param arguments: All necessary arguments are described in sample_cond.py
    """
    model = UNetConditional(num_classes=arguments.classes, image_size=arguments.image_size).to(arguments.device)
    ckpt = torch.load(arguments.path)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=arguments.image_size, device=arguments.device)
    if arguments.images:
        labels = torch.Tensor([arguments.images[1]] * arguments.images[0]).long().to(arguments.device)
        x = diffusion.inpaint_conditional(model, n=arguments.images[0], labels=labels, image_path=arguments.inpainting[0],
                                          mask_path=arguments.inpainting[1], cfg_scale=3)
    else:
        labels = torch.arange(arguments.classes).long().to(arguments.device)
        x = diffusion.inpaint_conditional(model, n=len(labels), labels=labels, image_path=arguments.inpainting[0],
                                          mask_path=arguments.inpainting[1], cfg_scale=3)
    if arguments.save != "":
        save_images(x, arguments.save)
    plot_images(x)


def start_training(arguments):
    """
    Set few additional arguments for training and launch training loop
    :param arguments: Arguments for training (described in train.py)
    """
    if arguments.continue_training:
        arguments.training_continue = True
        continue_epoch, checkpoint_file = arguments.continue_training
        arguments.epoch_continue = continue_epoch
        arguments.checkpoint = checkpoint_file
    else:
        arguments.training_continue = False

    arguments.lr = 3e-4

    if arguments.cuda:
        arguments.device = "cuda"
    else:
        arguments.device = "cpu"

    train(arguments)


def start_training_conditional(arguments):
    """
    Set few additional arguments for conditional training and launch training loop
    :param arguments: Arguments for training (described in train.py)
    """
    if arguments.continue_training:
        arguments.training_continue = True
        continue_epoch, checkpoint_file = arguments.continue_training
        arguments.epoch_continue = continue_epoch
        arguments.checkpoint = checkpoint_file
        ema = checkpoint_file.split(".pt")[0]
        ema += "_ema.pt"
        arguments.ema = ema
    else:
        arguments.training_continue = False

    arguments.lr = 3e-4

    if arguments.cuda:
        arguments.device = "cuda"
    else:
        arguments.device = "cpu"

    train_conditional(arguments)
