# KNN project - Diffusion model
# Script with Diffusion module, training, sampling and inpainting function
# Modified by Lukas Marek and Tomas Krsicka
# 24.4.2024

import os
import torch
import torch.nn as nn
import torchvision
from torch import optim
from tqdm import tqdm
from plot import save_images, plot_images, get_data, setup_logging
from model import UNet
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
        Function prepares array of betas for all noise steps
        :return: Tensor of 1D array
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Function takes given image (x) and applies noise equivalent to t steps
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
        Function takes given image (x) and applies noise equivalent to one step so input should be x(t-1)
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
        Function generates random numbers from 1 to noise_steps for each image to be sampled n
        :param n: Number of images to be sampled
        :return: Tensor with array of random numbers
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        Function samples n images with chosen model, sampling is completely unconditional
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

    def inpainting(self, model, n, image_path, mask_path):
        """
        Function inpaints selected image in specific mask
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


def train(args):
    """
    Function trains model on selected dataset and saves learned model every epoch
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
            t = diffusion.sample_time_steps(images.shape[0]).to(args.devicev)
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


def sample(model_dict, n, device, save, image_size):
    """
    Function prepares all necessities for sampling, launch sampling and plots results
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


def inpaint(model_dict, n, device, save, image_size, image, mask):
    """
    Function prepares all necessities for inpainting, launch it and plots results
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


def start_training(arguments):
    """
    Function sets few additional arguments for training and launches traning loop
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
