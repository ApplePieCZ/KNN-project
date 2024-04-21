# KNN project - Diffusion model
# Script with Diffusion module and training loop
# Modified by Lukas Marek and Tomas Krsicka
# 18.4.2024

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
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        random_noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * random_noise, random_noise

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), total=999, colour="green"):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
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

        image_tensor = transforms(image).unsqueeze(0).to("cuda")
        mask_tensor = transform(mask).unsqueeze(0).to("cuda")
        mask_tensor = (1 - mask_tensor)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(2, self.noise_steps)), total=999, colour="green"):
                noised = self.noise_images(image_tensor, torch.tensor([i]))[0]
                x = x * mask_tensor + noised.to("cuda") * (1 - mask_tensor)

                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                predicted_noise = model(x, t)

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(image_size=args.image_size).to(device)
    starting_epoch = 0
    if args.training_continue:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)
        starting_epoch = int(args.epoch_continue)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=args.image_size, device=device)

    for epoch in range(starting_epoch, args.epochs, 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", colour="green")
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        if epoch % 25 == 0 and epoch != 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"epoch{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint.pt"))


def sample(checkpoint, n, device, save, image_size):
    model = UNet(image_size=image_size).to(device)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=image_size, device=device)
    sampled_images = diffusion.sample(model, n)
    if save != "":
        save_images(sampled_images, save)
    plot_images(sampled_images)


def inpaint(checkpoint, n, device, save, image_size, image, mask):
    model = UNet(image_size=image_size).to(device)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=image_size, device=device)
    inpainted_images = diffusion.inpainting(model, n, image, mask)
    if save != "":
        save_images(inpainted_images, save)
    plot_images(inpainted_images)


def start_training(arguments):
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
