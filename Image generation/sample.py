# KNN project - Diffusion model
# Script for better sampling launching
# Created by Lukas Marek
# 18.4.2024

import os
import argparse
from ddm import sample, inpaint


def create_parser():
    parser = argparse.ArgumentParser(description="Parser for training script")

    parser.add_argument("path", type=str, help="Path to the saved model dictionary (.pt)")
    parser.add_argument("--images", type=int, help="Number of images to sample", default=8)
    parser.add_argument("--image_size", type=int, help="Resolution of images", default=64)
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA for GPU acceleration", default=False)
    parser.add_argument("--save", type=str, help="Save generated images with name", default="")
    parser.add_argument("--inpainting", nargs=2, type=str, help="Image and mask for inpainting")

    return parser


if __name__ == '__main__':
    argument_parser = create_parser()
    arguments = argument_parser.parse_args()

    if not os.path.exists(arguments.path):
        print(f"Checkpoint file {arguments.path} does not exist.")
        exit(-1)

    model = arguments.path.split("/")[-1]

    if arguments.cuda:
        device = "cuda"
    else:
        device = "cpu"

    if not arguments.inpainting:
        print("--- Sampling images ---")
        print("Model:", model)
        print("Number of images:", arguments.images)
        print("Image size:", arguments.image_size)
        sample(arguments.path, arguments.images, device, arguments.save, arguments.image_size)
    else:
        print("--- Inpainting images ---")
        print("Model:", model)
        print("Number of images:", arguments.images)
        print("Image size:", arguments.image_size)
        print("Image for inpainting:", arguments.inpainting[0])
        print("Mask for inpainting:", arguments.inpainting[1])
        inpaint(arguments.path, arguments.images, device, arguments.save, arguments.image_size,
                arguments.inpainting[0], arguments.inpainting[1])
