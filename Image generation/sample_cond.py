# KNN project - Diffusion model
# Script for better conditional sampling launching
# Created by Lukas Marek
# 24.4.2024

import os
import argparse
from ddm import sample_conditional, inpaint_conditional


def create_parser():
    parser = argparse.ArgumentParser(description="Parser for training script")

    parser.add_argument("path", type=str, help="Path to the saved model dictionary (.pt)")
    parser.add_argument("classes", type=int, help="NUmber of classes")
    parser.add_argument("--images", nargs=2, type=int, help="Number of images to sample")
    parser.add_argument("--image_size", type=int, help="Resolution of images", default=32)
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
        arguments.device = "cuda"
    else:
        arguments.device = "cpu"

    if not arguments.inpainting:
        print("<--- Sampling images --->")
        print("Model:", model)
        if arguments.images:
            print("Number of images:", arguments.images[0])
            print("Selected class:", arguments.images[1])
        else:
            print("Number of images:", arguments.classes)
            print("Selected class: One from each")
        print("Image size:", arguments.image_size)
        print("<----------------------->")
        sample_conditional(arguments)
    else:
        print("<--- Inpainting images --->")
        print("Model:", model)
        if arguments.images:
            print("Number of images:", arguments.images[0])
            print("Selected class:", arguments.images[1])
        else:
            print("Number of images:", arguments.classes)
            print("Selected class: One from each")
        print("Image size:", arguments.image_size)
        print("Image for inpainting:", arguments.inpainting[0])
        print("Mask for inpainting:", arguments.inpainting[1])
        print("<------------------------->")
        inpaint_conditional(arguments)
