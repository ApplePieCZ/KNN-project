# KNN project - Diffusion model
# Script for better sampling launching
# Created by Lukas Marek
# 18.4.2024

from ddm import *


def create_parser():
    parser = argparse.ArgumentParser(description="Parser for training script")

    parser.add_argument("path", type=str, help="Path to the saved model dictionary (.pt)")
    parser.add_argument("images", type=int, help="Number of images to sample")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA for GPU acceleration", default=True)
    parser.add_argument("--save", type=str, help="Save generated images with name")

    return parser


if __name__ == '__main__':
    argument_parser = create_parser()
    arguments = argument_parser.parse_args()

    if not os.path.exists(arguments.path):
        print(f"Checkpoint file {arguments.path} does not exist.")
        exit(-1)

    model = arguments.path.split("/")[-1]
    print("--- Sampling images ---")
    print("Model:", model)
    print("Number of images:", arguments.images)

    if arguments.cuda:
        device = "cuda"
    else:
        device = "cpu"

    if arguments.save:
        save = arguments.save
    else:
        save = ""

    sample(arguments.path, arguments.images, device, save)
