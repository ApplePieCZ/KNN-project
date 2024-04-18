# KNN project - Diffusion model
# Script for better training launching
# Created by Lukas Marek
# 18.4.2024

from ddm import *


def create_parser():
    parser = argparse.ArgumentParser(description="Parser for training script")

    parser.add_argument("dataset_path", type=str, help="Path to the data")
    parser.add_argument("epochs", type=int, help="Final epoch for training")
    parser.add_argument("run_name", type=str, help="Name of the training run")
    parser.add_argument("--continue_training", nargs=2, metavar=("epoch", "checkpoint_file"),
                        help="Continue training from a checkpoint")
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=6)
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA for GPU acceleration", default=True)

    return parser


def check_model():
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file {checkpoint_file} does not exist.")
        exit(-1)
    checkpoint = checkpoint_file.split("/")[-1]
    ending = checkpoint.split(".")[-1]

    if ending != "pt":
        print(f"Checkpoint file {checkpoint_file} is in incorrect format.")
        exit(-1)


if __name__ == '__main__':
    argument_parser = create_parser()
    arguments = argument_parser.parse_args()

    if arguments.continue_training:
        continue_epoch, checkpoint_file = arguments.continue_training
        check_model()
        print(f"--- Continuing training from epoch {continue_epoch} with checkpoint file {checkpoint_file} ---")
    else:
        print("--- Starting fresh training ---")

    if not os.path.exists(arguments.dataset_path):
        print(f"Dataset folder {arguments.dataset_path} does not exist.")
        exit(-1)

    dataset = arguments.dataset_path.split("/")[-1]
    print("Dataset:", dataset)
    print("Epochs:", arguments.epochs)
    print("Name:", arguments.run_name)
    print("Batch size:", arguments.batch_size)
    print("CUDA Enabled:", arguments.cuda)
    print("-------------------------------")

    start_training(arguments)
