import argparse
import os
import sys
from tools import console
from data_preparation.sampling import sampling
from inference.prediction import prediction

from train.train_contrastive import train_contrastive


def add_arguments(obj):
    obj.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration file 'config.yaml'",
    )
    obj.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Input path to load data.",
    )
    obj.add_argument(
        "-o",
        "--output_folder",
        default="results",
        type=str,
        help="Output path to write the results.",
    )


def main(args=None):

    parser = argparse.ArgumentParser(description="Plankton Fusion Classifier")
    subparsers = parser.add_subparsers()

    # sampling
    parser_sampling = subparsers.add_parser(
        "sampling",
        help="Prepare dataset ready for training.",
    )
    add_arguments(parser_sampling)
    parser_sampling.set_defaults(func=call_sampling)

    # training contrastive
    parser_training_contrastive = subparsers.add_parser(
        "training_contrastive", help="Train a contrastive architecture"
    )
    add_arguments(parser_training_contrastive)
    parser_training_contrastive.set_defaults(func=call_training_contrastive)

    # prediction
    parser_prediction = subparsers.add_parser(
        "prediction", help="Prediction from trained model"
    )
    add_arguments(parser_prediction)
    parser_prediction.set_defaults(func=call_prediction)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)
    args.func(args)


def call_sampling(args):
    sampling(args.configuration_file)


def call_training_contrastive(args):
    train_contrastive(args.configuration_file, args.input_folder, args.output_folder)


def call_prediction(args):
    prediction(args.configuration_file, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
