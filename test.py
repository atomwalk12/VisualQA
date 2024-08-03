import argparse
import logging
import os
import warnings

import numpy as np
import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.blip_trainer import TorchBase
from lib.lightning_trainer import BLIP2PLModule
from lib.representations import DatasetFactory, ModelFactory
from lib.utils import set_seed
from lib.visualization import show_images_with_captions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["blip2"],
        type=str,
        help="The model name to evaluate",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "easy-vqa",
        ],
        type=str,
        help="The dataset to use",
    )
    parser.add_argument(
        "--classification",
        type=argparse.BooleanOptionalAction,
        help="Whether to treat the problem as a classification problem",
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="Set the seed for reproducible results."
    )
    return parser


def evaluate_model(args):
    test_args = {"split": "test", "classify": args.classification, "load": False}

    logger.info("Evaluating using manual loop")

    # Create simple fine tuning loop module
    module: TorchBase = TorchBase.prepare_module_for_testing(
        model_name=args.model, ds_name=args.dataset, test_args=test_args
    )

    module.test()


if __name__ == "__main__":
    # Disable warnings
    warnings.filterwarnings("ignore")

    parser = get_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    evaluate_model(args)

