import argparse
import gc
import logging
import os
import warnings

import numpy as np
import torch

from lib.blip_trainer import TorchBase
from lib.lightning_trainer import LightningFineTune
from lib.types import LightningConfig, TorchTrainerConfig
from lib.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["blip2", "vilt"],
        type=str,
        default="blip2",
        help="The type of model to finetune on",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "easy-vqa",
        ],
        type=str,
        default="easy-vqa",
        help="The dataset to use",
    )

    # Use train[:count] to retrieve a specific number of items.
    # If train is used, it will fetch the entire dataset.
    parser.add_argument(
        "--train",
        type=str,
        help="Number of training examples to generate.",
        default="train",
    )

    # Same semnatics as in --train
    parser.add_argument(
        "--val",
        type=str,
        help="Number of validation examples to generate.",
        default="val",
    )

    # Whether to use the lightning trainer or not
    parser.add_argument(
        "--use-lightning",
        action=argparse.BooleanOptionalAction,
        help="Whether to use the Pytorch Lightning library to fine-tune.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Whether to make the results reproducible by setting a given seed.",
        default=42,
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler to use",
        choices=["CosineAnnealingLR", "CosineAnnealingWarmRestarts"],
        default="CosineAnnealingLR",
    )

    parser.add_argument(
        "--generate",
        help="Whether to generate the dataset if it doesn't exist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def main(args: argparse.Namespace):
    if isinstance(args.seed, int):
        set_seed(args.seed)

    classification_task = True if args.model == "vilt" else False
    train_args = {
        "split": args.train,
        "classify": classification_task,
        "load": args.generate
    }
    val_args = {
        "split": args.val,
        "classify": classification_task,
        "load": args.generate,
    }

    if args.use_lightning:
        logger.info("Fine tuning using torch lightning")

        # Create the lighting module used for fine-tuning.
        module = LightningFineTune.create_module(
            model_name=args.model,
            ds_name=args.dataset,
            train_args=train_args,
            val_args=val_args,
        )

        # Lightning configuration file: contains batch_size, lr, etc.
        config = LightningConfig(
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )

        trainer = LightningFineTune(config)
        trainer.finetune(module)
    else:
        logger.info("Fine tuning using manual loop")

        hyperparameters = TorchTrainerConfig(scheduler_name=args.scheduler)

        # Create simple fine tuning loop module
        module = TorchBase.prepare_module_for_training(
            torch_config=hyperparameters,
            model_name=args.model,
            ds_name=args.dataset,
            train_args=train_args,
            val_args=val_args,
        )

        model, history = module.finetune()

        del model, history, module
        _ = gc.collect()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = get_parser()
    args = parser.parse_args()

    main(args)
