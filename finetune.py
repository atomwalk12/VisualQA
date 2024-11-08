import argparse
import gc
import logging
import warnings

import sys

from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import (
    SAVE_PATHS,
    ModelTypes,
    Suffix,
    TrainingParameters,
    VQAParameters,
    DatasetTypes,
)
from lib.utils import EXPERIMENT




logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[choice for choice in ModelTypes],
        type=str,
        help="The model type to finetune",
    )
    parser.add_argument(
        "--dataset",
        choices=[choice for choice in DatasetTypes],
        type=str,
        default="easy-vqa",
        help="The dataset to use",
    )

    # Use train[:count] to retrieve a specific number of items.
    # If train is used, it will fetch the entire dataset.
    parser.add_argument(
        "--train-split",
        type=str,
        help="Number of training examples to generate.",
        default=Suffix.Train,
    )

    # Same semnatics as in --train
    parser.add_argument(
        "--val-split",
        type=str,
        help="Number of validation examples to generate.",
        default=Suffix.Val,
    )

    # Whether to use the lightning trainer or not

    parser.add_argument(
        "--seed",
        type=int,
        help="Whether to make the results reproducible by setting a given seed.",
        default=2024,
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        help="The number of epochs to train for.",
        default=5000,
    )

    parser.add_argument(
        "--resume-training",
        help="Whether to continue training from checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    

    return parser


def main(args: argparse.Namespace):
    if isinstance(args.seed, int):
        EXPERIMENT.set_seed(args.seed).apply_seed()

    logger.info("Fine tuning using Torch Trainer")
    train_args = VQAParameters(split=args.train_split, use_proportional_split=True, multi_class_classifier=True)
    val_args = VQAParameters(split=args.val_split, use_proportional_split=True, multi_class_classifier=True)
    parameters = TrainingParameters(
        num_epochs=args.num_epochs,
        resume_checkpoint=args.resume_training,
        model_name=args.model,
        dataset_name=args.dataset,
        is_trainable=True,
        train_args=train_args,
        val_args=val_args,
        test_args=None,
        resume_state=False,
        scheduler_name="CosineAnnealingWarmRestarts" if args.model == ModelTypes.BLIP2Generator else "CosineAnnealingLR"
    )

    if args.model == ModelTypes.BLIP2Generator or args.model == ModelTypes.BLIP2FinetunedGenerator:
        module = GenerationTrainer(parameters)
    elif args.model == ModelTypes.BLIP2Classifier or args.model == ModelTypes.BLIP2FinetunedClassifier:
        module = ClassificationTrainer(parameters)

    model, history = module.finetune()

    del model, history, module
    _ = gc.collect()


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    SAVE_PATHS.make_dirs()

    parser = get_parser()
    args = parser.parse_args()

    main(args)
