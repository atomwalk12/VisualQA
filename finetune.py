import argparse
import gc
import logging
import warnings

from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import SAVE_PATHS, ModelTypes, TrainingParameters, VQAParameters
from lib.utils import set_seed

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
        "--train-split",
        type=str,
        help="Number of training examples to generate.",
        default="train",
    )

    # Same semnatics as in --train
    parser.add_argument(
        "--val-split",
        type=str,
        help="Number of validation examples to generate.",
        default="val",
    )

    # Whether to use the lightning trainer or not
    # TODO remove this part
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
        "--resume-training",
        help="Whether to continue training from checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def main(args: argparse.Namespace):
    if isinstance(args.seed, int):
        set_seed(args.seed)

    logger.info("Fine tuning using Torch Trainer")
    train_args = VQAParameters(split=args.train_split, use_stratified_split=True)
    val_args = VQAParameters(split=args.val_split, use_stratified_split=True)
    parameters = TrainingParameters(
        resume=args.resume_training,
        model_name=args.model,
        is_trainable=True,
        train_args=train_args,
        val_args=val_args,
        test_args=None,
    )

    if args.model == ModelTypes.BLIP2Generator:
        module = GenerationTrainer(parameters)
    elif (
        args.model == ModelTypes.BLIP2Classifier
        or args.model == ModelTypes.BLIP2BaseClassifier
    ):
        module = ClassificationTrainer(parameters)

    model, history = module.finetune()

    del model, history, module
    _ = gc.collect()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SAVE_PATHS.make_dirs()

    parser = get_parser()
    args = parser.parse_args()

    main(args)
