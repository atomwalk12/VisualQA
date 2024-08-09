import argparse
import logging
import warnings


from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import ModelTypes, TrainingParameters, VQAParameters
from lib.trainers.base_trainer import TorchBase
from lib.utils import set_seed

logging.basicConfig(level=logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[choice for choice in ModelTypes],
        type=str,
        help="The model to evaluate",
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
        "--test-split",
        type=str,
        required=True,
        help="The dataset to use",
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="Set the seed for reproducible results."
    )
    return parser


def evaluate_model(args):
    test_args = VQAParameters(
        split=args.test_split, is_testing=True, use_stratified_split=True
    )
    parameters = TrainingParameters(
        resume=True,
        model_name=args.model,
        is_trainable=False,
        train_args=None,
        val_args=None,
        test_args=test_args,
    )

    if args.model == ModelTypes.BLIP2Generator:
        module: TorchBase = GenerationTrainer(parameters)
    elif (
        args.model == ModelTypes.BLIP2Classifier
        or args.model == ModelTypes.BLIP2BaseClassifier
    ):
        module: TorchBase = ClassificationTrainer(parameters)

    history = module.test()
    del history


if __name__ == "__main__":
    # Disable warnings
    warnings.filterwarnings("ignore")

    parser = get_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    evaluate_model(args)
