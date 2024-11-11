import argparse
import logging
import warnings

from lib.trainers.base_trainer import TorchBase
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import DatasetTypes, ModelTypes, TrainingParameters, VQAParameters
from lib.utils import EXPERIMENT

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
        choices=[choice for choice in DatasetTypes],
        type=str,
        help="The dataset to use",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        required=True,
        help="The dataset to use",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Set the seed for reproducible results.")
    return parser


def evaluate_model(args):
    if isinstance(args.seed, int):
        EXPERIMENT.set_seed(args.seed).apply_seed()

    test_args = VQAParameters(split=args.test_split, is_testing=True, use_proportional_split=True)
    parameters = TrainingParameters(
        dataset_name=args.dataset,
        resume_checkpoint=True,
        model_name=args.model,
        is_trainable=False,
        train_args=None,
        val_args=None,
        test_args=test_args,
        resume_state=False,
        is_testing=True,
    )

    if args.model == ModelTypes.BLIP2Generator or args.model == ModelTypes.BLIP2FinetunedGenerator:
        module: TorchBase = GenerationTrainer(parameters)
    elif args.model == ModelTypes.BLIP2Classifier or args.model == ModelTypes.BLIP2FinetunedClassifier:
        module: TorchBase = ClassificationTrainer(parameters)

    history = module.test()
    del history


if __name__ == "__main__":
    # Disable warnings
    warnings.filterwarnings("ignore")

    parser = get_parser()
    args = parser.parse_args()

    evaluate_model(args)
