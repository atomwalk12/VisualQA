# @title Setup paths

# Load the embeddings

import argparse
import logging

from lib.trainers.visualizer import Visualizer
from lib.types import SAVE_PATHS, DatasetTypes, ModelTypes, State, TrainingParameters, VQAParameters
from lib.utils import EXPERIMENT

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
        default="daquar",
        help="The dataset to use",
    )

    parser.add_argument(
        "--test-split",
        type=str,
        required=True,
        help="The dataset to use",
    )
    return parser


def main(args):
    train_args = VQAParameters(split=args.test_split, use_stratified_split=True)
    EXPERIMENT.set_seed(2024).apply_seed()
    state = State()
    root = (
        SAVE_PATHS.BLIP2_Generator_EasyVQA
        if args.dataset == DatasetTypes.EASY_VQA
        else SAVE_PATHS.BLIP2_Generator_DAQUAR
    )

    parameters = TrainingParameters(
        dataset_name=DatasetTypes.DAQUAR,
        resume_checkpoint=True,
        resume_state=False,
        model_name=ModelTypes.BLIP2Generator,
        is_trainable=False,
        train_args=train_args,
        val_args=None,
        test_args=None,
        use_wandb=False,
    )

    visualizer = Visualizer(parameters)

    try:
        history = state.load_state(root, args.dataset, "embeddings.pkl")
    except Exception:
        history = visualizer.generate_embeddings()

    visualizer.display(history, num_samples=100, filename="result_train[:1000].pdf")


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    SAVE_PATHS.make_dirs()

    parser = get_parser()
    args = parser.parse_args()

    main(args)
