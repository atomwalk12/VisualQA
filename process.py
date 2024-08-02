import argparse
import logging

from lib.representations import DatasetFactory
from lib.utils import existing_directory

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

    parser.add_argument(
        "--train",
        type=str,
        help="Number of training examples to generate.",
        default="train",
    )

    parser.add_argument(
        "--val",
        type=str,
        help="Number of validation examples to generate.",
        default="val",
    )

    return parser


def main(args: argparse.Namespace):
    # Whether to treat the problem as a classification problem
    # Vilt treats the problem as a classification problem while
    # Blip as free range generation.
    classification = True if args.model == "vilt" else False
    train_args = {"split": args.train, "classify": classification}
    val_args = {"split": args.val, "classify": classification}

    # Load the two datasets and prepare for training
    train_ds, val_ds = DatasetFactory.create_dataset(args.dataset, train_args, val_args)

    # Finally, save them to disk
    train_ds.save()
    val_ds.save()


if __name__ == "__main__":
    # TODO[R]
    # warnings.filterwarnings("ignore")

    parser = get_parser()
    args = parser.parse_args()

    main(args)
