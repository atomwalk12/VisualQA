import logging
from lib.representations import DatasetFactory
from lib.utils import existing_directory
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "blip2",
        ],
        type=str,
        help="The type of model to finetune on",
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
        "--task",
        choices=[
            "process_data",
            "finetune",
        ],
        type=str,
        help="Specific task - preprocessing or training",
    )
    parser.add_argument(
        "--output-dir",
        type=existing_directory,
        help="Path to directory where intermediate data will be pickled and stored.",
        default="data/out/",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=existing_directory,
        help="Path to directory where models will be stored.",
        default="data/models/",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        help="Number of training epochs.",
        default=5,
    )

    return parser


def main(args: argparse.Namespace):
    if args.task == 'process_data':
        train_args = {'split': 'train', 'classify': False, 'load_raw': True}
        val_args = {'split': 'val', 'classify': False, 'load_raw': True}
        train_ds, val_ds = DatasetFactory.create_dataset(args.dataset, train_args, val_args)
        train_ds.save(args.output_dir)
        val_ds.save(args.output_dir)

    if args.task == 'finetune':
        pass


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
