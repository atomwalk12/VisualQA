import logging
from lib.lightning_trainer import LightningFineTune
from lib.representations import DatasetFactory
from lib.types import LightningConfig
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
            "process-data",
            "fine-tune",
        ],
        type=str,
        help="Specific task - preprocessing or training",
    )
    parser.add_argument(
        "--output-dir",
        type=existing_directory,
        help="Path to directory where intermediate data will be pickled and stored.",
        default="data/",
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
    load_raw = True if args.task == "process-data" else False
    train_args = {"split": "train", "classify": False, "load_raw": load_raw}
    val_args = {"split": "val", "classify": False, "load_raw": load_raw}

    train_ds, val_ds = DatasetFactory.create_dataset(args.dataset, train_args, val_args)

    if args.task == "process-data":
        train_ds.initialize_for_training()
        val_ds.initialize_for_training()
        train_ds.save(args.output_dir)
        val_ds.save(args.output_dir)

    if args.task == "fine-tune":
        pickle_dir = args.output_dir
        train_ds = train_ds.load(pickle_dir)
        val_ds = val_ds.load(pickle_dir)

        module = LightningFineTune.create_module(
            args.model, train_ds=train_ds, val_ds=val_ds
        )

        config = LightningConfig()

        trainer = LightningFineTune(config)
        trainer.finetune(module)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
