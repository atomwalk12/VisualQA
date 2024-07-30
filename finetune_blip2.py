import argparse
import logging

from lib.classic_trainer import SimpleFinetuneLoop
from lib.lightning_trainer import LightningFineTune
from lib.representations import DatasetFactory
from lib.types import LightningConfig
from lib.utils import existing_directory

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
        "--data-dir",
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

    parser.add_argument(
        "--limit-train-batches",
        type=float,
        help="Number of batches to train for.",
        default=1.0,
    )

    parser.add_argument(
        "--limit-val-batches",
        type=float,
        help="Number of batches to perform validation for.",
        default=1.0,
    )

    parser.add_argument("--lightning", action=argparse.BooleanOptionalAction)

    return parser


def main(args: argparse.Namespace):
    if args.task == "process-data":
        # Training class parameters
        train_args = {"split": args.train, "classify": False, "load_raw": True}
        val_args = {"split": args.val, "classify": False, "load_raw": True}

        # Load the two datasets and prepare for training
        train_ds, val_ds = DatasetFactory.create_dataset(
            args.dataset, train_args, val_args
        )
        train_ds.initialize_for_training()
        val_ds.initialize_for_training()

        # Finally, save them to disk
        train_ds.save(args.data_dir)
        val_ds.save(args.data_dir)

    if args.task == "fine-tune":
        train_args = {"split": args.train, "classify": False, "load_raw": False}
        val_args = {"split": args.val, "classify": False, "load_raw": False}
        if args.lightning:
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
            module = SimpleFinetuneLoop.create_module(
                model_name=args.model,
                ds_name=args.dataset,
                train_args=train_args,
                val_args=val_args,
            )

            module.finetune()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
