# @title Setup paths

# Load the embeddings

import argparse
import logging

from lib.types import SAVE_PATHS, DatasetTypes, EvaluationMetrics, ModelTypes
from lib.utils import EXPERIMENT
from lib.visualization import check_label_distribution, show_umap_clustering

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
        "--split",
        type=str,
        required=True,
        help="The dataset to use",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=100,
        help="The number of samples to show in the graph",
    )
    parser.add_argument(
        "--metric",
        choices=[metric for metric in EvaluationMetrics],
    )
    return parser


def main(args):
    # Set the seed
    EXPERIMENT.set_seed(2024).apply_seed()

    if args.metric == EvaluationMetrics.UMAP:
        show_umap_clustering(args.split, args.dataset, num_samples=args.num_samples)
    elif args.metric == EvaluationMetrics.DATA_DISTRIBUTION:
        check_label_distribution(args.dataset)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    SAVE_PATHS.make_dirs()

    parser = get_parser()
    args = parser.parse_args()

    main(args)
