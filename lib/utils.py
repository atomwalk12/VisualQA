import logging
import os
from pathlib import Path

import numpy as np
import torch
from datetime import datetime
from transformers import PreTrainedModel
from enum import StrEnum
import numpy
import random

ROOT_DATA_DIR = "data/"
MODELS_DIR = f"{ROOT_DATA_DIR}/models/"



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_split_slicer(split: str):
    if "[" in split:
        split, slicer = split.split("[")
        slicer = slicer[:-1]  # Remove the trailing ']'
        if ":" in slicer:
            start, end = slicer.split(":")
            start = int(start) if start else None
            end = int(end) if end else None
        else:
            start = int(slicer)
            end = None
    else:
        start = None
        end = None

    return split, start, end


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def format_time(seconds):
    """
    Format time in seconds to hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int((seconds % 3600) % 60)
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"


def set_seed(seed):
    """Sets a seed for the run in order to make the results reproducible."""
    logger.info(f"Setting {seed=}")
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    # whether to use the torch autotuner and find the best algorithm for current hardware
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)



def seed_worker(seed_worker):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g