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

from lib.experiments import Reproducible

EXPERIMENT: Reproducible = Reproducible()
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
