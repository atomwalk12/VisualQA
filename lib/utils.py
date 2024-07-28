import os
from pathlib import Path


def parse_split_slicer(split: str):
    if "[" in split:
        split, count = split.split("[:")
        count = int(count[:-1])
    else:
        count = None

    return split, count


def is_dir(path):
    p = Path(path)
    return p.is_dir()


def get_complete_path(out, opt_name):
    if os.path.isdir(out):
        out = f"{out}/{opt_name}.pkl"
    return os.path.abspath(out)
