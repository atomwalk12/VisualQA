import os
from pathlib import Path


ROOT_DATA_DIR = "data/"


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


def is_dir(path):
    p = Path(path)
    return p.is_dir()


def get_make_complete_path(file_name, dataset_name):
    Path(f"{ROOT_DATA_DIR}/{dataset_name}").mkdir(parents=True, exist_ok=True)

    out = f"{ROOT_DATA_DIR}/{dataset_name}/{file_name}.pkl"
    return os.path.abspath(out)


def existing_directory(arg: str):
    """Return `Path(arg)` but raise a `ValueError` if it does not refer to an
    existing directory."""
    path = Path(arg)
    if not path.is_dir():
        raise ValueError(f"{arg=}) is not a directory")

    return path


def likely_pickle_dir(dataset_name):
    return f"{ROOT_DATA_DIR}/{dataset_name}"


def format_time(seconds):
    """
    Format time in seconds to hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int((seconds % 3600) % 60)
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
