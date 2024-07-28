def parse_split_slicer(split: str):
    if '[' in split:
        split, count = split.split("[:")
        count = int(count[:-1])
    else:
        count = None

    return split, count
