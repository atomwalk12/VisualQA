def get_index(possible_idx, batch_size, ds):
    ds_size = len(ds)
    idx = batch_size + possible_idx if possible_idx + batch_size < ds_size else ds_size
    return idx
