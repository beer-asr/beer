import numpy as np

def mini_batches(data, mini_batch_size, seed=None):
    rng = np.random.RandomState()
    if seed is not None:
        rng.seed(seed)
    indices = rng.choice(data.shape[0], size=data.shape[0], replace=False)
    splits = np.array_split(indices, data.shape[0] // mini_batch_size)
    for split in splits:
        yield data[split]
