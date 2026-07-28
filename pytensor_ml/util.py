import numpy as np

from pytensor import config


class DataLoader:
    """
    Draw shuffled, fixed-size batches from a dataset, cycling indefinitely.

    Calling the loader returns the next ``(X_batch, y_batch)``. Batches are always ``batch_size`` rows: a
    pass that runs off the end of the data is topped up from a freshly shuffled order rather than coming up
    short, so a batch may straddle two epochs.

    Parameters
    ----------
    X : ndarray
        Features, indexed along the first axis.
    y : ndarray
        Targets, indexed along the first axis and aligned row-wise with ``X``.
    batch_size : int
        Rows per batch. Default 64.
    dtype : str, optional
        Dtype to cast ``X`` and ``y`` to. Defaults to ``floatX``.
    random_state : int or numpy Generator, optional
        Seed for the shuffling generator, for reproducible batch sequences.
    """

    def __init__(self, X, y, batch_size=64, dtype=None, random_state=None):
        if dtype is None:
            dtype = config.floatX
        self.rng = np.random.default_rng(random_state)

        self.X = X.astype(dtype)
        self.y = y.astype(dtype)
        self.n = X.shape[0]

        self.batch_size = batch_size
        self.indices = np.arange(self.n, dtype="int32")
        self.reset()

    def shuffle(self):
        self.rng.shuffle(self.indices)

    def move_cursor(self):
        start, stop = self.cursor, self.cursor + self.batch_size
        n_wraps, stop = divmod(stop, self.n)

        batch_slice = slice(start, None if n_wraps else stop)
        # Copy, don't view: the wrap branch below reshuffles self.indices in place, which would otherwise
        # rewrite the rows already selected here and silently drop some rows while repeating others.
        batch_indices = self.indices[batch_slice].copy()

        if n_wraps:
            shortfall = self.batch_size - batch_indices.shape[0]
            self.shuffle()
            self.epoch += 1
            batch_indices = np.r_[batch_indices, self.indices[:shortfall]]

        self.cursor = (self.cursor + self.batch_size) % self.n

        return batch_indices

    def reset(self):
        self.cursor = 0
        self.epoch = 0
        self.shuffle()

    def __call__(self):
        batch_indices = self.move_cursor()
        return self.X[batch_indices], self.y[batch_indices]
