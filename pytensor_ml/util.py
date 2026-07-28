import numpy as np

from pytensor import config


class DataLoader:
    def __init__(self, X, y, batch_size=64, dtype=None, random_state=None):
        if dtype is None:
            dtype = config.floatX
        self.rng = np.random.default_rng(random_state)

        self.X = X.astype(dtype)
        self.y = y.astype(dtype)
        self.n = X.shape[0]

        self.batch_size = batch_size
        self.cursor = 0
        self.indices = np.arange(len(X)).astype("int32")
        self.reset()

    def shuffle(self):
        self.rng.shuffle(self.indices)

    def move_cursor(self):
        start, stop = self.cursor, self.cursor + self.batch_size

        has_remainder, stop = divmod(stop, self.n)

        idx = slice(start, None if has_remainder > 0 else stop)
        indices = self.indices[idx]

        if has_remainder > 0:
            excess = self.batch_size - indices.shape[0]
            self.shuffle()
            self.epoch += 1
            indices = np.r_[indices, self.indices[:excess]]

        self.cursor = (self.cursor + self.batch_size) % self.n

        return indices

    def reset(self):
        self.cursor = 0
        self.epoch = 0
        self.shuffle()

    def __call__(self):
        idx = self.move_cursor()
        return self.X[idx], self.y[idx]
