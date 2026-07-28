import numpy as np

from pytensor_ml.util import DataLoader


def make_data(n: int = 10, n_features: int = 3):
    """Row ``i`` of X starts with ``i * n_features``, and ``y[i] == i``, so a shuffle cannot hide a
    desynchronization between the two."""
    X = np.arange(n * n_features, dtype="float64").reshape(n, n_features)
    y = np.arange(n, dtype="float64")
    return X, y


def test_a_full_pass_visits_every_row_exactly_once():
    n, batch_size = 10, 5
    loader = DataLoader(*make_data(n=n), batch_size=batch_size, random_state=0)

    visited = np.concatenate([loader()[1] for _ in range(n // batch_size)])

    np.testing.assert_array_equal(np.sort(visited), np.arange(n))


def test_batches_are_full_size_when_n_is_not_divisible_by_batch_size():
    n, batch_size = 10, 4
    loader = DataLoader(*make_data(n=n), batch_size=batch_size, random_state=0)

    # Five batches walk the cursor through every offset it can hold (0, 4, 8, 2, 6) and back to 0, so this
    # covers every wrap alignment -- including the one landing exactly on the end of the data.
    for _ in range(5):
        X_batch, y_batch = loader()
        assert X_batch.shape[0] == batch_size
        assert y_batch.shape[0] == batch_size


def test_a_batch_wider_than_the_dataset_is_built_from_whole_passes():
    # One refilled pass supplies at most n rows, so a batch this wide needs several, one epoch apiece.
    n, batch_size = 3, 25
    loader = DataLoader(*make_data(n=n), batch_size=batch_size, random_state=0)

    _, y_batch = loader()

    assert y_batch.shape[0] == batch_size
    assert loader.epoch == batch_size // n

    # Whole passes rather than a repeated prefix, so no row is over-sampled relative to another.
    counts = np.bincount(y_batch.astype(int), minlength=n)
    assert counts.max() - counts.min() <= 1


def test_epoch_increments_once_per_pass():
    batches_per_pass = 2
    loader = DataLoader(*make_data(n=10), batch_size=5, random_state=0)
    assert loader.epoch == 0

    for expected_epoch in (1, 2, 3):
        for _ in range(batches_per_pass):
            loader()
        assert loader.epoch == expected_epoch


def test_rows_stay_aligned_between_X_and_y():
    n_features = 3
    loader = DataLoader(*make_data(n=10, n_features=n_features), batch_size=4, random_state=0)

    for _ in range(5):
        X_batch, y_batch = loader()
        np.testing.assert_array_equal(X_batch[:, 0], y_batch * n_features)


def test_equal_seeds_give_identical_batch_sequences():
    first = DataLoader(*make_data(), batch_size=4, random_state=42)
    second = DataLoader(*make_data(), batch_size=4, random_state=42)

    for _ in range(6):
        X_first, y_first = first()
        X_second, y_second = second()
        np.testing.assert_array_equal(X_first, X_second)
        np.testing.assert_array_equal(y_first, y_second)


def test_reset_restarts_the_current_pass():
    loader = DataLoader(*make_data(n=10), batch_size=5, random_state=0)
    loader()

    loader.reset()
    assert loader.epoch == 0

    # A rewound cursor cannot complete a pass on its first batch; a cursor left mid-pass would wrap here.
    loader()
    assert loader.epoch == 0

    loader()
    assert loader.epoch == 1
