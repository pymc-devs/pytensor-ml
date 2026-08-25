Quickstart
==========

.. note::

   **WRITEME.** This page is a stub. Walk a new user end-to-end through
   building, training, evaluating, and saving a model. Cross-link to the
   :doc:`/examples/gallery` for the full notebooks.

The snippet below trains a small classifier on scikit-learn's digits dataset.

.. code-block:: python

    import numpy as np
    import pytensor

    pytensor.config.floatX = "float32"

    from sklearn.datasets import load_digits

    from pytensor_ml.activations import ReLU
    from pytensor_ml.layers import Input, Linear, Sequential
    from pytensor_ml.loss import CrossEntropy
    from pytensor_ml.model import Model
    from pytensor_ml.optim import adam, chain, clip_by_global_norm, cosine_schedule
    from pytensor_ml.util import DataLoader

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype("float32")
    y_onehot = np.eye(10, dtype="float32")[y]

    X_in = Input("X_in", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=128),
        ReLU(),
        Linear("logits", n_in=128, n_out=10),
    )
    model = Model(X_in, network(X_in)).initialize(seed=0)

    rule = chain(adam(learning_rate=cosine_schedule(1e-3, total_steps=500)), clip_by_global_norm(1.0))
    loss_fn = CrossEntropy(expect_onehot_labels=True, expect_logits=True, reduction="mean")
    step = model.compile_train(rule, loss_fn, ndim_out=2)

    loader = DataLoader(X, y_onehot, batch_size=64, random_state=0)
    for _ in range(500):
        loss_value = step(*loader())

    accuracy = (model.predict(X).argmax(axis=-1) == y).mean()

:meth:`~pytensor_ml.model.Model.compile_train` builds the loss against a
target placeholder, differentiates it, folds in any stateful layer updates
(batch norm running statistics, RNG advances, the training clock a schedule
reads), and compiles a one-step function.
:meth:`~pytensor_ml.model.Model.predict` compiles a separate inference pass,
with dropout removed and batch norm reading its running statistics.

A :class:`~pytensor_ml.model.Model` is a convenience, not a requirement:
:func:`pytensor_ml.optim.compile_train` trains any loss graph you hand it.
