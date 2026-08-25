Contributing
============

.. note::

   **WRITEME.** This page is a stub. Cover the branching and PR workflow,
   test conventions (PyTorch as the reference implementation, per-backend
   parametrization), and where new layers and optimizers should live.

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e ".[dev]"
    pre-commit install

Running tests
-------------

.. code-block:: bash

    pytest

Style and typing
----------------

Formatting and linting run through ``ruff`` under pre-commit, and ``mypy``
checks ``pytensor_ml/``. Both also run in CI, so a clean
``pre-commit run --all-files`` locally is the fastest way to keep a PR green.

Bug reports and feature requests belong in the
`issue tracker <https://github.com/pymc-devs/pytensor-ml/issues>`_.
