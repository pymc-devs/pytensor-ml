Installation
============

.. note::

   **WRITEME.** This page is a stub. Flesh out with per-backend install
   notes and any platform-specific caveats (MLX is macOS-only, JAX GPU
   wheels, Numba threading layers).

pytensor_ml targets Python ``>= 3.12``. Its hard dependencies are PyTensor
(``>= 3.2.3``), NumPy, and safetensors.

From PyPI
---------

.. code-block:: bash

    pip install pytensor-ml

From source
-----------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e .

Backends
--------

The default C backend needs nothing extra. Every other backend is an optional
dependency, installed separately and imported only when a graph is actually
compiled against it:

.. code-block:: bash

    pip install numba          # mode="NUMBA"
    pip install jax            # mode="JAX"
    pip install torch          # mode="PYTORCH"
    pip install mlx            # mode="MLX", macOS only

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e ".[dev]"
    pre-commit install

See :doc:`/dev/contributing` for the rest of the contributor setup.
