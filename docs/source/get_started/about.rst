About pytensor_ml
=================

.. note::

   **WRITEME.** This page is a stub. Fill in project motivation, scope, and
   how pytensor_ml relates to the rest of the ecosystem (PyTorch, JAX/Flax,
   Keras) and to PyMC.

pytensor_ml is a deep learning library built on PyTensor's symbolic graph and
rewrite system. A network is a graph, not an object hierarchy with a runtime
attached: layers are graph constructors, parameters are shared variables, and
a training step is a compiled PyTensor function whose updates are the
optimizer.

That design is what the library trades on. Gradients come from PyTensor's
symbolic differentiation, performance comes from its rewrites and its
backends, and a model composes with any other PyTensor graph — including a
PyMC model — because there is nothing else to interoperate with.

.. note::

   pytensor_ml is pre-alpha. The API is still moving and there is no
   release-to-release compatibility guarantee yet.
