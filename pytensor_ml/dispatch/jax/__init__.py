# Importing this package registers pytensor_ml's JAX funcify dispatches. Submodules mirror the core
# library layout (attention.py <- pytensor_ml/attention.py); add one per marker op that gets a kernel.
import pytensor_ml.dispatch.jax.attention
