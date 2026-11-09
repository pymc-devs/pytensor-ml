% Adapted from pymc-examples (MIT License): https://github.com/pymc-devs/pymc-examples/blob/main/examples/extra_installs.md

:::{attention}
This notebook uses libraries that pytensor_ml does not depend on, so they have to be installed
separately to run it. Open the dropdown below for instructions.
:::

::::::{dropdown} Extra dependencies install instructions
:icon: package-dependencies

Running this notebook needs a working pytensor_ml installation, described in {doc}`/install`, and
these extra packages. They can be installed with pip:

> $ pip install {{ pip_dependencies }}

To install them from inside the notebook, run pip through the notebook's own interpreter, so the
packages land in the environment the kernel uses:

> import sys
>
> !{sys.executable} -m pip install {{ pip_dependencies }}

Or with conda, preferably from [conda-forge](https://conda-forge.org/):

> $ conda install {{ conda_dependencies }}

{{ extra_install_notes }}
::::::
