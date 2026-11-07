import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor.compile.mode import MLX

from pytensor_ml.layers import rotary_embedding
from tests.test_positional import rope_np


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
@pytest.mark.parametrize("base", [10_000.0, 500_000.0], ids=["default_base", "base_past_float16"])
def test_rope_angles_survive_float16_inputs(pairing, base):
    """mlx executes float16 natively, so this is where the narrowing actually happens: position 2049
    has no float16 representation, 70000 is past its ceiling, and so is a base of 500000."""
    values = np.random.default_rng(0).normal(size=(4, 8)).astype("float16")
    positions = np.array([2049, 5001, 40001, 70000])

    x = pt.tensor("x", shape=values.shape, dtype="float16")
    position_ids = pt.lvector("position_ids")
    output = rotary_embedding(x, position_ids, base=base, pairing=pairing)
    computed = np.asarray(pytensor.function([x, position_ids], output, mode=MLX)(values, positions))

    assert output.dtype == "float16"
    # float32 spaces angles near 70000 about 0.008 apart, which sets the floor on this tolerance.
    np.testing.assert_allclose(
        computed.astype("float64"),
        rope_np(values, positions, base=base, pairing=pairing),
        atol=0.02,
    )
