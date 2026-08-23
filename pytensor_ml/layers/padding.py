import pytensor.tensor as pt

from pytensor.tensor.pad import PadMode
from pytensor.tensor.variable import TensorVariable


def _pad_spatial(
    X: TensorVariable,
    padding: tuple[tuple[int, int], ...],
    mode: PadMode,
    constant_value: float = 0.0,
) -> TensorVariable:
    """Pad the spatial axes of ``(batch, *spatial, channels)``, leaving batch and channels alone."""
    if not any(before or after for before, after in padding):
        return X
    pad_width = [(0, 0), *padding, (0, 0)]
    # Spelled out per branch rather than unpacked from a dict: only `constant` takes a fill value, and
    # `pad` is overloaded per mode, so a `**kwargs` call cannot be matched against those overloads.
    if mode == "constant":
        return pt.pad(X, pad_width, mode="constant", constant_values=constant_value)
    return pt.pad(X, pad_width, mode=mode)
