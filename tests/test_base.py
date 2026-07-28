import pytensor.tensor as pt

from pytensor_ml.layers import Linear


def test_layer_op_equality():
    X = pt.tensor("X", shape=(None, None))

    layer_1 = Linear("Linear_1", n_in=10, n_out=5, bias=True)(X)
    layer_2 = Linear("Linear_1", n_in=10, n_out=5, bias=True)(X)
    layer_3 = Linear("Linear_1", n_in=20, n_out=5, bias=True)(X)

    assert layer_1.owner.op == layer_2.owner.op
    assert layer_1.owner.op != layer_3.owner.op

    assert hash(layer_1.owner.op) == hash(layer_2.owner.op)
    assert hash(layer_1.owner.op) != hash(layer_3.owner.op)
