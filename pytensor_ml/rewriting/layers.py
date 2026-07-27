from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.tensor.variable import Variable

from pytensor_ml.layers import BatchNormLayer, DropoutLayer, PredictionBatchNormLayer

predict_db = EquilibriumDB()


@node_rewriter([DropoutLayer])
def remove_dropout_for_prediction(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Set dropout probability to zero for all dropout layers.

    Parameters
    ----------
    fgraph: FunctionGraph
        Graph being rewritten
    node: Node
        Node being rewritten

    Returns
    -------
    X: Variable
        The input to the dropout layer, removing the dropout from the graph
    """
    X, rng = node.inputs
    return [X]


predict_db.register(
    "remove_dropout_for_prediction",
    remove_dropout_for_prediction,
    "basic",
)


@node_rewriter([BatchNormLayer])
def rewrite_batch_stats_to_running_average_stats(
    fgraph: FunctionGraph, node: Apply
) -> list[Variable] | None:
    """
    Replace usage of batch mean and variance with running mean and variance.

    Parameters
    ----------
    fgraph: FunctionGraph
        Graph being rewritten
    node: Node
        Node being rewritten

    Returns
    -------
    X_normalized: Variable

    """
    X, loc, scale, running_mean, running_var = node.inputs

    batch_norm_op = PredictionBatchNormLayer(
        name=f"{node.op.name}",
        n_in=node.op.n_in,
        momentum=node.op.momentum,
        epsilon=node.op.epsilon,
        affine=node.op.affine,
    )

    X_normalized = batch_norm_op(X, loc, scale, running_mean, running_var)

    return [X_normalized, None, None]  # type: ignore[list-item]


predict_db.register(
    "rewrite_batch_stats_to_running_average_stats",
    rewrite_batch_stats_to_running_average_stats,
    "basic",
)
