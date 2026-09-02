import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.variable import Variable

from pytensor_ml.layers.dropout import DropoutLayer
from pytensor_ml.layers.norm import BatchNormLayer, PredictionBatchNormLayer


@register_specialize
@node_rewriter([DropoutLayer])
def drop_degenerate_dropout(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Replace a dropout that keeps everything, or nothing, with the value it computes.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    replacement : Variable or None
        The layer's input when it keeps everything, zeros when it keeps nothing, and None for the
        probabilities in between, which have a mask to apply.
    """
    X, _mask = node.inputs
    if node.op.p == 0.0:
        return [X]
    if node.op.p == 1.0:
        # Keeping nothing scales the survivors by 1/0, which pytensor evaluates while canonicalizing
        # even though the branch it sits on is never selected.
        return [pt.zeros_like(X)]
    return None


predict_db = EquilibriumDB()


@node_rewriter([DropoutLayer])
def remove_dropout_for_prediction(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Replace a dropout layer with its input, dropping it from the graph.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    X : Variable
        The dropout layer's input, which becomes the layer's replacement.
    """
    X, _mask = node.inputs
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
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    X_normalized : Variable
        The input normalized by the accumulated running statistics instead of the batch statistics.
    running_mean : Variable
        The accumulated running mean, unchanged, since prediction writes no statistics.
    running_variance : Variable
        The accumulated running variance, unchanged, on the same terms.
    """
    # The affine parameters are absent when the layer was built with affine=False, so bind them as a
    # variable-length group rather than positionally.
    X, *affine_params, running_mean, running_var = node.inputs

    batch_norm_op = PredictionBatchNormLayer(
        name=f"{node.op.name}",
        n_in=node.op.n_in,
        epsilon=node.op.epsilon,
        affine=node.op.affine,
    )

    X_normalized = batch_norm_op(X, *affine_params, running_mean, running_var)

    return [X_normalized, running_mean, running_var]


predict_db.register(
    "rewrite_batch_stats_to_running_average_stats",
    rewrite_batch_stats_to_running_average_stats,
    "basic",
)
