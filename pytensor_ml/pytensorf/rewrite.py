from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, rewrite_graph
from pytensor.tensor.variable import Variable

# Pytensor's gradient markers -- disconnected_grad, zero_grad, grad_clip, grad_scale -- are all ViewOp
# subclasses that do nothing at runtime, so canonicalize's local_view_op splices them out. Dropping one
# before differentiating discards the instruction it encodes, and a stop-gradient silently stops stopping.
# Keeping them costs nothing: pytensor removes them at compile time, after grad has read them.
_REWRITES_THAT_DROP_GRADIENT_MARKERS = ("local_view_op",)


def rewrite_pregrad(graph):
    """
    Apply simplifying and stabilizing rewrites that are safe to use before differentiating.

    Canonicalization and stabilization put the graph in the form pytensor differentiates most stably.
    Rewrites that would strip pytensor's gradient markers are held back, so a
    :func:`pytensor.gradient.disconnected_grad` or :func:`pytensor.gradient.grad_clip` in the graph still
    reaches ``grad``.

    Parameters
    ----------
    graph : FunctionGraph, Variable, or sequence of Variable
        The graph to rewrite.

    Returns
    -------
    FunctionGraph, Variable, or list of Variable
        The rewritten graph, matching the form of ``graph``. A FunctionGraph is rewritten in place and
        returned; a Variable or sequence is rewritten on a clone, leaving the original untouched.
    """
    return rewrite_graph(
        graph,
        include=("canonicalize", "stabilize"),
        exclude=_REWRITES_THAT_DROP_GRADIENT_MARKERS,
        clone=True,
    )


def rewrite_for_prediction(graph):
    """
    Apply rewrites to specialize a graph for forward passes (e.g. removing Dropout layers).

    Parameters
    ----------
    graph : FunctionGraph, Variable, or sequence of Variable
        The graph to specialize.

    Returns
    -------
    FunctionGraph, Variable, or list of Variable
        The specialized graph, matching the form of ``graph``. A FunctionGraph is rewritten in place and
        returned; a Variable or sequence is rewritten on a clone, leaving the original untouched.
    """
    # Local by design, matching pytensor's own op/rewrite pattern: the rewrites import the layer ops they
    # match on, so a module-scope import would tie this module to the whole layer surface.
    from pytensor_ml.rewriting.layers import predict_db

    rewriter = predict_db.query(RewriteDatabaseQuery(include=["basic"]))

    if isinstance(graph, FunctionGraph):
        rewriter.rewrite(graph)
        return graph

    has_single_output = isinstance(graph, Variable)
    fgraph = FunctionGraph(
        outputs=[graph] if has_single_output else list(graph), clone=True, copy_inputs=False
    )
    rewriter.rewrite(fgraph)

    return fgraph.outputs[0] if has_single_output else fgraph.outputs
