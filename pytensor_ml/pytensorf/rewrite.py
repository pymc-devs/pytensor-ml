from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, rewrite_graph
from pytensor.tensor.variable import Variable


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad."""
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


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
