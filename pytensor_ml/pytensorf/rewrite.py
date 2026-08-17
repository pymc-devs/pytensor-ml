from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, rewrite_graph
from pytensor.tensor.variable import Variable

from pytensor_ml.rewriting.scan import optimize_db


def hoist_scan_draws(outputs):
    """
    Lift any draw written inside a scan out of the loop, across a whole set of graphs at once.

    Rewriting the outputs and the updates together keeps a subgraph they share shared, so a loop read by
    both is lifted once. See :func:`~pytensor_ml.rewriting.scan.hoist_draws_out_of_scan` for what the
    lift does and why a draw cannot stay in the loop.

    Parameters
    ----------
    outputs : sequence of Variable
        Graphs to rewrite, which are left untouched; the rewrite runs on a clone.

    Returns
    -------
    list of Variable
        The rewritten graphs, in the order given.
    """
    fgraph = FunctionGraph(outputs=list(outputs), clone=True, copy_inputs=False)
    optimize_db.query(RewriteDatabaseQuery(include=["basic"])).rewrite(fgraph)
    return list(fgraph.outputs)


def rewrite_pregrad(graph):
    """
    Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad.

    Holds back the canonicalization that splices out pytensor's gradient markers, so a stop-gradient in
    ``graph`` still reaches ``grad``. Lifts a draw written inside a scan out of the loop, which
    :func:`grad` needs rather than merely tolerates: a draw inside the differentiated region leaves no
    fixed sample to take a gradient against, and scan reports it as an undefined gradient.
    """
    simplified = rewrite_graph(
        graph, include=("canonicalize", "stabilize"), exclude=("local_view_op",)
    )
    [hoisted] = hoist_scan_draws([simplified])
    return hoisted


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
