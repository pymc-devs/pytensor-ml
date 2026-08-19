from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.rewriting.basic import register_specialize

from pytensor_ml.layers.conv import ConvLayerGrad


@register_specialize
@node_rewriter([ConvLayerGrad])
def drop_unused_input_grad(fgraph, node):
    """
    Stop a convolution's pullback computing the input gradient nothing reads.

    :meth:`ConvLayer.pullback` asks for both gradients, because it cannot know which the caller wants --
    only the graph knows that, and only once it is built. Where the input gradient has no clients, which
    is the first convolution of a network, the op is swapped for one that returns the kernel gradient
    alone. A backend then differentiates with respect to the kernel alone, rather than computing an
    input gradient and discarding it: unused outputs of one node are not pruned, unlike unused nodes.
    """
    if not node.op.compute_dX:
        return None

    input_gradient, kernel_gradient = node.outputs
    if fgraph.clients[input_gradient]:
        return None

    lowered = ConvLayerGrad(node.op.kernel_size, node.op.stride, node.op.dilation, compute_dX=False)
    [replacement] = lowered(*node.inputs, return_list=True)
    return {kernel_gradient: replacement}
