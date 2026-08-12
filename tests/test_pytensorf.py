import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import (
    DisconnectedInputError,
    disconnected_grad,
    grad,
    grad_clip,
    zero_grad,
)
from pytensor.graph.traversal import ancestors
from pytensor.scan import scan
from pytensor.tensor import random as ptr

from pytensor_ml.layers import Dropout, DropoutLayer, Linear, Sequential
from pytensor_ml.pytensorf import (
    collect_trainable_params,
    compile_predict,
    function,
    rewrite_for_prediction,
    rewrite_pregrad,
)
from pytensor_ml.state import initialize_params


def contains_op(graph, op_type):
    return any(isinstance(var.owner.op, op_type) for var in ancestors([graph]) if var.owner)


def test_compile_predict_removes_dropout():
    # The inference rewrite drops Dropout, so repeated calls are deterministic; without it they would differ.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)
    parameters = collect_trainable_params(prediction)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)

    predict = compile_predict(prediction, inputs=[X])
    X_values = np.random.default_rng(0).normal(size=(8, 4)).astype(config.floatX)
    first, second = predict(X_values), predict(X_values)

    # Guards the test itself: at the zero initialization the output is identically zero, and so stable
    # across calls whether or not dropout was removed.
    assert np.any(first != 0)
    np.testing.assert_allclose(first, second)


def test_rewrite_for_prediction_leaves_the_original_graph_intact():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)

    assert not contains_op(rewrite_for_prediction(prediction), DropoutLayer)
    assert contains_op(prediction, DropoutLayer)


@pytest.mark.parametrize(
    "marker, expected_gradient",
    [
        (zero_grad, [0.0, 0.0, 0.0]),
        (lambda W: grad_clip(W, -1.0, 1.0), [1.0, 1.0, 1.0]),
    ],
    ids=["zero_grad", "grad_clip"],
)
def test_rewrite_pregrad_keeps_gradient_markers(marker, expected_gradient):
    # Every marker is a ViewOp, which canonicalize is otherwise free to splice out. Unmarked, the gradient
    # of this loss is 2 * W, so both expectations only hold while the marker survives into grad.
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (marker(W) ** 2).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), expected_gradient)


def test_rewrite_pregrad_keeps_a_stop_gradient_out_of_the_gradient():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    # Differentiating this reads the detached factor as a constant, so the gradient is W itself; with the
    # marker gone the graph is W ** 2 and the gradient comes out twice as large.
    loss = (W * disconnected_grad(W)).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), W.get_value())


def test_rewrite_pregrad_leaves_a_fully_detached_parameter_disconnected():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (disconnected_grad(W) ** 2).sum()

    with pytest.raises(DisconnectedInputError):
        grad(rewrite_pregrad(loss), W)


def test_compiling_leaves_a_generator_another_function_is_drawing_from_alone():
    """Compiling used to replace every generator it touched, so building a second function mid-training
    jumped the first one's noise stream."""

    def four_draws(with_intervening_compile):
        rng = shared(np.random.default_rng(0), name="rng")
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise)
        drawn = [float(draw()) for _ in range(2)]
        if with_intervening_compile:
            _, other_noise = ptr.normal(rng=rng, return_next_rng=True)
            function([], other_noise)
        return drawn + [float(draw()) for _ in range(2)]

    assert four_draws(with_intervening_compile=True) == four_draws(with_intervening_compile=False)


@pytest.mark.parametrize("applications", [1, 2], ids=["applied_once", "applied_twice"])
def test_a_seeded_dropout_reproduces_across_identical_runs(applications):
    """The seed a caller puts on a layer has to survive compilation, or `random_state=` means nothing. It has
    to hold however many times the layer is applied, since each application draws off its own generator."""

    def dropout_masks():
        X = pt.tensor("X", shape=(None, 3))
        dropout = Dropout(p=0.5, random_state=0)
        layers = [Linear("fc", n_in=3, n_out=3)]
        for _ in range(applications):
            layers.append(dropout)
        prediction = Sequential(*layers)(X)
        for parameter in collect_trainable_params(prediction):
            parameter.set_value(np.ones_like(parameter.get_value()))
        forward = function([X], prediction)
        features = np.ones((4, 3), dtype=config.floatX)
        return [forward(features) for _ in range(3)]

    for first, second in zip(dropout_masks(), dropout_masks()):
        np.testing.assert_allclose(first, second)


def test_random_seed_makes_an_unseeded_graph_reproducible():
    def two_draws():
        rng = shared(np.random.default_rng(), name="rng")  # deliberately unseeded
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise, random_seed=42)
        return [float(draw()) for _ in range(2)]

    assert two_draws() == two_draws()


def test_a_reused_dropout_instance_keeps_drawing_new_masks():
    """Using one Dropout object at two points in a network is an ordinary thing to write, and it used to
    freeze the mask for the whole run."""
    X = pt.tensor("X", shape=(None, 3))
    dropout = Dropout(p=0.5, random_state=0)
    prediction = Sequential(
        Linear("fc1", n_in=3, n_out=4), dropout, Linear("fc2", n_in=4, n_out=2), dropout
    )(X)
    for parameter in collect_trainable_params(prediction):
        parameter.set_value(np.ones_like(parameter.get_value()))

    forward = function([X], prediction)
    features = np.ones((4, 3), dtype=config.floatX)
    outputs = [forward(features) for _ in range(4)]

    assert (
        len(dropout.generators) == 2
    )  # one per application, so neither draw is starved of updates
    assert all(
        np.any(outputs[i] != outputs[j]) for i in range(4) for j in range(i + 1, 4)
    )  # a fresh mask on every call, not just eventually


def test_a_returned_generator_is_the_one_the_caller_asked_for():
    """Returning a generator alongside a draw from it is not two draws: an output reads the generator without
    consuming it, so it must keep reading the caller's own."""
    rng = shared(np.random.default_rng(0), name="rng")
    _, noise = ptr.normal(rng=rng, return_next_rng=True)
    draw = function([], [noise, rng])

    expected = rng.get_value(borrow=True).bit_generator.state
    returned = draw()[1].bit_generator.state

    assert returned == expected


def test_a_generator_read_only_by_an_update_still_advances():
    """A rule that adds noise to its step reads a generator the outputs never touch, so collecting updates
    from the outputs alone left that generator frozen and every step took the identical perturbation."""
    rng = shared(np.random.default_rng(0), name="rng")
    parameter = shared(np.zeros(()), name="w")
    _, noise = ptr.normal(rng=rng, return_next_rng=True)

    step = function([], parameter**2, updates={parameter: parameter + noise})

    increments = []
    for _ in range(3):
        before = float(parameter.get_value())
        step()
        increments.append(float(parameter.get_value()) - before)

    assert not np.allclose(increments[0], increments[1])
    assert not np.allclose(increments[1], increments[2])


def test_two_distinct_draws_off_one_generator_are_rejected():
    """One generator cannot serve two different draws: it has no single next state, so nothing can advance
    it and every call would repeat. pytensor calls such a graph inconsistent and threads no update, which
    would otherwise show up as a training run whose noise never changes."""
    rng = shared(np.random.default_rng(0), name="rng")
    _, normal_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, uniform_draw = ptr.uniform(rng=rng, return_next_rng=True)

    with pytest.raises(ValueError, match="which nothing advances"):
        function([], [normal_draw, uniform_draw])


def test_a_shared_generator_is_accepted_when_the_caller_advances_it():
    """The check is about the assembled updates, not about how the graph looks: a caller who threads the
    generator themselves has answered the question, and their update stands."""
    rng = shared(np.random.default_rng(0), name="rng")
    next_rng, normal_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, uniform_draw = ptr.uniform(rng=next_rng, return_next_rng=True)

    draw = function([], [normal_draw, uniform_draw], updates={rng: next_rng})

    first, second = draw(), draw()
    assert float(first[0]) != float(second[0])  # the draw off the generator advances
    assert float(first[1]) != float(second[1])  # and so does the one off the threaded next state


def test_a_draw_inside_a_scan_advances_its_generator():
    """Nothing is passed to `function` as an update: a scan carrying its generator as a recurrent state
    exposes it as an outer input with a matching outer output, and the collector reads that mapping. This is
    the whole of the inner-graph handling, inherited from pymc, and the first test to run it."""
    rng = shared(np.random.default_rng(0), name="rng")

    def one_step(total, generator):
        next_generator, draw = ptr.normal(size=(), rng=generator, return_next_rng=True)
        return total + draw, next_generator

    # Two recurrent states out, not a graph and an updates dict -- the generator is carried as state.
    trace, _generators = scan(
        one_step, outputs_info=[pt.zeros(()), rng], n_steps=5, return_updates=False
    )
    step = function([], trace)

    draws_within_one_call = np.diff(step(), prepend=0.0)

    # Both halves matter: a loop reusing one generator state draws the same number five times while its
    # outer state still advances between calls, so neither assertion catches that alone.
    assert len(np.unique(draws_within_one_call)) == 5
    assert not np.array_equal(step(), step())


def test_a_scan_that_draws_without_threading_its_generator_is_rejected():
    """The loop draws from a generator it captured rather than carried, so there is no outer output holding
    its final state and no way to advance it -- every call would replay the same five draws."""
    rng = shared(np.random.default_rng(0), name="rng")

    def one_step(total):
        _, draw = ptr.normal(size=(), rng=rng, return_next_rng=True)
        return total + draw

    trace = scan(one_step, outputs_info=[pt.zeros(())], n_steps=5, return_updates=False)

    with pytest.raises(ValueError, match="No update found for at least one RNG used in Scan"):
        function([], trace[-1])


def test_a_draw_inside_an_op_from_graph_advances_its_generator():
    """Every layer in this library is an OpFromGraph, so a layer that draws inside its own inner graph lands
    here. Handing the advanced generator back as an output is what makes it reachable from outside."""
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    next_rng, inner_draw = ptr.normal(size=(3,), rng=inner_rng, return_next_rng=True)
    draw, _ = OpFromGraph([inner_rng], [inner_draw, next_rng])(rng)

    step = function([], draw)

    assert not np.array_equal(step(), step())


def test_an_op_from_graph_that_draws_without_threading_is_rejected():
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    _, inner_draw = ptr.normal(size=(3,), rng=inner_rng, return_next_rng=True)
    draw = OpFromGraph([inner_rng], [inner_draw])(rng)

    with pytest.raises(
        ValueError, match="No update found for at least one RNG used in OpFromGraph"
    ):
        function([], draw)


def test_a_generator_an_inner_graph_never_draws_from_is_left_alone():
    """Receiving a generator is not drawing from one, and only the inner graph knows which happened. Treating
    every non-output use as a draw rejected this graph with a message saying it draws from a generator that
    nothing in it touches."""
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    inner_x = pt.vector("x")
    doubled = OpFromGraph([inner_rng, inner_x], [inner_x * 2.0])(rng, pt.ones(3))

    np.testing.assert_allclose(function([], doubled)(), np.full(3, 2.0))


def test_a_draw_shared_between_an_output_and_an_update_is_rejected():
    """An update expression is a reader like any other, so a generator drawn from by both an output and an
    update is read twice and cannot be advanced once."""
    rng = shared(np.random.default_rng(0), name="rng")
    accumulator = shared(np.zeros(()), name="accumulator")
    _, output_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, update_draw = ptr.uniform(rng=rng, return_next_rng=True)

    with pytest.raises(ValueError, match="which nothing advances"):
        function([], output_draw, updates={accumulator: accumulator + update_draw})
