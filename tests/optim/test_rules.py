import inspect

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.gradient import DisconnectedInputError, grad

from pytensor_ml import params
from pytensor_ml.optim import (
    adadelta,
    adadelta_updates,
    adagrad,
    adagrad_updates,
    adam,
    adam_updates,
    adamax,
    adamax_updates,
    adamw,
    adamw_updates,
    compile_train,
    cosine_schedule,
    lbfgs,
    lbfgs_updates,
    nadam,
    nadam_updates,
    rmsprop,
    rmsprop_updates,
    rprop,
    rprop_updates,
    sgd,
    sgd_updates,
)
from pytensor_ml.optim import alias as alias_module
from pytensor_ml.pytensorf import function
from tests.optim.test_lbfgs import dense_inverse_hessian

floatX = pytensor.config.floatX

# The closed-form step identities below are exact in real arithmetic, so the gap is pure rounding.
RTOL = 1e-6 if floatX == "float64" else 1e-4


def trainable(value, name=None, **kwargs):
    """Create a parameter at floatX; a float64 literal would not match the gradients it is updated with."""
    return params.trainable(np.asarray(value, dtype=floatX), name=name, **kwargs)


@pytest.mark.parametrize(
    "rule",
    [
        sgd(learning_rate=1e-2),
        sgd(learning_rate=1e-2, momentum=0.9),
        sgd(learning_rate=1e-2, momentum=0.9, nesterov=True),
        adam(learning_rate=1e-2),
        adam(learning_rate=1e-2, amsgrad=True),
        adamw(learning_rate=1e-2, weight_decay=1e-2),
        adamw(learning_rate=1e-2, weight_decay=1e-2, amsgrad=True),
        adagrad(learning_rate=1e-1),
        adadelta(learning_rate=1.0),
        rmsprop(learning_rate=1e-2),
        rmsprop(learning_rate=1e-2, momentum=0.9),
        rmsprop(learning_rate=1e-2, centered=True),
        nadam(learning_rate=1e-2),
        adamax(learning_rate=1e-2),
        rprop(learning_rate=1e-2),
        lbfgs(learning_rate=1e-2),
    ],
    ids=[
        "sgd",
        "sgd_momentum",
        "sgd_nesterov",
        "adam",
        "adam_amsgrad",
        "adamw",
        "adamw_amsgrad",
        "adagrad",
        "adadelta",
        "rmsprop",
        "rmsprop_momentum",
        "rmsprop_centered",
        "nadam",
        "adamax",
        "rprop",
        "lbfgs",
    ],
)
def test_rule_reduces_loss(run_training, rule):
    history = run_training(rule, n_steps=100)
    assert history[-1] < history[0]


@pytest.mark.parametrize(
    "alias, updates_name",
    [
        (adam, "adam_updates"),
        (adamw, "adamw_updates"),
        (nadam, "nadam_updates"),
        (adamax, "adamax_updates"),
        (rprop, "rprop_updates"),
        (rmsprop, "rmsprop_updates"),
        (adagrad, "adagrad_updates"),
        (adadelta, "adadelta_updates"),
        (lbfgs, "lbfgs_updates"),
    ],
    ids=["adam", "adamw", "nadam", "adamax", "rprop", "rmsprop", "adagrad", "adadelta", "lbfgs"],
)
def test_alias_forwards_every_argument_to_the_matching_parameter(alias, updates_name, monkeypatch):
    # test_rule_reduces_loss cannot see a mis-forward: the loss still falls if beta1 and beta2 are
    # swapped. Distinct values per argument make a swap visible, and reading the argument names off the
    # signature means a newly added hyperparameter fails here until it is forwarded too. sgd is excluded
    # because it composes transforms rather than forwarding.
    forwarded = {}

    def spy(loss_or_gradients, parameters, **kwargs):
        forwarded.update(kwargs)
        return {}

    monkeypatch.setattr(alias_module, updates_name, spy)
    sent = {name: 1.0 + i for i, name in enumerate(inspect.signature(alias).parameters)}

    alias(**sent)("loss", "parameters")

    assert forwarded == sent


@pytest.mark.parametrize("nesterov", [False, True], ids=["classical", "nesterov"])
def test_sgd_momentum_follows_closed_form_trajectory(nesterov):
    """Under a constant gradient, momentum SGD's step at iteration t is a geometric partial sum of the
    gradient. Classical momentum gives ``lr * g * (1 - m**t) / (1 - m)``; Nesterov's look-ahead advances the
    exponent by one to ``lr * g * (1 - m**(t + 1)) / (1 - m)``, so the two paths provably differ."""
    start = np.array([5.0, -3.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])
    loss = (pt.constant(g0, dtype=floatX) * p).sum()  # constant gradient g0, independent of p
    lr, momentum, n_steps = 0.1, 0.9, 5
    rule = sgd(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    fn = function([], loss, updates=rule(loss, [p]))

    previous = start.copy()
    for t in range(1, n_steps + 1):
        fn()
        current = p.get_value()
        exponent = t + 1 if nesterov else t
        expected_step = -lr * g0 * (1 - momentum**exponent) / (1 - momentum)
        np.testing.assert_allclose(current - previous, expected_step, rtol=RTOL)
        previous = current


def test_adam_first_step_is_sign_descent():
    """Bias correction makes Adam's first step exactly ``lr * sign(g)`` per coordinate: the corrected moments
    are ``m_hat = g`` and ``v_hat = g**2``, so the step is ``lr * g / (|g| + eps)``, independent of the
    gradient magnitude."""
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr = 0.1
    function([], loss, updates=adam_updates(loss, [p], learning_rate=lr))()

    step = start - p.get_value()
    np.testing.assert_allclose(step, lr * np.sign(start), rtol=RTOL)


def test_rule_updates_advance_their_counter_without_compile_train():
    """adam_updates has to stand on its own: bias correction reads the counter, so a caller who compiles the
    updates directly must get an advancing count rather than one frozen at the first step."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adam_updates(loss, [p])
    clock = next(key for key in updates if key.name == "adam/step_count")

    step = function([], loss, updates=updates)
    step()
    step()

    assert int(clock.get_value()) == 2


def test_adam_updates_keyed_by_object_with_named_state():
    """State is discovered by object identity; names exist only for serialization."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adam_updates(loss, [p], amsgrad=True)

    assert p in updates  # the exact param object is a key, not a renamed copy
    state_names = {key.name for key in updates if key is not p}
    assert state_names == {
        "adam/step_count",
        "w/adam/first_moment",
        "w/adam/second_moment",
        "w/adam/max_second_moment",
    }


def test_adamw_names_its_own_state():
    """adamw keeps adam's moments but not adam's names: a rule-wide name shared between two rules collides
    when both appear in one training step, and reads as adam's state in a checkpoint."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adamw_updates(loss, [p], amsgrad=True)

    state_names = {key.name for key in updates if key is not p}
    assert state_names == {
        "adamw/step_count",
        "w/adamw/first_moment",
        "w/adamw/second_moment",
        "w/adamw/max_second_moment",
    }


def test_adam_and_adamw_in_one_step_keep_separate_state():
    """The collision the name test above describes in prose, actually run. Both rules derive every slot from
    one namespace argument, so a namespace that defaulted or went missing would hand them the same memoized
    buffers -- and sharing a step counter is invisible, since both write the same increment to it."""
    by_adam = trainable(np.array([1.0, -2.0]), name="by_adam")
    by_adamw = trainable(np.array([1.0, -2.0]), name="by_adamw")
    loss = 0.5 * (by_adam**2).sum() + 0.5 * (by_adamw**2).sum()

    adam_state = adam_updates(loss, [by_adam], learning_rate=0.1, amsgrad=True)
    adamw_state = adamw_updates(loss, [by_adamw], learning_rate=0.1, amsgrad=True)

    assert not set(adam_state) & set(
        adamw_state
    )  # not one buffer between them, including the counter
    function([], loss, updates={**adam_state, **adamw_state})()

    counters = {
        key.name: key.get_value()
        for key in (*adam_state, *adamw_state)
        if key.name.endswith("step_count")
    }
    assert counters == {"adam/step_count": 1, "adamw/step_count": 1}


def test_two_rules_of_one_kind_keep_separate_state_when_named():
    """Two groups on the same rule is the ordinary reason to want two rules, and every slot the rule keeps
    hangs off its namespace -- including the step counter, which is one variable for the whole rule rather
    than one per parameter, so it is the slot that actually collides."""
    weights = trainable(np.array([1.0, -2.0]), name="weights")
    biases = trainable(np.array([0.5]), name="biases")
    loss = 0.5 * (weights**2).sum() + 0.5 * (biases**2).sum()

    on_weights = adam_updates(loss, [weights], learning_rate=0.1, namespace="fast")
    on_biases = adam_updates(loss, [biases], learning_rate=0.01, namespace="slow")

    assert not set(on_weights) & set(on_biases)
    function([], loss, updates={**on_weights, **on_biases})()

    counters = {
        key.name: key.get_value()
        for key in (*on_weights, *on_biases)
        if key.name.endswith("step_count")
    }
    assert counters == {"fast/step_count": 1, "slow/step_count": 1}


@pytest.mark.parametrize(
    "rule_updates, name",
    [
        (adam_updates, "adam"),
        (adamw_updates, "adamw"),
        (nadam_updates, "nadam"),
        (adamax_updates, "adamax"),
        (adagrad_updates, "adagrad"),
        (rmsprop_updates, "rmsprop"),
        (adadelta_updates, "adadelta"),
        (rprop_updates, "rprop"),
        (lbfgs_updates, "lbfgs"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_rule_names_its_state_after_itself_by_default(rule_updates, name):
    """The default has to stay the rule's own name: it is what a checkpoint written before ``namespace``
    existed is keyed on. Plain sgd keeps no state of its own, so it is covered through its alias."""
    parameter = trainable(np.array([1.0, -2.0]), name="w")
    loss = (parameter**2).sum()

    updates = rule_updates(loss, [parameter])

    slots = {key.name for key in updates if key is not parameter}
    assert slots, f"{name} keeps no state to name"
    assert all(slot.split("/")[-2] == name for slot in slots), slots


def test_sgd_names_the_step_counter_its_schedule_reads():
    """sgd allocates no state of its own, so its namespace shows up only on the counter the alias keeps
    for a schedule to read -- which is the slot two sgd rules in one step would collide on."""
    parameter = trainable(np.array([1.0, -2.0]), name="w")
    loss = (parameter**2).sum()

    step = compile_train(loss, sgd(cosine_schedule(0.1, 10), namespace="fast"), inputs=[])

    counters = [
        str(shared.name) for shared in step.get_shared() if str(shared.name).endswith("step_count")
    ]
    assert counters == ["fast/step_count"]


@pytest.mark.parametrize(
    "make_rule",
    [lambda: adam(learning_rate=1e-2), lambda: sgd(learning_rate=1e-2, momentum=0.9)],
    ids=["state_from_a_rule", "state_from_a_transform"],
)
def test_each_invocation_of_a_rule_allocates_its_own_state(make_rule):
    """A rule builds a graph, and the graph holds the state. Two invocations are two optimizers, so the
    buffers they allocate are distinct objects even though their derived names agree. Momentum SGD is
    included because its velocity comes from a transform rather than the rule, a separate allocation path."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    rule = make_rule()

    first = {key for key in rule(loss, [p]) if key is not p}
    second = {key for key in rule(loss, [p]) if key is not p}

    assert first and not first & second


def test_two_functions_from_one_updates_dict_continue_the_same_momentum():
    """Two training functions share state by being compiled from one updates dict. The second then
    continues the first's trajectory instead of restarting it. Under a constant gradient, momentum SGD's
    step at iteration ``t`` is ``lr * g * (1 - m**t) / (1 - m)``, so a continued second step is 1.9x a
    restarted one at ``m = 0.9``."""
    p = trainable(np.zeros(2), name="w")
    gradient = np.array([2.0, -0.5])
    loss = (pt.constant(gradient, dtype=floatX) * p).sum()  # constant gradient, independent of p
    learning_rate, momentum = 0.1, 0.9
    updates = sgd(learning_rate=learning_rate, momentum=momentum)(loss, [p])

    step_once = function([], loss, updates=updates)
    step_again = function([], loss, updates=updates)

    step_once()
    before = p.get_value().copy()
    step_again()

    continued = -learning_rate * gradient * (1 - momentum**2) / (1 - momentum)
    np.testing.assert_allclose(p.get_value() - before, continued, rtol=RTOL)


def test_adamw_first_step_applies_decoupled_decay():
    """AdamW adds a decoupled decay term to Adam's sign-descent step: the first-step displacement is
    ``-lr * (sign(g) + weight_decay * p)``. At t = 1 bias correction makes the Adam part ``sign(g)`` (the
    corrected moments are m_hat = g and v_hat = g**2), and ``weight_decay * p`` is applied straight to the
    parameter rather than through the moments."""
    start = np.array([1.0, -2.0, 3.0])
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, weight_decay = 0.1, 0.25
    function(
        [], loss, updates=adamw_updates(loss, [p], learning_rate=lr, weight_decay=weight_decay)
    )()

    step = p.get_value() - start
    np.testing.assert_allclose(step, -lr * (np.sign(start) + weight_decay * start), rtol=RTOL)


def test_adamw_mask_excludes_parameters_from_decay():
    """The ``mask`` predicate selects which parameters receive decoupled decay; the rest take a pure Adam
    step. Here decay reaches ``w`` but not ``b``, so only ``w``'s step carries the ``weight_decay * p`` term.
    """
    w = trainable(np.array([2.0]), name="w")
    b = trainable(np.array([2.0]), name="b")
    loss = 0.5 * (w**2).sum() + 0.5 * (b**2).sum()  # gradient of each is the parameter itself
    lr, weight_decay = 0.1, 0.5
    updates = adamw_updates(
        loss,
        [w, b],
        learning_rate=lr,
        weight_decay=weight_decay,
        mask=lambda param: param.name == "w",
    )
    function([], loss, updates=updates)()

    np.testing.assert_allclose(w.get_value() - 2.0, -lr * (1.0 + weight_decay * 2.0), rtol=RTOL)
    np.testing.assert_allclose(b.get_value() - 2.0, -lr * 1.0, rtol=RTOL)


def test_adamw_without_decay_is_adam():
    """The decoupled decay term is the *only* difference between the two rules, which is what lets them share
    an implementation. Run over enough steps that the moments, the bias corrections and the accumulated
    trajectory all have to agree, and compare exactly: with no decay these are meant to be the same
    computation, not merely close, so the two drifting apart at all means they stopped sharing it."""
    start = np.array([1.5, -0.5, 2.0])
    by_adam = trainable(start.copy(), name="by_adam")
    by_adamw = trainable(start.copy(), name="by_adamw")

    adam_step = function([], [], updates=adam_updates(0.5 * (by_adam**2).sum(), [by_adam]))
    adamw_step = function(
        [], [], updates=adamw_updates(0.5 * (by_adamw**2).sum(), [by_adamw], weight_decay=0.0)
    )
    for _ in range(20):
        adam_step()
        adamw_step()

    np.testing.assert_array_equal(by_adamw.get_value(), by_adam.get_value())


def test_adagrad_step_decays_as_inverse_sqrt_t():
    """Under a constant gradient the accumulator grows as ``t * g**2``, so AdaGrad's step magnitude decays as
    ``lr / sqrt(t)`` for every coordinate — independent of the gradient magnitude itself."""
    start = np.array([5.0, -3.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])  # 4x apart, yet both coordinates take the same step size
    loss = (pt.constant(g0, dtype=floatX) * p).sum()  # constant gradient g0, independent of p
    lr, n_steps = 0.1, 6
    fn = function([], loss, updates=adagrad_updates(loss, [p], learning_rate=lr))

    previous = start.copy()
    for t in range(1, n_steps + 1):
        fn()
        current = p.get_value()
        step_magnitude = np.abs(current - previous)
        np.testing.assert_allclose(step_magnitude, lr / np.sqrt(t), rtol=1e-4)
        previous = current


def test_adadelta_is_invariant_to_gradient_scale():
    """AdaDelta needs no learning-rate tuning because its update is invariant to the scale of the gradient:
    the ``sqrt(accumulated_update) / sqrt(accumulated_gradient)`` ratio cancels any constant factor on the
    loss. Scaling the loss 100x leaves the parameter trajectory unchanged."""
    start = np.array([1.0, -2.0])

    def trajectory(loss_scale):
        p = trainable(start.copy(), name="w")
        loss = 0.5 * loss_scale * (p**2).sum()  # gradient is loss_scale * p
        fn = function([], loss, updates=adadelta_updates(loss, [p], learning_rate=1.0, rho=0.9))
        values = []
        for _ in range(5):
            fn()
            values.append(p.get_value())
        return np.array(values)

    np.testing.assert_allclose(trajectory(1.0), trajectory(100.0), rtol=1e-4)


def test_rmsprop_first_step_normalizes_gradient_magnitude():
    """RMSProp's defining behavior: the first step size depends on ``learning_rate`` and ``rho`` alone, not
    on the gradient magnitude. With ``v_1 = (1 - rho) g**2`` the step is ``lr * g / sqrt((1 - rho) g**2) =
    lr / sqrt(1 - rho)`` along ``-sign(g)`` — identical for every coordinate no matter how large its gradient.
    """
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, rho = 0.1, 0.9
    function([], loss, updates=rmsprop_updates(loss, [p], learning_rate=lr, rho=rho))()

    step = start - p.get_value()
    expected_magnitude = lr / np.sqrt(1 - rho)
    np.testing.assert_allclose(step, expected_magnitude * np.sign(start), rtol=1e-4)


def test_rmsprop_centered_first_step_uses_centered_variance():
    """Centering subtracts the squared running-mean gradient from the second moment. After one step the
    variance is ``(1 - rho) * rho * g**2``, so the step magnitude is ``lr / sqrt(rho * (1 - rho))`` — larger
    than the uncentered ``lr / sqrt(1 - rho)`` by ``1 / sqrt(rho)``, and still independent of gradient scale.
    """
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, rho = 0.1, 0.9
    function(
        [], loss, updates=rmsprop_updates(loss, [p], learning_rate=lr, rho=rho, centered=True)
    )()

    step = start - p.get_value()
    expected_magnitude = lr / np.sqrt(rho * (1 - rho))
    np.testing.assert_allclose(step, expected_magnitude * np.sign(start), rtol=1e-4)


def test_rmsprop_momentum_converges_to_terminal_velocity():
    """With momentum, RMSProp accumulates the normalized gradient into a velocity buffer. Under a constant
    gradient the normalized gradient tends to sign(g) and the velocity to its fixed point
    ``sign(g) / (1 - momentum)``, so the step magnitude converges to ``lr / (1 - momentum)`` for every
    coordinate, independent of the gradient magnitude."""
    start = np.array([10.0, -10.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])
    loss = (pt.constant(g0, dtype=floatX) * p).sum()  # constant gradient g0, independent of p
    lr, momentum, n_steps = 1e-3, 0.9, 200
    fn = function([], loss, updates=rmsprop_updates(loss, [p], learning_rate=lr, momentum=momentum))

    for _ in range(n_steps - 1):
        fn()
    before = p.get_value().copy()
    fn()
    np.testing.assert_allclose(np.abs(p.get_value() - before), lr / (1 - momentum), rtol=1e-3)


def test_nadam_first_step_scales_by_one_plus_beta1():
    """The Nesterov look-ahead makes Nadam's first step ``lr * (1 + beta1) * sign(g)`` per coordinate: at
    t = 1 the numerator ``beta1 * m_hat + (1 - beta1) * g / (1 - beta1)`` reduces to ``(1 + beta1) g`` while
    ``v_hat = g**2`` normalizes the magnitude away. This is Adam's ``lr * sign(g)`` amplified by ``1 + beta1``.
    """
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, beta1 = 0.1, 0.9
    function([], loss, updates=nadam_updates(loss, [p], learning_rate=lr, beta1=beta1))()

    step = start - p.get_value()
    np.testing.assert_allclose(step, lr * (1 + beta1) * np.sign(start), rtol=RTOL)


def test_adamax_takes_constant_step_under_constant_gradient():
    """AdaMax's infinity-norm denominator saturates at ``|g|`` under a constant gradient while bias
    correction drives the corrected first moment to ``g``, so every step is exactly ``lr * sign(g)`` — it
    never decays the way AdaGrad's does, and is independent of the gradient magnitude."""
    start = np.array([5.0, -3.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])  # 4x apart, yet both coordinates take the same step size
    loss = (pt.constant(g0, dtype=floatX) * p).sum()  # constant gradient g0, independent of p
    lr, n_steps = 0.1, 6
    fn = function([], loss, updates=adamax_updates(loss, [p], learning_rate=lr))

    previous = start.copy()
    for _ in range(n_steps):
        fn()
        current = p.get_value()
        np.testing.assert_allclose(np.abs(current - previous), lr, rtol=1e-4)
        previous = current


def test_rprop_step_grows_geometrically_under_constant_sign():
    """Rprop ignores gradient magnitude and steps by a per-parameter step size that grows by ``eta_plus``
    each time the gradient keeps its sign. Under a constant gradient the step at iteration t is therefore
    ``lr * eta_plus ** (t - 1)``, identical for every coordinate regardless of its gradient."""
    start = np.array([5.0, -3.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])  # 4x apart, yet both coordinates take the same step size
    loss = (pt.constant(g0, dtype=floatX) * p).sum()  # constant gradient g0, independent of p
    lr, eta_plus, n_steps = 0.01, 1.2, 5
    fn = function([], loss, updates=rprop_updates(loss, [p], learning_rate=lr, eta_plus=eta_plus))

    previous = start.copy()
    for t in range(1, n_steps + 1):
        fn()
        current = p.get_value()
        np.testing.assert_allclose(np.abs(current - previous), lr * eta_plus ** (t - 1), rtol=RTOL)
        previous = current


def test_rprop_shrinks_and_skips_on_sign_flip():
    """When the gradient reverses sign, Rprop shrinks that coordinate's step by ``eta_minus``, skips the
    update for that iteration, and zeroes the remembered gradient so the next step is treated as neutral (no
    further size change)."""
    g = pt.vector("g")
    p = trainable(np.zeros(1), name="w")
    lr, eta_minus = 0.1, 0.5
    fn = function([g], p, updates=rprop_updates([g], [p], learning_rate=lr, eta_minus=eta_minus))

    fn([1.0])  # neutral start: step by lr against the gradient sign
    np.testing.assert_allclose(p.get_value(), [-lr])
    fn([-1.0])  # sign flip: update skipped, step size shrinks to lr * eta_minus
    np.testing.assert_allclose(p.get_value(), [-lr])
    fn([-1.0])  # remembered gradient was zeroed, so this step is neutral at the shrunk size
    np.testing.assert_allclose(p.get_value(), [-lr + lr * eta_minus])


def test_lbfgs_satisfies_the_secant_condition_on_the_newest_pair():
    """The inverse-Hessian estimate maps the newest gradient difference onto the parameter difference
    that produced it, ``H y = s``, whatever the initial scaling. A zero gradient holds the parameters
    still while the move before it becomes the newest pair with ``y = -g``, so feeding ``-g`` next has
    to move them by ``-lr * s``. Two parameters, so the memory is split across tensors."""
    g_u, g_v = pt.vector("g_u"), pt.vector("g_v")
    u = trainable(np.zeros(2), name="u")
    v = trainable(np.zeros(1), name="v")
    lr = 0.3
    fn = function([g_u, g_v], [u, v], updates=lbfgs_updates([g_u, g_v], [u, v], learning_rate=lr))
    g = [np.array([1.0, -2.0], dtype=floatX), np.array([0.5], dtype=floatX)]

    fn(*g)
    fn(*[0.5 * gp for gp in g])
    before_move = [x.get_value().copy() for x in (u, v)]
    fn(*[0.5 * gp for gp in g])
    after_move = [x.get_value().copy() for x in (u, v)]
    fn(*[np.zeros_like(gp) for gp in g])  # no move; (after - before, -0.5 g) is now the newest pair
    fn(*[-0.5 * gp for gp in g])

    for x, x_after_move, x_before_move in zip((u, v), after_move, before_move):
        np.testing.assert_allclose(
            x.get_value(), x_after_move - lr * (x_after_move - x_before_move), rtol=RTOL
        )


def test_lbfgs_reaches_the_minimum_of_a_quadratic():
    # Two slots for two dimensions: once both hold pairs the estimate is close to the true inverse
    # Hessian and unit steps close in on the minimizer, which is known in closed form.
    A = np.array([[3.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, -2.0])
    u = trainable(np.array([5.0]), name="u")
    v = trainable(np.array([-3.0]), name="v")
    x = pt.concatenate([u, v])
    loss = 0.5 * x @ pt.constant(A, dtype=floatX) @ x - pt.constant(b, dtype=floatX) @ x
    step = function([], loss, updates=lbfgs_updates(loss, [u, v], learning_rate=1.0, memory_size=2))

    for _ in range(12):
        step()

    np.testing.assert_allclose(
        np.concatenate([u.get_value(), v.get_value()]), np.linalg.solve(A, b), rtol=1e-4
    )


@pytest.mark.parametrize("gradient", [[3.0, -4.0], [0.3, -0.4]], ids=["long", "short"])
def test_lbfgs_first_step_is_the_gradient_capped_to_the_unit_ball(gradient):
    # A gradient of norm 5 is cut to unit length, one of norm 0.5 is left as it is.
    p = trainable(np.zeros(2), name="w")
    loss = (pt.constant(np.array(gradient), dtype=floatX) * p).sum()
    step = function([], loss, updates=lbfgs_updates(loss, [p], learning_rate=1.0))

    step()

    g = np.array(gradient)
    np.testing.assert_allclose(p.get_value(), -min(1.0, 1.0 / np.linalg.norm(g)) * g, rtol=RTOL)


def test_lbfgs_step_matches_the_dense_update_through_a_ring_wrap():
    """On a strictly convex quadratic every pair is accepted, so the memory is the last ``memory_size``
    chronological pairs and each step is ``-lr * H g`` for the dense BFGS matrix built from them. Two
    slots over six steps wrap the ring twice; a rule that overwrote the wrong slot or read the newest
    pair off by one would drift from the dense reference from the third step on."""
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0]) + 0.1
    A = A @ A.T
    b = np.array([0.3, -1.0, 2.0, 0.5, -0.7])
    u = trainable(np.array([1.0, -2.0, 0.5]), name="u")
    v = trainable(np.array([3.0, 1.0]), name="v")
    x = pt.concatenate([u, v])
    loss = 0.5 * x @ pt.constant(A, dtype=floatX) @ x - pt.constant(b, dtype=floatX) @ x
    memory_size, lr = 2, 0.5
    updates = lbfgs_updates(loss, [u, v], learning_rate=lr, memory_size=memory_size)
    step = function([], pt.grad(loss, [u, v]), updates=updates)  # gradient before the update

    pairs = []
    previous = None
    for _ in range(6):
        x_before = np.concatenate([u.get_value(), v.get_value()])
        g_before = np.concatenate([g.ravel() for g in step()])
        if previous is not None:
            pairs.append((x_before - previous[0], g_before - previous[1]))
        if pairs:
            s, y = pairs[-1]
            gamma = (s @ y) / (y @ y)
        else:
            gamma = min(1.0, 1.0 / np.linalg.norm(g_before))
        H = dense_inverse_hessian(gamma, pairs[-memory_size:], x_before.size)
        np.testing.assert_allclose(
            np.concatenate([u.get_value(), v.get_value()]), x_before - lr * H @ g_before, rtol=RTOL
        )
        previous = (x_before, g_before)


def test_lbfgs_schedule_reads_the_rules_own_clock():
    # The rule keeps a step counter to tell the first step apart; a scheduled rate must read that same
    # clock rather than allocate a second one measuring the same time.
    parameter = trainable(np.array([1.0, -2.0]), name="w")
    loss = (parameter**2).sum()

    step = compile_train(loss, lbfgs(cosine_schedule(0.1, 10), memory_size=2), inputs=[])

    counters = [
        str(shared.name) for shared in step.get_shared() if str(shared.name).endswith("step_count")
    ]
    assert counters == ["lbfgs/step_count"]


def test_lbfgs_without_initial_scaling_starts_along_the_raw_gradient():
    p = trainable(np.array([3.0, -4.0]), name="w")
    loss = (pt.constant(np.array([3.0, -4.0]), dtype=floatX) * p).sum()
    step = function(
        [], loss, updates=lbfgs_updates(loss, [p], learning_rate=0.1, scale_init_precond=False)
    )

    step()

    np.testing.assert_allclose(p.get_value(), [3.0, -4.0] - 0.1 * np.array([3.0, -4.0]), rtol=RTOL)


def test_lbfgs_rejects_a_pair_with_negative_curvature():
    """A step whose gradient change opposes the parameter change would make the inverse-Hessian estimate
    indefinite, so the pair is left out of the memory, the ring index does not advance, and the next step
    is the one an empty memory gives."""
    g = pt.vector("g")
    p = trainable(np.zeros(2), name="w")
    lr = 0.1
    updates = lbfgs_updates([g], [p], learning_rate=lr, memory_size=2)
    memory = next(key for key in updates if key.name == "w/lbfgs/value_differences")
    pairs_written = next(key for key in updates if key.name == "lbfgs/pairs_written")
    fn = function([g], p, updates=updates)

    fn(np.array([1.0, 0.0], dtype=floatX))  # first step: no previous point, nothing to write
    before = p.get_value().copy()
    fn(np.array([2.0, 0.0], dtype=floatX))  # p moved along -g and g grew: y . s < 0, rejected
    assert int(pairs_written.get_value()) == 0
    np.testing.assert_array_equal(memory.get_value(), 0.0)
    np.testing.assert_allclose(p.get_value(), before - lr * 0.5 * np.array([2.0, 0.0]), rtol=RTOL)
    fn(np.array([0.5, 0.0], dtype=floatX))  # g shrank along the move: y . s > 0, accepted
    assert int(pairs_written.get_value()) == 1
    assert np.any(memory.get_value()[0] != 0.0)


def test_lbfgs_rejects_a_zero_memory_size():
    p = trainable(np.zeros(2), name="w")
    with pytest.raises(ValueError, match="memory_size must be at least 1"):
        lbfgs_updates((p**2).sum(), [p], memory_size=0)


def test_amsgrad_caps_step_after_gradient_spike():
    """AMSGrad divides by the running maximum of the second moment, so a large gradient permanently caps the
    denominator. Once gradients shrink it therefore takes a smaller step than plain Adam, whose decaying
    second moment lets the effective step size grow back."""
    g = pt.vector("g")

    spike = np.array([10.0], dtype=floatX)
    # 1e-3 is not exact in float32, so pytensor rejects the bare literal rather than downcasting it.
    settled = np.array([1e-3], dtype=floatX)

    def step_after_spike(amsgrad):
        p = trainable(np.zeros(1), name="w")
        updates = adam_updates([g], [p], learning_rate=0.1, beta2=0.9, amsgrad=amsgrad)
        fn = function([g], p, updates=updates)
        fn(spike)
        for _ in range(20):
            fn(settled)
        before = p.get_value().copy()
        fn(settled)
        return np.abs(p.get_value() - before)[0]

    assert step_after_spike(amsgrad=True) < step_after_spike(amsgrad=False)


def test_precomputed_gradients_accepted():
    p = trainable(np.ones(2), name="w")
    gradients = [pt.constant(np.array([0.5, -0.5], dtype=floatX))]
    updates = sgd(learning_rate=1.0)(gradients, [p])
    np.testing.assert_allclose(function([], updates[p])(), [0.5, 1.5])


def test_get_gradients_rejects_count_mismatch():
    weight = trainable(np.ones(2), name="w")
    bias = trainable(np.ones(2), name="b")
    one_gradient = [pt.constant(np.ones(2, dtype=floatX))]
    with pytest.raises(ValueError, match="1 gradients for 2 parameters"):
        sgd_updates(one_gradient, [weight, bias])


def test_get_gradients_names_the_parameters_the_loss_cannot_reach():
    """Pytensor raises with an empty message here, which says nothing about which parameter is at fault."""
    reachable = trainable(np.ones(2), name="reachable")
    unreachable = trainable(np.ones(2), name="unreachable")
    loss = (reachable**2).sum()

    with pytest.raises(DisconnectedInputError, match=r"\['unreachable'\]"):
        sgd_updates(loss, [reachable, unreachable])


def test_get_gradients_names_a_parameter_lost_to_a_second_derivative():
    """The shape a PINN hits: an output bias is additive in the network output, so it survives in the loss as
    written and vanishes once the loss differentiates twice with respect to the input."""
    x = pt.scalar("x")
    weight = trainable(1.0, name="weight")
    scale = trainable(1.0, name="scale")
    output_bias = trainable(1.0, name="output_bias")
    u = pt.tanh(x * weight) * scale + output_bias
    loss = grad(grad(u, x), x) ** 2

    with pytest.raises(DisconnectedInputError, match=r"\['output_bias'\]"):
        sgd_updates(loss, [weight, scale, output_bias])


def test_a_functional_rule_reads_a_schedule_off_its_own_clock():
    """A schedule is a learning rate, so the functional API takes one where it takes a float. The rule
    resolves it against the clock it already counts its own steps on, so the graph holds one clock."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adam_updates(loss, [p], learning_rate=cosine_schedule(0.1, total_steps=10))
    clocks = [key for key in updates if isinstance(key, params.StepCounter)]

    step = function([], loss, updates=updates)
    step()
    step()

    assert [clock.name for clock in clocks] == ["adam/step_count"]
    assert int(clocks[0].get_value()) == 2


def test_a_numpy_scalar_rate_does_not_widen_a_float32_graph():
    """A rate read back from a config or a checkpoint arrives as a numpy scalar, and a ``np.float64`` in a
    float32 graph makes an update pytensor refuses with an error naming the parameter."""
    with pytensor.config.change_flags(floatX="float32"):
        p = params.trainable(np.zeros(3, dtype="float32"), name="w")
        loss = (p**2).sum()
        updates = sgd_updates(loss, [p], learning_rate=np.float64(0.1))

        assert updates[p].dtype == "float32"


def test_sgd_momentum_keeps_its_velocity_under_its_own_namespace():
    """The ``namespace`` the docstring offers against colliding state has to reach the velocity too, or a
    momentum sgd next to a ``trace`` still collides and the remedy on the sgd side does nothing."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = sgd(learning_rate=0.1, momentum=0.9, namespace="slow")(loss, [p])

    assert {key.name for key in updates if key is not p} == {"w/slow/velocity"}
