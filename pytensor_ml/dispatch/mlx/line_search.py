from functools import cache

import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify

from pytensor_ml.optim.line_search import (
    _ZOOM_FIELDS,
    FINISH,
    START,
    STEP,
    TRIAL,
    LineSearchOp,
    ZoomLineSearch,
    find_piece,
)

_FIELD = {field: index for index, field in enumerate(_ZOOM_FIELDS)}

# One trial of `ZoomLineSearch.step`, written out for one thread: the state the trial was proposed
# from, then the trial step and the loss and slope there, give the next state and the next trial step.
# It follows the PyTensor graph expression for expression, so the two take the same steps. `maximum`
# and `minimum` propagate NaN, as PyTensor's do and Metal's `fmax`/`fmin` do not.
_ZOOM_HEADER = """
inline float nan_max(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : fmax(a, b); }
inline float nan_min(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : fmin(a, b); }

inline float cubic_minimizer(float a, float fa, float fpa, float b, float fb, float c, float fc) {
    float db = b - a;
    float dc = c - a;
    float denominator = (db * dc) * (db * dc) * (db - dc);
    float residual_b = fb - fa - fpa * db;
    float residual_c = fc - fa - fpa * dc;
    float cubic = (dc * dc * residual_b - db * db * residual_c) / denominator;
    float quadratic = (-(dc * dc * dc) * residual_b + db * db * db * residual_c) / denominator;
    return a + (-quadratic + sqrt(quadratic * quadratic - 3.0f * cubic * fpa)) / (3.0f * cubic);
}

inline float quadratic_minimizer(float a, float fa, float fpa, float b, float fb) {
    float db = b - a;
    float curvature = (fb - fa - fpa * db) / (db * db);
    return a - fpa / (2.0f * curvature);
}
"""

_ZOOM_SOURCE = """
    float S[{n_fields}];
    for (uint i = 0; i < {n_fields}; ++i) S[i] = state[i];
    float t = trial[0];
    float f = trial[1];
    float s = trial[2];
    float value0 = anchor[0];
    float slope0 = anchor[1];
    float guess = anchor[2];

    for (uint i = 0; i < {n_fields}; ++i) next_state[i] = S[i];
    next_trial[0] = t;
    if (S[{done}] != 0.0f || S[{failed}] != 0.0f) return;

    float count = S[{count}];
    bool zooming = S[{interval_found}] != 0.0f;
    float raw_trial = count == 0.0f ? guess : {increase_factor} * S[{stepsize}];
    bool max_reached = {has_max} ? raw_trial >= {max_learning_rate} : false;
    float low = S[{low}], value_low = S[{value_low}], slope_low = S[{slope_low}];
    float high = S[{high}], value_high = S[{value_high}], slope_high = S[{slope_high}];
    float width = fabs(high - low);

    float decrease = f - value0 - {slope_rtol} * t * slope0;
    if ({has_approx}) {{
        float approximate = nan_max(
            s - (2.0f * {slope_rtol} - 1.0f) * slope0,
            f - value0 - {approx_dec_rtol} * fabs(value0)
        );
        decrease = nan_min(approximate, decrease);
    }}
    decrease = nan_max(decrease, 0.0f);
    if (isnan(decrease)) decrease = INFINITY;
    float curvature = nan_max(fabs(s) - {curv_rtol} * fabs(slope0), 0.0f);
    if (isnan(curvature)) curvature = INFINITY;
    bool satisfied = nan_max(decrease, curvature) <= {tol};
    bool sufficient = decrease <= {tol};
    bool out_of_steps = count + 1.0f >= {max_steps};

    float safe_stepsize, safe_value, cubic_ref, value_cubic_ref;
    float new_low, new_value_low, new_slope_low, new_high, new_value_high, new_slope_high;
    bool interval_found, done, failed;
    if (!zooming) {{
        safe_stepsize = sufficient ? t : S[{safe_stepsize}];
        safe_value = sufficient ? f : S[{safe_value}];
        bool high_is_trial = decrease > 0.0f || (f >= S[{value}] && count > 0.0f);
        bool low_is_trial = s >= 0.0f && !high_is_trial;
        new_low = low_is_trial ? t : S[{stepsize}];
        new_value_low = low_is_trial ? f : S[{value}];
        new_slope_low = low_is_trial ? s : S[{slope}];
        new_high = low_is_trial ? S[{stepsize}] : t;
        new_value_high = low_is_trial ? S[{value}] : f;
        new_slope_high = low_is_trial ? S[{slope}] : s;
        cubic_ref = new_low;
        value_cubic_ref = new_value_low;
        interval_found = high_is_trial || low_is_trial || satisfied;
        done = satisfied || (max_reached && !interval_found);
        failed = out_of_steps && !done;
    }} else {{
        bool improves = sufficient && f < S[{safe_value}];
        safe_stepsize = improves ? t : S[{safe_stepsize}];
        safe_value = improves ? f : S[{safe_value}];
        done = satisfied;
        bool high_is_middle = decrease > 0.0f || f >= value_low;
        bool high_is_low = (s * (high - low) >= 0.0f) && !high_is_middle;
        float middle_high = high_is_middle ? t : high;
        float middle_value_high = high_is_middle ? f : value_high;
        float middle_slope_high = high_is_middle ? s : slope_high;
        new_high = high_is_low ? low : middle_high;
        new_value_high = high_is_low ? value_low : middle_value_high;
        new_slope_high = high_is_low ? slope_low : middle_slope_high;
        new_low = !high_is_middle ? t : low;
        new_value_low = !high_is_middle ? f : value_low;
        new_slope_low = !high_is_middle ? s : slope_low;
        cubic_ref = (high_is_middle || high_is_low) ? high : low;
        value_cubic_ref = (high_is_middle || high_is_low) ? value_high : value_low;
        interval_found = true;
        bool too_narrow = width <= {stepsize_precision} && safe_stepsize > 0.0f;
        failed = (out_of_steps || too_narrow) && !done;
    }}

    next_state[{count}] = count + 1.0f;
    next_state[{stepsize}] = t;
    next_state[{value}] = f;
    next_state[{slope}] = s;
    next_state[{decrease_error}] = decrease;
    next_state[{curvature_error}] = curvature;
    next_state[{interval_found}] = interval_found ? 1.0f : 0.0f;
    next_state[{done}] = done ? 1.0f : 0.0f;
    next_state[{failed}] = failed ? 1.0f : 0.0f;
    next_state[{low}] = new_low;
    next_state[{value_low}] = new_value_low;
    next_state[{slope_low}] = new_slope_low;
    next_state[{high}] = new_high;
    next_state[{value_high}] = new_value_high;
    next_state[{slope_high}] = new_slope_high;
    next_state[{cubic_ref}] = cubic_ref;
    next_state[{value_cubic_ref}] = value_cubic_ref;
    next_state[{safe_stepsize}] = safe_stepsize;
    next_state[{safe_value}] = safe_value;

    // The next trial, proposed from the state just written, as the next call of `step` would
    float bracketing_trial = {increase_factor} * t;
    if ({has_max}) bracketing_trial = fmin(bracketing_trial, {max_learning_rate});
    float next_width = fabs(new_high - new_low);
    float left = fmin(new_high, new_low);
    float right = fmax(new_high, new_low);
    float cubic = cubic_minimizer(
        new_low, new_value_low, new_slope_low, new_high, new_value_high, cubic_ref, value_cubic_ref
    );
    bool use_cubic = cubic > left + 0.2f * next_width && cubic < right - 0.2f * next_width;
    float quadratic = quadratic_minimizer(
        new_low, new_value_low, new_slope_low, new_high, new_value_high
    );
    bool use_quadratic =
        !use_cubic && quadratic > left + 0.1f * next_width && quadratic < right - 0.1f * next_width;
    float middle = use_cubic ? cubic : (use_quadratic ? quadratic : (new_low + new_high) / 2.0f);
    next_trial[0] = (done || failed) ? t : (interval_found ? middle : bracketing_trial);
"""


def _literal(value: float) -> str:
    return "INFINITY" if value == float("inf") else f"{float(value)!r}f"


@cache
def _zoom_kernel(search: ZoomLineSearch):
    """Build the fused update for one search configuration, which Metal compiles on first call."""
    source = _ZOOM_SOURCE.format(
        n_fields=len(_ZOOM_FIELDS),
        **_FIELD,
        increase_factor=_literal(search.increase_factor),
        has_max="true" if search.max_learning_rate is not None else "false",
        max_learning_rate=_literal(search.max_learning_rate or 0.0),
        slope_rtol=_literal(search.slope_rtol),
        curv_rtol=_literal(search.curv_rtol),
        has_approx="true" if search.approx_dec_rtol is not None else "false",
        approx_dec_rtol=_literal(search.approx_dec_rtol or 0.0),
        tol=_literal(search.tol),
        stepsize_precision=_literal(search.stepsize_precision),
        max_steps=_literal(search.max_steps),
    )
    return mx.fast.metal_kernel(
        name="pytensor_ml_zoom_line_search",
        input_names=["state", "trial", "anchor"],
        output_names=["next_state", "next_trial"],
        header=_ZOOM_HEADER,
        source=source,
    )


@mlx_funcify.register(LineSearchOp)
def mlx_funcify_LineSearchOp(op, node=None, **kwargs):
    """Run every trial the search allows, holding the state once a trial reports that it is done."""
    kwargs.pop("storage_map", None)
    # The pieces as the linker rewrote them, converted by the backend's own OpFromGraph dispatch
    start_op, step_op, trial_op, finish_op = (
        find_piece(op, name) for name in (START, STEP, TRIAL, FINISH)
    )
    start, step, trial, finish = (
        mlx_funcify(piece, **kwargs) for piece in (start_op, step_op, trial_op, finish_op)
    )
    search = op.line_search
    n_parameters = op.n_parameters
    declared = [variable.type.dtype for variable in finish_op.inner_inputs]
    field_dtypes = [
        getattr(mx, "bool_" if dtype == "bool" else dtype) for dtype in declared[: len(_FIELD)]
    ]
    # Decided from the declared dtype: mlx quietly computes a float64 graph at single precision on the
    # GPU, so the arrays reaching the loop would claim float32 for a search declared in float64.
    single_precision = declared[_FIELD["value"]] == "float32"
    # A zoom search at single precision on the GPU runs each trial's bookkeeping as one Metal kernel
    # between loss evaluations, chosen once from the default device when the function is built
    kernel = (
        _zoom_kernel(search)
        if isinstance(search, ZoomLineSearch)
        and single_precision
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        else None
    )

    # Stopping early would read the stop flag back to Python, which `mx.compile` refuses while it
    # traces, so both loops run all `max_steps` trials
    def stepped(inputs, state):
        done = mx.array(False)
        for _ in range(search.max_steps):
            *next_state, stop = step(*state, *inputs)
            state = [mx.where(done, held, taken) for held, taken in zip(state, next_state)]
            done = done | stop
        return state

    def fused(inputs, state):
        moved = (*inputs[: 2 * n_parameters], *inputs[2 * n_parameters + 3 :])
        value0, slope0, guess = inputs[2 * n_parameters : 2 * n_parameters + 3]
        anchor = mx.stack([value0, slope0, guess]).astype(mx.float32)
        packed = mx.stack([field.astype(mx.float32) for field in state[: len(_ZOOM_FIELDS)]])
        record = state[len(_ZOOM_FIELDS) :]
        stepsize = guess.astype(mx.float32)
        if search.max_learning_rate is not None:
            stepsize = mx.minimum(stepsize, search.max_learning_rate)
        for _ in range(search.max_steps):
            # Once the kernel holds a finished search it proposes the same step, so the last pass
            # evaluates the trial the search ended on.
            value, slope, *gradients = trial(stepsize, *moved)
            record = [stepsize, value, *gradients]
            packed, next_trial = kernel(
                inputs=[packed, mx.stack([stepsize, value, slope]).astype(mx.float32), anchor],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[packed.shape, (1,)],
                output_dtypes=[mx.float32, mx.float32],
            )
            stepsize = next_trial[0]
        fields = [packed[index].astype(dtype) for index, dtype in enumerate(field_dtypes)]
        return [*fields, *(new.astype(old.dtype) for new, old in zip(record, state[len(fields) :]))]

    def line_search(*inputs):
        value0, slope0 = inputs[2 * n_parameters : 2 * n_parameters + 2]
        directions = inputs[n_parameters : 2 * n_parameters]
        state = list(start(value0, slope0, *directions))
        state = stepped(inputs, state) if kernel is None else fused(inputs, state)
        return tuple(finish(*state))

    return line_search
