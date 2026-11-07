import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.optim.line_search import FINISH, START, STEP, LineSearchOp, find_piece


@jax_funcify.register(LineSearchOp)
def jax_funcify_LineSearchOp(op, node=None, **kwargs):
    """Run the search's ``step`` piece in a ``lax.while_loop`` that stops when ``step`` does."""
    kwargs.pop("storage_map", None)
    start, step, finish = (
        jax_funcify(find_piece(op, name), **kwargs) for name in (START, STEP, FINISH)
    )
    n_parameters = op.n_parameters
    max_steps = op.line_search.max_steps

    def line_search(*inputs):
        value0, slope0 = inputs[2 * n_parameters : 2 * n_parameters + 2]
        directions = inputs[n_parameters : 2 * n_parameters]
        initial = tuple(jnp.asarray(field) for field in start(value0, slope0, *directions))

        def keep_going(carry):
            iteration, stop, _ = carry
            return ~stop & (iteration < max_steps)

        def take_step(carry):
            iteration, _, state = carry
            *next_state, stop = step(*state, *inputs)
            # The loop's carry has to keep its types, which the step's outputs need not match exactly
            next_state = tuple(
                jnp.asarray(new, dtype=old.dtype) for new, old in zip(next_state, state)
            )
            return iteration + 1, jnp.asarray(stop, dtype=bool), next_state

        _, _, final = jax.lax.while_loop(
            keep_going, take_step, (jnp.asarray(0), jnp.asarray(False), initial)
        )
        return finish(*final)

    return line_search
