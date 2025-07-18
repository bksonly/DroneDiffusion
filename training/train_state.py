import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state

from diffusion.core import GaussianDiffusion


@struct.dataclass
class TrainStateEMA(train_state.TrainState):
    """
    Extension of Flax's TrainState to support:
      - Exponential Moving Average (EMA) of parameters
      - Gradient accumulation over multiple steps
    """

    ema_params: FrozenDict
    ema_step: int = struct.field(pytree_node=False)

    # EMA throttling
    warmup_steps: int = struct.field(pytree_node=False)
    update_every: int = struct.field(pytree_node=False)

    # Accumulation config
    grad_accum_steps: int = struct.field(pytree_node=False)
    grad_accum_count: int = struct.field(pytree_node=False)
    grad_accum: FrozenDict


def create_train_state(
    rng: jax.random.PRNGKey,
    diffusion: GaussianDiffusion,
    sample_x: jnp.ndarray,
    sample_cond: jnp.ndarray,
    sample_time: jnp.ndarray,
    learning_rate: float,
    ema_decay: float = 0.9999,
    grad_accum_steps: int = 1,
    update_ema_every: int = 1,
    warmup_steps: int = 1000,
) -> TrainStateEMA:
    """
    Build a TrainStateEMA that holds:
      - params:         the parameters of diffusion.model_def
      - apply_fn:       a function that delegates to diffusion.p_loss
      - tx / opt_state: your optimizer (with gradient clipping)
      - ema_params:     an exponential-moving average copy of params
      - ema_step:       the step size for EMA updates
      - grad_accum:     zero-initialized tree for gradient accumulation
      - warmup / throttling for EMA
    """

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to avoid exploding
        optax.adam(learning_rate=learning_rate, eps=1e-8),
    )
    # Initialize model parameters
    params = diffusion.model_def.init(rng, sample_x, sample_cond, sample_time)["params"]
    zero_grad = jax.tree_map(jnp.zeros_like, params)

    loss_fn = lambda params, rng, x, cond, mask: diffusion.p_loss(
        rng, x, cond, mask, params
    )

    return TrainStateEMA.create(
        apply_fn=loss_fn,
        params=params,
        tx=tx,
        ema_params=params,
        ema_step=1 - ema_decay,
        grad_accum_steps=grad_accum_steps,
        grad_accum_count=0,
        grad_accum=zero_grad,
        update_every=update_ema_every,
        warmup_steps=warmup_steps,
    )
