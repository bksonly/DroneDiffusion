import jax
import jax.numpy as jnp
from typing import Tuple


######################################################################################
######################## Index Extractions and Conditioning ##########################
######################################################################################
def extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Tuple[int, ...]) -> jnp.ndarray:
    """
    Extracts values from `a` at indices `t`, and reshapes to match `x_shape`
    for broadcasting.

    Args:
        a: [T] or [T, ...] array from which to extract.
        t: [B] array of time indices.
        x_shape: shape of the tensor that will be broadcasted to.

    Returns:
        Extracted values reshaped to [B, 1, ..., 1] (with len(x_shape) - 1 dims).
    """
    val = a[t]  # [B]
    expand_shape = len(x_shape) - 1
    return jnp.reshape(val, (val.shape[0],) + (1,) * expand_shape)


def apply_conditioning(
    x: jnp.ndarray, conditions: jnp.ndarray, action_dim: int
) -> jnp.ndarray:
    """
    Overwrites the observation portion of the first timestep in each trajectory.

    Args:
        x: [B, H, D] - full trajectory (actions + observations).
        conditions: [B, obs_dim] - conditioning observations.
        action_dim: Index where observation features start in x.

    Returns:
        jnp.ndarray: Modified trajectory with conditions applied at t=0.
    """
    return x.at[:, 0, action_dim:].set(conditions)


######################################################################################
######################## Diffusion Schedules #########################################
######################################################################################


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Cosine beta schedule from "Improved DDPM" paper:
    https://openreview.net/forum?id=-NEXDKk8gZ

    Args:
        timesteps: Number of diffusion steps.
        s: Small offset for stability.
        dtype: Output dtype.

    Returns:
        jnp.ndarray: Beta values for each timestep [T].
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps, dtype=dtype) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, a_min=0.0, a_max=0.999)


#######################################################################################
######################### Loss Functions ##############################################
#######################################################################################


@jax.jit
def mse_loss(pred, targ, mask):
    """
    pred, targ: [batch, horizon, transition_dim]
    mask: [batch, horizon]
    """
    loss = jnp.mean((pred - targ) ** 2, axis=-1)  # [batch, horizon]
    loss = jnp.where(mask, loss, 0.0)  # Apply mask
    loss = jnp.mean(loss)  # Average over batch and horizon
    return loss
