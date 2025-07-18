import functools
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from .helpers import extract, apply_conditioning, mse_loss, cosine_beta_schedule
from .model import TemporalUnet


###############################################################################
######################### Sample Function #####################################
###############################################################################


@functools.partial(jax.jit, static_argnums=(0,))
def ddpm_sample(
    diffusion: "GaussianDiffusion",
    rng: jax.random.PRNGKey,
    x: jnp.ndarray,
    cond: jnp.ndarray,
    t: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """
    One DDPM sampling step from x_t to x_{t-1}.

    Returns:
        jnp.ndarray: x_{t-1}
    """
    model_mean, _, model_log_variance = diffusion.p_mean_variance(x, cond, t, params)
    model_std = jnp.exp(0.5 * model_log_variance)
    noise = jax.random.normal(rng, shape=x.shape)
    noise = jnp.where((t == 0)[:, None, None], 0.0, noise)  # No noise at t=0
    x_next = model_mean + model_std * noise
    return x_next


###############################################################################
######################### Gaussian Diffusion Class ############################
###############################################################################


class GaussianDiffusion:
    def __init__(
        self,
        model_def: TemporalUnet,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        clip_denoised: bool = True,
        predict_epsilon: bool = False,
    ):
        """
        Implements Gaussian diffusion with DDPM-style training and sampling.

        Args:
            model_def (TemporalUnet): The denoising model.
            horizon (int): Time horizon of each trajectory.
            observation_dim (int): Dimensionality of observations.
            action_dim (int): Dimensionality of actions.
            n_timesteps (int): Number of diffusion steps.
            clip_denoised (bool): Whether to clip predicted x_0 to [-1, 1].
            predict_epsilon (bool): Whether model predicts noise instead of x_0.
        """
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model_def = model_def
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = n_timesteps

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones((1,)), alphas_cumprod[:-1]], axis=0
        )

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Calculations for diffusion q(x_t | x_{t-1}) and q(x_t | x_0)
        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance

        # Clipped posterior variance to avoid numerical issues
        self.posterior_log_variance_clipped = jnp.log(
            jnp.clip(posterior_variance, a_min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def predict_start_from_noise(
        self, x_t: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        """
        If predict_epsilon is True, predict (scaled) noise.
        Otherwise, predict the original x_0.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate the posterior mean and variance for q(x_{t-1} | x_t, x_0).
        x_start is the predicted x_0, x_t is the noisy observation at time t
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @functools.partial(jax.jit, static_argnums=(0,))
    def p_mean_variance(
        self, x: jnp.ndarray, cond: jnp.ndarray, t: jnp.ndarray, params: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate the mean and variance for p(x_{t-1} | x_t, x_0).

        x: the noisy observation at time t
        cond: the conditioning information (e.g., current observation)
        t: the current time step
        params: model parameters
        """
        noise_estimate = self.model_def.apply({"params": params}, x, cond, t)

        x_recon = self.predict_start_from_noise(x, t, noise_estimate)
        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x, t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @functools.partial(jax.jit, static_argnums=(0, 3, 5))
    def p_sample_loop(
        self,
        rng: jax.random.PRNGKey,
        params: jnp.ndarray,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        sample_fn: Callable = ddpm_sample,
    ) -> jnp.ndarray:
        """
        Reverse diffusion loop from noise to sample.
        """
        batch_size = shape[0]
        rng, rng_init = jax.random.split(rng)
        x = jax.random.normal(rng_init, shape)
        x = apply_conditioning(x, cond, self.action_dim)

        def loop_body(carry, diffusion_step):
            x_curr, rng_curr = carry
            t = jnp.full((batch_size,), diffusion_step, dtype=jnp.int32)
            rng_curr, rng_step = jax.random.split(rng_curr)
            x_next = sample_fn(self, rng_step, x_curr, cond, t, params)
            x_next = apply_conditioning(x_next, cond, self.action_dim)
            return (x_next, rng_curr), None

        reverse_timesteps = jnp.arange(self.n_timesteps - 1, -1, -1)
        (x_final, _), _ = jax.lax.scan(loop_body, (x, rng), reverse_timesteps)
        return x_final

    def conditional_sample(
        self,
        rng: jax.random.PRNGKey,
        params: jnp.ndarray,
        cond: jnp.ndarray,
        horizon: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Generates a sample conditined on the observation.
        """
        horizon = horizon or self.horizon
        shape = (cond.shape[0], horizon, self.transition_dim)
        return self.p_sample_loop(rng, params, shape, cond)

    ####################################################################
    ######################## Training Utilities
    ####################################################################

    def q_sample(
        self, x_start: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Sample from the diffusion process q(x_t | x_0).
        x_start: the original observation (x_0)
        t: the time step
        noise: random noise to add
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def p_loss(
        self,
        rng: jax.random.PRNGKey,
        x_start: jnp.ndarray,
        cond: jnp.ndarray,
        mask: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the loss for the diffusion process.
        rng: random number generator key
        x_start: the original observation (x_0)
        cond: the conditioning information (e.g., current observation)
        mask: mask to apply to the loss
        params: model parameters
        """

        rng, rng_t, rng_noise = jax.random.split(rng, 3)
        t = jax.random.randint(rng_t, (x_start.shape[0],), 0, self.n_timesteps)
        noise = jax.random.normal(rng_noise, shape=x_start.shape)
        x_noisy = self.q_sample(x_start, t, noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model_def.apply({"params": params}, x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        if self.predict_epsilon:
            loss = mse_loss(x_recon, noise, ~mask)
        else:
            loss = mse_loss(x_recon, x_start, ~mask)

        return loss
