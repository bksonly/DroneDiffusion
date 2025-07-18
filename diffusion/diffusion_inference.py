import logging
import sys
from functools import partial

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from .core import GaussianDiffusion
from .model import TemporalUnet
from data.dataset import DiffusionTrajDataset
from utils.config import load_config, ConfigError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Policy:
    """
    A drift/policy wrapper that loads a Gaussian Diffusion model
    and exposes it as a callable dynamics model for the controller.
    """

    def __init__(
        self,
        config: str,
        use_ema: bool = True,
        rng: int = 0,
    ):
        """
        config:    Config file to use.
        use_ema:   Whether to use EMA weights or the raw model weights.
        rng:       PRNG key for sampling.
        """
        try:
            config = load_config(config)
            from pathlib import Path

            config.workdir = str(Path(config.workdir).expanduser().resolve())
        except ConfigError as e:
            print(f"[ERROR] Failed to load configuration: {e}", file=sys.stderr)
            sys.exit(1)

        self.use_ema = use_ema
        self.rng = jax.random.PRNGKey(rng)

        ds = DiffusionTrajDataset(
            dataset_path=config.dataset_path,
            horizon=config.horizon,
            max_path_length=config.max_path_length,
        )
        self.action_dim = ds.action_dim
        self.normalizer = ds.normalizer
        self.normalizer.to_jax()

        self.model_def = TemporalUnet(
            horizon=config.horizon,
            transition_dim=ds.transition_dim,
            cond_dim=ds.obs_dim,
            dim=config.dim,
            dim_mults=config.dim_mults,
            attention=config.attention,
        )

        self.diffusion = GaussianDiffusion(
            model_def=self.model_def,
            horizon=config.horizon,
            observation_dim=ds.obs_dim,
            action_dim=ds.action_dim,
            n_timesteps=config.n_timesteps,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
        )

        # Load model checkpoint
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.workdir,
            target=None,
        )
        self.params = ckpt["ema_params"] if use_ema else ckpt["params"]

        logging.info(
            f"Loaded {'EMA' if use_ema else 'raw'} parameters "
            f"from step {ckpt.get('step', 'unknown')}."
        )

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.normalizer.normalize_observation(x)
        x = jnp.expand_dims(x, axis=0)  # Add batch dimension

        self.rng, sample_rng = jax.random.split(self.rng)
        sample = self.diffusion.conditional_sample(
            rng=sample_rng,
            params=self.params,
            cond=x,
            horizon=self.diffusion.horizon,
        ).squeeze(0)[
            :, : self.action_dim
        ]  # Remove batch dimension

        return self.normalizer.unnormalize_action(sample)
