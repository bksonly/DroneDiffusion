import os

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import sys
import argparse

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logger import setup_logger, log_config
from utils.config import load_config, ConfigError
from diffusion.core import GaussianDiffusion
from diffusion.model import TemporalUnet
from data.dataset import DiffusionTrajDataset
from data.loader import DataLoader
from training.train_state import create_train_state
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gaussian Diffusion model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., experiment.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)
        from pathlib import Path

        config.workdir = str(Path(config.workdir).expanduser().resolve())
    except ConfigError as e:
        print(f"[ERROR] Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    setup_logger(log_file=os.path.join(config.workdir, "training.log"))
    log_config(config)

    # Seed and PRNG setup
    rng = jax.random.PRNGKey(config.seed)
    rng, loader_rng, init_rng = jax.random.split(rng, 3)

    ds = DiffusionTrajDataset(
        dataset_path=config.dataset_path,
        horizon=config.horizon,
        max_path_length=config.max_path_length,
    )

    loader = DataLoader(
        dataset=ds,
        batch_size=config.batch_size,
        shuffle=True,
        rng_key=loader_rng,
        prefetch=config.prefetch_batches,
    )

    model = TemporalUnet(
        horizon=config.horizon,
        transition_dim=ds.transition_dim,
        cond_dim=ds.obs_dim,
        dim=config.dim,
        dim_mults=config.dim_mults,
        attention=config.attention,
    )

    diffusion = GaussianDiffusion(
        model_def=model,
        horizon=config.horizon,
        observation_dim=ds.obs_dim,
        action_dim=ds.action_dim,
        n_timesteps=config.n_timesteps,
        clip_denoised=config.clip_denoised,
        predict_epsilon=config.predict_epsilon,
    )

    # Grab a sample batch for shape inference
    batch0 = next(iter(loader))
    sample_x = batch0.trajectories
    sample_cond = batch0.conditions
    sample_t = jnp.zeros((sample_x.shape[0],), dtype=jnp.int32)

    state = create_train_state(
        rng=init_rng,
        diffusion=diffusion,
        sample_x=sample_x,
        sample_cond=sample_cond,
        sample_time=sample_t,
        learning_rate=config.learning_rate,
        ema_decay=config.ema_decay,
        grad_accum_steps=config.grad_accum_steps,
        update_ema_every=config.update_ema_every,
        warmup_steps=config.warmup_steps,
    )

    trainer = Trainer(
        diffusion=diffusion,
        state=state,
        train_loader=loader,
        workdir=config.workdir,
        total_steps=config.total_steps,
        log_every=config.log_every,
        save_every=config.save_every,
        resume=config.resume,
    )
    trainer.train(rng=rng)


if __name__ == "__main__":
    main()
