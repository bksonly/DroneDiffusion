import logging
import os
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

from data.loader import DataLoader, Batch
from diffusion.core import GaussianDiffusion
from .train_state import TrainStateEMA


def cycle(dl: DataLoader) -> Iterator[Batch]:
    """
    Creates an infinite iterator over the DataLoader.

    Args:
        dl (DataLoader): A finite-length dataloader.

    Yields:
        Batch: The next batch, cycling indefinitely.
    """
    while True:
        for batch in dl:
            yield batch


@jax.jit
def train_step(
    state: TrainStateEMA, batch: Batch, rng: jax.random.PRNGKey
) -> Tuple[TrainStateEMA, jnp.ndarray]:
    """
    Performs a single training step with gradient accumulation and EMA update.

    Args:
        state: Current training state.
        batch: One batch of training data.
        rng: PRNG key.

    Returns:
        Tuple of updated training state and scalar loss.
    """

    # Compute this-batch gradients
    loss, grads = jax.value_and_grad(state.apply_fn)(
        state.params, rng, batch.trajectories, batch.conditions, batch.mask
    )

    # Accumulate gradients
    accum = jax.tree_map(lambda a, g: a + g, state.grad_accum, grads)
    count = state.grad_accum_count + 1

    # Decide whether to apply accumulated gradients
    if count >= state.grad_accum_steps:
        # Scale grads down by the number of accumulation steps
        scaled = jax.tree_map(lambda g: g / state.grad_accum_steps, accum)
        # Apply optimizer updates
        new_state = state.apply_gradients(grads=scaled)

        # EMA update (warmup + throttling)
        do_ema = (new_state.step >= new_state.warmup_steps) & (
            (new_state.step - new_state.warmup_steps) % new_state.update_every == 0
        )

        def do_fn(ema_old):
            return optax.incremental_update(
                new_state.params, ema_old, new_state.ema_step
            )

        new_ema = jax.lax.cond(do_ema, do_fn, lambda e: e, new_state.ema_params)

        # reset accumulation
        zero = jax.tree_map(jnp.zeros_like, state.grad_accum)
        state = new_state.replace(
            ema_params=new_ema,
            grad_accum=zero,
            grad_accum_count=0,
        )
    else:
        # If we haven't hit the threshold, just update the state
        state = state.replace(
            grad_accum=accum,
            grad_accum_count=count,
        )

    return state, loss


class Trainer:
    def __init__(
        self,
        diffusion: GaussianDiffusion,
        state: TrainStateEMA,
        train_loader: DataLoader,
        workdir: str,
        total_steps: int,
        log_every: int = 100,
        save_every: int = 1000,
        resume: bool = False,
    ):
        """
        Initializes the training loop.

        Args:
            diffusion: The GaussianDiffusion model.
            state: Initial TrainStateEMA.
            train_loader: Iterable of training batches.
            workdir: Directory to store logs and checkpoints.
            total_steps: Total training steps.
            log_every: Log frequency (steps).
            save_every: Checkpoint save frequency (steps).
            resume: Whether to load latest checkpoint.
        """
        self.diffusion = diffusion
        self.state = state
        self.loader = cycle(train_loader)
        self.workdir = workdir
        self.total_steps = total_steps
        self.log_every = log_every
        self.save_every = save_every

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Trainer initialized with workdir: {self.workdir}")

        # Resume training if specified
        if resume:
            restored = checkpoints.restore_checkpoint(self.workdir, target=self.state)
            if int(restored.step) > 0:
                self.start_step = int(self.state.step)
                self.logger.info(f"Resuming training from step {self.start_step}")
            else:
                self.start_step = 0
                self.logger.info("No valid checkpoint found, starting from scratch.")
        else:
            self.start_step = 0
            self.logger.info("Starting training from scratch.")

    def _save_checkpoint(self, step: int):
        """Save the current state to a checkpoint."""
        ckpt_dir = os.path.join(self.workdir)
        checkpoints.save_checkpoint(
            ckpt_dir, target=self.state, step=step, overwrite=True
        )
        self.logger.info(f"[Checkpoint] Saved checkpoint at step {step}.")

    def train(self, rng: jax.random.PRNGKey):
        """Main training loop."""
        state = self.state

        for step in range(self.start_step, self.total_steps):
            rng, step_rng = jax.random.split(rng)
            batch = next(self.loader)
            state, loss = train_step(state, batch, step_rng)

            if (step + 1) % self.log_every == 0:
                self.logger.info(
                    f"Step {step + 1}/{self.total_steps}, Loss: {loss:.4f}"
                )

            if (step + 1) % self.save_every == 0:
                self.state = state
                self._save_checkpoint(step + 1)

        # Final save
        self.state = state
        self._save_checkpoint(self.total_steps)
        self.logger.info("Training completed. Final checkpoint saved.")
