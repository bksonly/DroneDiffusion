from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]


# -------------------------------------------------------------
# Limit Normalization
# -------------------------------------------------------------
class LimitNormalizer:
    """
    Normalizes and unnormalizes actions and observations to and from [-1, 1] range
    using known min/max limits for each.

    Supports both NumPy and JAX arrays.
    """

    def __init__(
        self, action_max: Array, action_min: Array, obs_max: Array, obs_min: Array
    ):
        """
        Initializes the normalizer with bounds for actions and observations.

        Args:
            action_max (Array): Maximum action values (D_a,).
            action_min (Array): Minimum action values (D_a,).
            obs_max (Array): Maximum observation values (D_o,).
            obs_min (Array): Minimum observation values (D_o,).
        """
        self.action_max = action_max
        self.action_min = action_min
        self.observation_max = obs_max
        self.observation_min = obs_min

    def normalize_action(self, action: Array) -> Array:
        """
        Normalizes the action to the range [-1, 1].

        Args:
            action (Array): Action to normalize.

        Returns:
            Array: Normalized action.
        """
        return (action - self.action_min) / (self.action_max - self.action_min) * 2 - 1

    def unnormalize_action(self, action: Array) -> Array:
        """
        Unnormalizes the action back to its original range.

        Args:
            action (Array): Action to unnormalize.

        Returns:
            Array: Unnormalized action.
        """
        return (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def normalize_observation(self, observation: Array) -> Array:
        # Same as normalize_action but for observations
        return (observation - self.observation_min) / (
            self.observation_max - self.observation_min
        ) * 2 - 1

    def unnormalize_observation(self, observation: Array) -> Array:
        # Same as unnormalize_action but for observations
        return (observation + 1) / 2 * (
            self.observation_max - self.observation_min
        ) + self.observation_min

    def normalize(self, action: Array, observation: Array) -> Tuple[Array, Array]:
        """
        Normalizes both actions and observations.

        Args:
            action (Array): Raw action.
            observation (Array): Raw observation.

        Returns:
            Tuple[Array, Array]: Normalized (action, observation).
        """
        return self.normalize_action(action), self.normalize_observation(observation)

    def unnormalize(self, action: Array, observation: Array) -> Tuple[Array, Array]:
        """
        Unnormalizes both actions and observations.

        Args:
            action (Array): Normalized action.
            observation (Array): Normalized observation.

        Returns:
            Tuple[Array, Array]: Original-scale (action, observation).
        """
        return self.unnormalize_action(action), self.unnormalize_observation(
            observation
        )

    def to_jax(self) -> "LimitNormalizer":
        """
        Converts all internal limits to JAX arrays (for compatibility with JAX operations).

        Returns:
            LimitNormalizer: Self with JAX-converted bounds.
        """
        self.action_max = jnp.array(self.action_max)
        self.action_min = jnp.array(self.action_min)
        self.observation_max = jnp.array(self.observation_max)
        self.observation_min = jnp.array(self.observation_min)
        return self
