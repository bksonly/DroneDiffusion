import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Type

import yaml


class ConfigError(Exception):
    """Raised if the YAML is missing a required key or has the wrong type."""

    pass


@dataclass
class Config:
    # -------------------------------------
    # Dataset and Loader Configuration
    # -------------------------------------
    dataset_path: str
    horizon: int
    max_path_length: int
    batch_size: int

    # -------------------------------------
    # Model Configuration
    # -------------------------------------
    dim: int
    attention: bool
    dim_mults: list[int]

    # -------------------------------------
    # Diffision Configuration
    # -------------------------------------
    n_timesteps: int
    predict_epsilon: bool
    clip_denoised: bool

    # -------------------------------------
    # Training Configuration
    # -------------------------------------
    total_steps: int
    learning_rate: float
    grad_accum_steps: int
    ema_decay: float
    update_ema_every: int
    warmup_steps: int
    prefetch_batches: int
    seed: int

    # -------------------------------------
    # Logging & Checkpointing Configuration
    # -------------------------------------
    workdir: str
    log_every: int
    save_every: int
    resume: bool


def _cast_value(value: Any, target_type: Type) -> Any:
    """
    Attempts to cast 'value' to 'target_type'. Raises ConfigError on failure.

    Args:
        value: The raw value from YAML.
        target_type: The expected type.

    Returns:
        The casted value.

    Raises:
        ConfigError: If casting fails.
    """
    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.lower()
                if low in ["true", "yes", "1"]:
                    return True
                if low in ["false", "no", "0"]:
                    return False
            raise ValueError(f"Cannot cast {value} to {target_type.__name__}")

        # For other types, use the built-in constructor
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise ConfigError(
            f"Failed to cast value '{value}' to type '{target_type.__name__}': {e}"
        )


def load_config(config_path: str) -> Config:
    """
    Load a YAML config file and return a validated Config object.

    Args:
        config_path: Path to the YAML file (relative to 'config/' directory).

    Returns:
        Config: A fully constructed and validated Config dataclass.

    Raises:
        ConfigError: On missing keys or invalid types.
    """
    full_path = os.path.join("config", config_path)
    try:
        raw_config = yaml.safe_load(Path(full_path).read_text())
    except Exception as e:
        raise ConfigError(f"Failed to read or parse config file: {e}")

    if not isinstance(raw_config, dict):
        raise ConfigError("Expected top-level YAML structure to be a dictionary.")

    init_kwargs = {}
    missing_keys = []

    for field in fields(Config):
        key = field.name
        typ = field.type

        if key not in raw_config:
            missing_keys.append(key)
            continue

        raw_val = raw_config[key]
        init_kwargs[key] = _cast_value(raw_val, typ)

    if missing_keys:
        raise ConfigError(f"Missing required keys in config: {', '.join(missing_keys)}")

    return Config(**init_kwargs)
