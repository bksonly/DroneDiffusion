import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Union


def setup_logger(log_file: Optional[str] = None, log_level: int = logging.INFO):
    """
    Configure the root logger with optional file output.

    Args:
        log_file: Optional path to a log file.
        log_level: Logging level (e.g., logging.INFO).
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Stream (console) handler
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Optional file handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)


def format_config(config: Union[Mapping[str, Any], Any]) -> str:
    """
    Format a configuration object into a pretty multi-line string.

    Args:
        config: A dataclass, dictionary, or similar config object.

    Returns:
        str: Formatted string of key-value pairs.
    """
    cfg_dict = (
        asdict(config)
        if is_dataclass(config)
        else dict(config) if isinstance(config, Mapping) else {}
    )
    # compute padding for alignment
    width = max(len(k) for k in cfg_dict)
    lines = []
    for k, v in cfg_dict.items():
        if isinstance(v, (list, tuple)):
            v_str = "[" + ", ".join(str(x) for x in v) + "]"
        else:
            v_str = str(v)
        lines.append(f"{k.ljust(width)} : {v_str}")
    return "\n".join(lines)


def log_config(config: Any, logger: Optional[logging.Logger] = None):
    """Log config via the root logger (or given one) at INFO."""
    if logger is None:
        logger = logging.getLogger()
    logger.info("Loaded experiment configuration:\n%s", format_config(config))
