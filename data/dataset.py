import logging
import os
import pickle
from typing import List, Tuple

import numpy as np

from .normalization import LimitNormalizer


# -------------------------------------------------------------
# Segment function
# -------------------------------------------------------------
def segment(
    observations: np.ndarray, terminals: np.ndarray, max_path_length: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Segments a stream of observations+actions and terminals into padded trajectories,
    with corresponding early-termination masks.

    Args:
        observations (np.ndarray): Array of shape (N, dim).
        terminals (np.ndarray): Boolean flags indicating episode termination (N,).
        max_path_length (int): Max length for trajectory padding.

    Returns:
        Tuple containing:
            - Padded trajectories (np.ndarray): (P, max_path_length, dim).
            - Early termination masks (np.ndarray): (P, max_path_length), True after terminal.
            - List of actual lengths of each trajectory.
    """

    assert len(observations) == len(
        terminals
    ), "Observations and terminals must have the same length."

    obs_dim = observations.shape[1]

    # Split into list of trajectories
    trajs = [[]]
    for obs, term in zip(observations, terminals):
        trajs[-1].append(obs)
        if term.squeeze():
            trajs.append([])
    if len(trajs[-1]) == 0:
        trajs.pop()
    trajs = [np.stack(traj, axis=0) for traj in trajs]

    # Pad trajectories and build early-termination masks
    n_paths = len(trajs)
    path_lengths = [len(traj) for traj in trajs]

    pad_trajs = np.zeros((n_paths, max_path_length, obs_dim), dtype=observations.dtype)
    early_term = np.zeros((n_paths, max_path_length), dtype=bool)
    for i, traj in enumerate(trajs):
        L = path_lengths[i]
        pad_trajs[i, :L] = traj
        early_term[i, L:] = True

    return pad_trajs, early_term, path_lengths


# -------------------------------------------------------------
# Dataset class for diffusion trajectories
# -------------------------------------------------------------
class DiffusionTrajDataset:
    def __init__(
        self,
        dataset_path: str = "data",
        horizon: int = 16,
        max_path_length: int = 1000,
    ):
        """
        Initializes the dataset by loading trajectories and conditions from a pickle file.

        Args:
            dataset_path (str): Path to the dataset file.
            horizon (int): The number of timesteps in each trajectory.
            max_path_length (int): Maximum length of paths for padding.
        """
        logging.info(
            f"[Dataset] horizon: {horizon}, max_path_length: {max_path_length}"
        )
        self.horizon = horizon
        self.max_path_length = max_path_length

        try:
            with open(os.path.join(dataset_path, "data.pkl"), "rb") as f:
                ds = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found at {dataset_path}/data.pkl"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        ds = {k: v.astype(np.float32) for k, v in ds.items()}
        obs = ds["states"]
        acts = ds["actions"]
        terms = ds["terminal"]
        if acts.ndim == 1:
            acts = acts[:, None]

        # Normalize observations and actions
        a_max, a_min = acts.max(axis=0), acts.min(axis=0)
        o_max, o_min = obs.max(axis=0), obs.min(axis=0)
        self.normalizer = LimitNormalizer(a_max, a_min, o_max, o_min)
        acts_n, obs_n = self.normalizer.normalize(acts, obs)

        # Join and segment trajectories
        joined = np.concatenate([acts_n, obs_n], axis=-1)
        joined_seg, term_flags, path_lengths = segment(joined, terms, max_path_length)

        # Build flat indices of all valid (path, start, start+horizon) tuples
        indices = []
        for p, L in enumerate(path_lengths):
            end = L - 1
            for i in range(end):
                if i + self.horizon <= max_path_length:
                    indices.append((p, i))
        self.indices = np.array(indices, dtype=np.int32)  # (N_idx, 2)

        # Pad trajectories so slicing to start+horizon never goes OOB
        Np, T, D = joined_seg.shape
        pad_with = horizon - 1
        joined_seg = np.concatenate(
            [joined_seg, np.zeros((Np, pad_with, D), dtype=joined_seg.dtype)], axis=1
        )
        term_flags = np.concatenate(
            [term_flags, np.ones((Np, pad_with), dtype=bool)], axis=1
        )

        # Pre-slice into one big array
        path_ids = self.indices[:, 0]  # (N_idx,)
        start_ids = self.indices[:, 1]  # (N_idx,)
        offsets = np.arange(horizon, dtype=np.int32)  # (horizon,)

        time_idxs = start_ids[:, None] + offsets[None, :]  # (N_idx, horizon)

        all_trajs = joined_seg[
            path_ids[:, None],  # (N_idx, 1) -> Broadcast to (N_idx, horizon)
            time_idxs,  # (N_idx, horizon)
            :,  # all D dimensions
        ]  # (N_idx, horizon, D)

        all_masks = term_flags[path_ids[:, None], time_idxs].astype(
            bool
        )  # (N_idx, horizon)

        # Conditions: first observation in each trajectory slice
        action_dim = acts_n.shape[1]
        all_conds = all_trajs[:, 0, action_dim:]  # (N_idx, obs_dim)

        self.all_trajs = all_trajs  # np.ndarray, shape (N_idx, H, D)
        self.all_masks = all_masks  # np.ndarray, shape (N_idx, H)
        self.all_conds = all_conds  # np.ndarray, shape (N_idx, obs_dim)

        # dims for downstream use
        self.obs_dim = obs.shape[1]
        self.action_dim = acts.shape[1]
        self.transition_dim = self.obs_dim + self.action_dim
        self.joined_dim = D

    def __len__(self) -> int:
        return self.all_trajs.shape[0]
