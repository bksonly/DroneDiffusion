import queue
import threading
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import device_put

from .dataset import DiffusionTrajDataset


# -------------------------------------------------------------
# Named tuple for batch data
# -------------------------------------------------------------
@struct.dataclass
class Batch:
    """
    A named container to hold a batch of data.

    Attributes:
        trajectories (jnp.ndarray): Batched trajectories of shape (B, H, D).
        conditions (jnp.ndarray): Batched conditions of shape (B, obs_dim).
        mask (jnp.ndarray): Early termination mask of shape (B, H).
    """

    trajectories: jnp.ndarray
    conditions: jnp.ndarray
    mask: jnp.ndarray


# -------------------------------------------------------------
# JAX Compatible Dataloader over pre-sliced dataset
# -------------------------------------------------------------
class DataLoader:
    def __init__(
        self,
        dataset: DiffusionTrajDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        rng_key: Optional[jax.random.PRNGKey] = None,
        prefetch: int = 2,
        drop_last: bool = True,
    ):
        """
        Initializes the DataLoader with a dataset and parameters for batching.

        Args:
            dataset (DiffusionTrajDataset): The dataset to load from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
            rng_key (jax.random.PRNGKey): Random key for shuffling.
            prefetch (int): Number of batches to prefetch.
            drop_last (bool): If True, drops the last incomplete batch.
        """
        self.ds = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        self.N = len(dataset)
        self.prefetch = prefetch
        self.drop_last = drop_last

        # Will hold the queue and background thread once __iter__ is called
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._sentinel = None  # Sentinel value to signal end of iteration

    def __iter__(self) -> Iterator[Batch]:
        # Compute the order of indices
        idxs = np.arange(self.N, dtype=np.int32)
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            idxs = np.array(jax.random.permutation(subkey, self.N), dtype=np.int32)

        max_i = (
            (self.N // self.batch_size) * self.batch_size if self.drop_last else self.N
        )

        # Set up queue and sentinel
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()
        self._queue = q
        self._sentinel = sentinel

        # Producer thread: slice, device_put, and enqueue batches
        def producer():
            for i0 in range(0, max_i, self.batch_size):
                i1 = i0 + self.batch_size
                batch_idx = idxs[i0:i1]

                # Slice on host (NumPy)
                host_trajs = self.ds.all_trajs[batch_idx]  # (B, H, D)
                host_masks = self.ds.all_masks[batch_idx]  # (B, H)
                host_conds = self.ds.all_conds[batch_idx]  # (B, obs_dim)

                # Push *just this batch* to device
                batch_dev = Batch(
                    trajectories=device_put(host_trajs),
                    conditions=device_put(host_conds),
                    mask=device_put(host_masks),
                )
                q.put(batch_dev)
            q.put(sentinel)  # Signal end of iteration

        # Start the producer thread
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        self._thread = thread

        return self

    def __next__(self) -> Batch:
        """
        Fetches the next batch from the queue.

        Returns:
            Batch: A named tuple containing trajectories, conditions, and mask.

        Raises:
            StopIteration: If no more batches are available.
            RuntimeError: If the DataLoader has not been initialized.
        """
        if self._queue is None or self._thread is None:
            raise RuntimeError("DataLoader not initialized. Call __iter__() first.")

        batch = self._queue.get()
        if batch is self._sentinel:
            raise StopIteration

        return batch
