# Copyright (c) DeepGen. All rights reserved.
"""
Weighted Infinite Sampler for ConcatDataset with per-dataset sample weights.

This module implements a weighted infinite sampler that supports:
- Per-dataset sample weights (e.g., sample_weight=2.0 means 2x probability compared to uniform)
- Distributed training (shards across GPUs)
- Infinite sampling (never stops)

Usage in mmengine config:
    # Option 1: sample_weight in each dataset dict (recommended)
    dataset_a = dict(type=MyDataset, ..., sample_weight=1.0)
    dataset_b = dict(type=MyDataset, ..., sample_weight=3.0)

    # Option 2: sample_weights list in sampler config
    sampler=dict(type=WeightedInfiniteSampler, sample_weights=[1.0, 3.0], shuffle=True)
"""
import itertools
from typing import Iterator, List, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class WeightedInfiniteSampler(Sampler):
    """
    Weighted Infinite Sampler for ConcatDataset.

    Supports per-dataset sample weights where weight=1.0 is uniform sampling,
    weight=2.0 means 2x probability (like duplicating the dataset), etc.

    Sample weights can be specified in two ways:
    1. In each dataset's config dict as 'sample_weight' attribute (recommended)
    2. As a list passed to sampler's 'sample_weights' parameter

    Args:
        dataset: A ConcatDataset with cumulative_sizes attribute.
        sample_weights: Optional list of weights, one per dataset.
                       If None, will try to read from each dataset's sample_weight attribute.
                       Default is uniform (all 1.0) if neither is specified.
        shuffle: Whether to shuffle indices. Default True.
        seed: Random seed. If None, syncs across processes.
    """

    def __init__(
        self,
        dataset: Sized,
        sample_weights: Optional[List[float]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # Get distributed info
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = sync_random_seed() if seed is None else seed

        # Get dataset structure
        if hasattr(dataset, 'cumulative_sizes'):
            # ConcatDataset
            self.cumulative_sizes = [0] + list(dataset.cumulative_sizes)
            self.num_datasets = len(dataset.cumulative_sizes)
            self.dataset_lengths = [
                self.cumulative_sizes[i + 1] - self.cumulative_sizes[i]
                for i in range(self.num_datasets)
            ]
            self.sub_datasets = dataset.datasets
        else:
            # Single dataset
            self.cumulative_sizes = [0, len(dataset)]
            self.num_datasets = 1
            self.dataset_lengths = [len(dataset)]
            self.sub_datasets = [dataset]

        # Parse sample weights - priority: explicit list > dataset attribute > default 1.0
        self.sample_weights = self._parse_sample_weights(sample_weights)

        # Build per-sample weight tensor for weighted sampling
        self._build_sample_weights()

    def _parse_sample_weights(self, sample_weights: Optional[List[float]]) -> List[float]:
        """
        Parse sample weights from explicit list or dataset attributes.

        Priority:
        1. Explicit sample_weights list passed to sampler
        2. sample_weight attribute in each dataset
        3. Default value of 1.0
        """
        if sample_weights is not None:
            # Explicit list provided
            if len(sample_weights) != self.num_datasets:
                raise ValueError(
                    f"sample_weights length ({len(sample_weights)}) must match "
                    f"number of datasets ({self.num_datasets})"
                )
            weights = [float(w) for w in sample_weights]
        else:
            # Try to read from each dataset's sample_weight attribute
            weights = []
            for i, ds in enumerate(self.sub_datasets):
                if hasattr(ds, 'sample_weight'):
                    w = float(ds.sample_weight)
                else:
                    w = 1.0  # Default weight
                weights.append(w)

        # Validate weights
        for i, w in enumerate(weights):
            if w <= 0:
                raise ValueError(f"sample_weight must be positive, got {w} for dataset {i}")

        return weights

    def _build_sample_weights(self) -> None:
        """Build per-sample weight tensor for multinomial sampling."""
        weights = []
        for i, (length, weight) in enumerate(zip(self.dataset_lengths, self.sample_weights)):
            # sample_proportional mode: weight is a multiplier on uniform
            # Each sample in dataset i has weight = sample_weight[i]
            weights.extend([weight] * length)

        self.weights_tensor = torch.tensor(weights, dtype=torch.float64)
        # Normalize to get probabilities
        self.weights_tensor = self.weights_tensor / self.weights_tensor.sum()

    def _infinite_weighted_indices(self) -> Iterator[int]:
        """Infinitely yield weighted random indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)

        total_samples = len(self.dataset)
        batch_size = min(10000, total_samples)  # Sample in batches for efficiency

        while True:
            if self.shuffle:
                # Weighted random sampling with replacement
                indices = torch.multinomial(
                    self.weights_tensor,
                    num_samples=batch_size,
                    replacement=True,
                    generator=g
                ).tolist()
                yield from indices
            else:
                # Sequential (non-shuffled) - just cycle through
                yield from range(total_samples)

    def __iter__(self) -> Iterator[int]:
        """Yield indices, sharded across distributed ranks."""
        # Each rank gets every world_size-th index starting from its rank
        yield from itertools.islice(
            self._infinite_weighted_indices(),
            self.rank,
            None,
            self.world_size
        )

    def __len__(self) -> int:
        """Return total dataset length (for compatibility)."""
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Update seed for epoch (for reproducibility across epochs)."""
        self.seed = self.seed + epoch

    def get_sample_weights(self) -> List[float]:
        """Return the sample weights for logging purposes."""
        return self.sample_weights
