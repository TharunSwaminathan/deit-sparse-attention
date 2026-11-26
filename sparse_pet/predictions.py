"""Aligned prediction export for sequential evaluation loaders."""
from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import BatchSampler, SequentialSampler, Subset, default_collate


def evaluation_indices(loader):
    """Return underlying dataset indices in loader order (not batch offsets).

    Refuse samplers that cannot guarantee this order. Validation subsets map
    back to the official trainval index; test uses the official test index.
    """
    if (type(loader.sampler) is not SequentialSampler or
            type(loader.batch_sampler) is not BatchSampler or
            loader.batch_sampler.sampler is not loader.sampler or
            loader.collate_fn is not default_collate or
            loader.drop_last or loader.batch_sampler.drop_last or
            not getattr(loader, "in_order", True)):
        raise ValueError("Prediction export requires sequential, ordered, non-dropping batches")
    dataset = loader.dataset
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate dataset indices in prediction export")
    return indices


def write_predictions(path, rows, split, configuration):
    """Save one row per image with stable split-qualified IDs and 0-based labels."""
    prefix = {"val": "trainval", "test": "test"}[split]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "configuration", "image_id", "image_index",
                         "ground_truth", "predicted_class", "correct"])
        for index, target, prediction in rows:
            writer.writerow([split, configuration, f"{prefix}:{index}", index,
                             target, prediction, int(target == prediction)])
