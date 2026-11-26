"""Official Oxford-IIIT Pet dataset and fixed stratified validation split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet


def build_loaders(data_root, train_transform, eval_transform, manifest_path,
                  batch_size=32, workers=2, seed=42, download=False, val_fraction=.20):
    """Return train, validation, test loaders and 37 breed names.

    Test loader is created but must not be USED in tuning/calibration. The
    manifest makes train/val indices stable across runs and team members.
    """
    train_set = OxfordIIITPet(data_root, split="trainval", target_types="category",
                             transform=train_transform, download=download)
    val_set = OxfordIIITPet(data_root, split="trainval", target_types="category",
                           transform=eval_transform, download=False)
    test_set = OxfordIIITPet(data_root, split="test", target_types="category",
                            transform=eval_transform, download=download)
    labels = np.asarray(train_set._labels)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if payload["dataset_count"] != len(labels):
            raise ValueError("Saved split manifest differs from official trainval length")
        train_idx, val_idx = payload["train_indices"], payload["val_indices"]
        if (len(train_idx) + len(val_idx) != len(labels) or
                len(set(train_idx) & set(val_idx)) or
                set(train_idx + val_idx) != set(range(len(labels)))):
            raise ValueError("Invalid indices in split manifest")
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.arange(len(labels)), labels))
        train_idx, val_idx = sorted(train_idx.tolist()), sorted(val_idx.tolist())
        payload = {"dataset": "OxfordIIITPet", "seed": seed, "val_fraction": val_fraction,
                   "dataset_count": len(labels), "train_indices": train_idx, "val_indices": val_idx}
        manifest_path.write_text(json.dumps(payload, indent=2))

    pin = True
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    val_loader = DataLoader(Subset(val_set, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    return train_loader, val_loader, test_loader, train_set.classes
