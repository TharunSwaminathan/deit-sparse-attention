"""Model construction, deterministic setup, and evaluation utilities."""
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_model(pretrained=True, num_classes=37):
    import timm
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def get_transforms(model):
    from timm.data import create_transform, resolve_model_data_config
    data_config = resolve_model_data_config(model)
    return (create_transform(**data_config, is_training=True),
            create_transform(**data_config, is_training=False))


def build_datasets_for_model(model, data_root, manifest, batch_size=32,
                             workers=2, download=False, seed=42):
    from .data import build_loaders
    train_tf, eval_tf = get_transforms(model)
    return build_loaders(data_root, train_tf, eval_tf, manifest, batch_size=batch_size,
                         workers=workers, download=download, seed=seed)


@torch.inference_mode()
def evaluate_model(model, loader, device, max_images=None):
    model.eval()
    preds, labels = [], []
    total_loss, count, correct = 0., 0, 0
    for xs, ys in loader:
        if max_images is not None and count >= max_images:
            break
        if max_images is not None:
            take = max_images - count
            xs, ys = xs[:take], ys[:take]
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        logits = model(xs)
        loss = torch.nn.functional.cross_entropy(logits, ys, reduction="sum")
        total_loss += float(loss.item())
        batch_pred = logits.argmax(-1)
        correct += int((batch_pred == ys).sum().item())
        count += len(ys)
        preds.extend(batch_pred.cpu().tolist())
        labels.extend(ys.cpu().tolist())
    if count == 0:
        raise RuntimeError("No images were evaluated")
    return {"n": count, "loss": total_loss / count, "accuracy": correct / count,
            "macro_f1": float(f1_score(labels, preds, labels=list(range(37)),
                                       average="macro", zero_division=0))}


def load_dense_checkpoint(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("model_name", MODEL_NAME) != MODEL_NAME:
        raise ValueError("checkpoint model name mismatch")
    model = load_model(pretrained=False, num_classes=obj.get("num_classes", 37))
    model.load_state_dict(obj["state_dict"], strict=True)
    model.to(device).eval()
    return model, obj


def save_json(path, payload):
    p=Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def sha256_file(path, chunk_bytes=1024 * 1024):
    """Stable checkpoint identity across paths / Colab and Google Drive copies."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()
