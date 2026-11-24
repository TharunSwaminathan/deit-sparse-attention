"""Stage 1: establish ONE dense DeiT-Tiny checkpoint on Oxford-IIIT Pet.

Do not consider 6-10 epochs or final-three-layer unfreezing mandatory.
Start with a frozen-backbone linear probe; full fine-tune is optional and
should follow course advice and validation evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .common import MODEL_NAME, build_datasets_for_model, evaluate_model, load_model, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/downloads")
    ap.add_argument("--manifest", default="outputs/split_manifest.json")
    ap.add_argument("--output", default="outputs/dense_head")
    ap.add_argument("--mode", choices=["head", "full"], default="head")
    ap.add_argument("--resume", help="optional previous 37-class checkpoint for full fine-tune")
    ap.add_argument("--epochs", type=int, default=8, help="pilot value, adjust from validation")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--download", action="store_true", help="download the official dataset")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        from .common import load_dense_checkpoint
        model, _ = load_dense_checkpoint(args.resume, device)
    else:
        model = load_model(pretrained=True, num_classes=37).to(device)
    train_loader, val_loader, _, names = build_datasets_for_model(
        model, args.data_root, args.manifest, args.batch_size,
        args.workers, args.download, args.seed)
    if args.mode == "head":
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    else:
        model.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == "head" else 1e-5)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=args.weight_decay)

    def lr_factor(epoch):
        if epoch < args.warmup_epochs:
            return max(.01, (epoch + 1) / max(1, args.warmup_epochs))
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(.05, .5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.eval() if args.mode == "head" else model.train()
        # In head-only mode, train the head/dropout while leaving backbone in eval.
        if args.mode == "head":
            model.head.train()
        train_loss, seen = 0., 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(images)
                loss = nn.functional.cross_entropy(prediction, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        result = evaluate_model(model, val_loader, device)
        entry = {"epoch": epoch+1, "train_loss": train_loss/seen,
                 "val_loss": result["loss"], "val_accuracy": result["accuracy"],
                 "val_macro_f1": result["macro_f1"], "lr": optimizer.param_groups[0]["lr"],
                 "seconds": time.perf_counter()-start}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            torch.save({"model_name": MODEL_NAME, "num_classes": 37,
                        "state_dict": model.state_dict(), "epoch": epoch+1,
                        "best_val": result, "class_names": names, "mode": args.mode,
                        "seed": args.seed, "manifest": args.manifest}, outdir / "best.pt")
        scheduler.step()
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
    summary={"checkpoint":str(outdir / "best.pt"), "best_val_loss":best_loss,
             "trainable_parameters":trainable, "total_parameters":total,
             "device":str(device), "mode":args.mode}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("TRAINING COMPLETE", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
