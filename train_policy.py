#!/usr/bin/env python
"""Behavior-cloning training for the ResNet18 policy. See POLICY_TRAINING.md.

    source env.sh
    python train_policy.py --epochs 15 --batch-size 128

No Cosmos involved -- this trains against real mujoco frames/labels only, on a
plain torch/torchvision stack (no diffusers pipeline, so env.sh's LD_LIBRARY_PATH
fix for cudnn's runtime-compiled engine isn't needed here, but sourcing it anyway
is harmless and keeps the invocation consistent with everything else in this repo).
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from policy_data import NUM_CLASSES, PolicyFrameDataset, class_weights
from policy_model import PolicyNet, imagenet_transform
from policy_data import primitives  # noqa: F401 -- re-exported for the class-count log line below

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/dataset")
CKPT_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy/checkpoints")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    correct = total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for images, gripper, labels in loader:
        images, gripper, labels = images.to(device), gripper.to(device), labels.to(device)
        logits = model(images, gripper)
        loss_sum += ce(logits, labels).item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        for t, p in zip(labels.cpu().numpy(), pred.cpu().numpy()):
            confusion[t, p] += 1
    return {"loss": loss_sum / total, "accuracy": correct / total, "confusion": confusion.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=CKPT_ROOT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-hours", type=float, default=None, help="wall-clock safety cap; --epochs becomes a ceiling, not a target")
    ap.add_argument("--limit-train-batches", type=int, default=None, help="prototyping/smoke-test cap")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    tfm = imagenet_transform()
    train_ds = PolicyFrameDataset(args.dataset_root, "train", transform=tfm, augment=args.augment, augment_seed=args.seed)
    val_ds = PolicyFrameDataset(args.dataset_root, "val", transform=tfm, augment=False)
    log.info("train examples: %d  val examples: %d", len(train_ds), len(val_ds))

    counts = train_ds.class_counts()
    log.info("train class counts: %s", dict(zip(primitives.NAMES, counts.tolist())))
    weights = class_weights(counts).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = PolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(weight=weights)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if args.max_hours is not None and (time.monotonic() - t0) / 3600.0 >= args.max_hours:
            log.info("hit --max-hours=%.2f after epoch %d; stopping", args.max_hours, epoch - 1)
            break
        model.train()
        running = 0.0
        n_batches = 0
        for bi, (images, gripper, labels) in enumerate(train_loader):
            if args.limit_train_batches is not None and bi >= args.limit_train_batches:
                break
            images, gripper, labels = images.to(device), gripper.to(device), labels.to(device)
            logits = model(images, gripper)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        log.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  elapsed=%.1fmin",
            epoch, args.epochs, train_loss, val_metrics["loss"], val_metrics["accuracy"],
            (time.monotonic() - t0) / 60,
        )
        torch.save(model.state_dict(), args.out / "latest.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        (args.out / "val_confusion_latest.json").write_text(json.dumps(val_metrics["confusion"]))
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), args.out / "best.pt")
            (args.out / "val_confusion_best.json").write_text(json.dumps(val_metrics["confusion"]))
            log.info("new best val_accuracy=%.4f -> best.pt", best_val_accuracy)

    log.info(
        "done: %d epochs run, %.1f min total, best val_accuracy=%.4f, checkpoints at %s",
        len(history["train_loss"]), (time.monotonic() - t0) / 60, best_val_accuracy, args.out,
    )


if __name__ == "__main__":
    main()
