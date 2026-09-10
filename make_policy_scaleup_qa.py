#!/usr/bin/env python
"""Static QA charts comparing the 800-episode prototype vs. the 2000-episode
scale-up: success rates, loss/accuracy curves, and confusion-matrix recall.
Palette: dataviz skill's validated categorical slots 1 (blue, "before") and
2 (orange, "after"), single-hue blue sequential for the heatmaps, muted gray for
reference lines -- never a third categorical hue, never a rainbow.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#0b0b0b"
OUT = pathlib.Path("qa/policy_scaleup")
OUT.mkdir(parents=True, exist_ok=True)

BEFORE_CKPT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy/checkpoints")
AFTER_CKPT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy_scaleup/checkpoints")

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRAY, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def success_rate_chart():
    splits = ["train", "val", "test"]
    before = {"train": 0.867, "val": 0.633, "test": 0.600}
    after = {}
    for s in splits:
        after[s] = json.load(open(f"out/policy_eval/{s}/results.json"))["policy_success_rate"]

    x = np.arange(len(splits))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, [before[s] for s in splits], width=w, color=BLUE, label="800 episodes (before)")
    ax.bar(x + w / 2, [after[s] for s in splits], width=w, color=ORANGE, label="2,000 episodes (after)")
    ax.axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
    ax.text(2.55, 1.0, "expert ceiling", color=GRAY, fontsize=9, va="center")
    for xi, s in zip(x, splits):
        ax.text(xi - w / 2, before[s] + 0.02, f"{before[s]*100:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, after[s] + 0.02, f"{after[s]*100:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [s.capitalize() for s in splits])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Closed-loop success rate")
    ax.set_title("Policy success rate: before vs. after scaling up training data")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "success_rate_comparison.png", dpi=150)
    plt.close(fig)


def loss_curve_chart():
    before = json.load(open(BEFORE_CKPT / "history.json"))
    after = json.load(open(AFTER_CKPT / "history.json"))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(range(1, len(before["train_loss"]) + 1), before["train_loss"], color=BLUE, linestyle="-", label="train loss (before)")
    ax.plot(range(1, len(before["val_loss"]) + 1), before["val_loss"], color=BLUE, linestyle="--", label="val loss (before)")
    ax.plot(range(1, len(after["train_loss"]) + 1), after["train_loss"], color=ORANGE, linestyle="-", label="train loss (after)")
    ax.plot(range(1, len(after["val_loss"]) + 1), after["val_loss"], color=ORANGE, linestyle="--", label="val loss (after)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Train/val loss: 800 episodes vs. 2,000 episodes")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=150)
    plt.close(fig)


def accuracy_curve_chart():
    before = json.load(open(BEFORE_CKPT / "history.json"))
    after = json.load(open(AFTER_CKPT / "history.json"))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(range(1, len(before["val_accuracy"]) + 1), before["val_accuracy"], color=BLUE, marker="o", markersize=4, label="800 episodes (before)")
    ax.plot(range(1, len(after["val_accuracy"]) + 1), after["val_accuracy"], color=ORANGE, marker="o", markersize=4, label="2,000 episodes (after)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Held-out frame accuracy")
    ax.set_ylim(0.6, 0.85)
    ax.set_title("Held-out per-frame classification accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "val_accuracy_curve.png", dpi=150)
    plt.close(fig)


def confusion_heatmaps():
    before = np.array(json.load(open(BEFORE_CKPT / "val_confusion_latest.json")))
    after = np.array(json.load(open(AFTER_CKPT / "val_confusion_best.json")))
    names = [n.replace("_", "\n") for n in primitives.NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, conf, title in [(axes[0], before, "800 episodes (before)"), (axes[1], after, "2,000 episodes (after)")]:
        norm = conf / conf.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        for i in range(len(names)):
            for j in range(len(names)):
                if norm[i, j] > 0.08:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                             fontsize=6, color="white" if norm[i, j] > 0.5 else INK)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Row-normalized (recall)")
    fig.suptitle("Per-primitive confusion matrix (held-out val frames)", fontsize=12)
    fig.savefig(OUT / "confusion_matrix_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    success_rate_chart()
    loss_curve_chart()
    accuracy_curve_chart()
    confusion_heatmaps()
    print(f"wrote charts to {OUT}/")
