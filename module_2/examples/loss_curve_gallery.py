"""
loss_curve_gallery.py
=====================

Module 2 / Lecture 10 demo --- generates six synthetic training runs, each
intentionally exhibiting a canonical loss-curve pathology. The instructor walks
through the resulting figure on the projector; students should leave able to
identify each pathology by sight in any future training output.

The six panels:
  (a) Healthy training            -- monotonic decrease, val tracks train.
  (b) Underfitting                -- both losses high and flat.
  (c) Overfitting                 -- train decreases, val decreases then rises.
  (d) Unstable (LR too high)      -- loss oscillates wildly, possibly NaN.
  (e) Dead training               -- loss flat from the start.
  (f) Leaky split                 -- val loss BELOW train loss.

Usage
-----
    python loss_curve_gallery.py --out loss_curve_gallery.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("loss_curve_gallery.png"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def healthy(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(n)
    # tau = n/8, not n/4: the curve must FLATTEN inside the window, because
    # "reaches a stable plateau" is the thing this panel is supposed to teach.
    train = 1.0 * np.exp(-epochs / (n / 8.0)) + 1e-3 + 1e-4 * rng.standard_normal(n)
    val = train + 0.05 + 1e-3 * rng.standard_normal(n)
    return np.clip(train, 1e-5, None), np.clip(val, 1e-5, None)


def underfit(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(n)
    # Amplitude 3.0, not 0.4: underfitting must still visibly DESCEND to a high
    # plateau, or panel (b) is indistinguishable from panel (e), dead training --
    # which is the single distinction this panel exists to teach.
    train = 3.0 * np.exp(-epochs / (n / 3.0)) + 0.5 + 1e-3 * rng.standard_normal(n)
    val = train + 0.02 + 1e-3 * rng.standard_normal(n)
    return train, val


def overfit(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(n)
    train = 1.0 * np.exp(-epochs / (n / 5.0)) + 1e-4 + 1e-5 * rng.standard_normal(n)
    val = 0.6 * np.exp(-epochs / (n / 8.0)) + 0.05 + 0.5 * (1 - np.exp(-epochs / (n / 1.5))) + 1e-3 * rng.standard_normal(n)
    return np.clip(train, 1e-5, None), val


def unstable(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(n)
    base = 0.5 * np.exp(-epochs / (n / 3.0)) + 0.1
    noise = 0.4 * np.abs(rng.standard_normal(n))
    train = base + noise
    train[n // 3] = 5.0  # spike
    train[2 * n // 3] = 8.0  # bigger spike
    val = train + 0.05 + 0.1 * np.abs(rng.standard_normal(n))
    return train, val


def dead(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flat from step one -- nothing is being learned at all."""
    train = 0.7 + 1e-3 * rng.standard_normal(n)
    val = 0.7 + 1e-3 * rng.standard_normal(n)
    return train, val


def leaky(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(n)
    train = 1.0 * np.exp(-epochs / (n / 4.0)) + 0.01 + 1e-3 * rng.standard_normal(n)
    val = train * 0.5 + 1e-3 * rng.standard_normal(n)  # IMPLAUSIBLY below train
    return np.clip(train, 1e-5, None), np.clip(val, 1e-5, None)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n = 500

    cases = [
        ("(a) healthy", healthy),
        ("(b) underfit", underfit),
        ("(c) overfit", overfit),
        ("(d) unstable", unstable),
        ("(e) dead", dead),
        ("(f) leaky", leaky),
    ]

    # House style (2026-09-18): matplotlib's default C0/C1 blue-orange fought the
    # cream-and-brown deck it is projected inside, and separated train from val by
    # hue alone. Train is solid brown, val is dashed red: distinguishable in
    # greyscale and to a colour-blind reader, which hue alone is not.
    CREAM, INK, TRAIN, VAL, GRID = "#FBF8EF", "#492F24", "#492F24", "#A4161A", "#C9BFA9"

    # ONE shared y-range for all six panels. Autoscaling each panel separately
    # was actively misleading: (e) dead training has 1e-3 noise on a value of
    # 0.7, and autoscaled it rendered as the wildest curve in the gallery.
    # A gallery exists so shapes can be COMPARED; comparison needs one ruler.
    # 9.5 x 4.0, not 13 x 7. This figure is projected as the keystone of L10, scaled
    # to about half size on the slide -- so every point of font size in here lands
    # at about half a point on the wall. A small canvas with large type is the only
    # way the axis labels survive the projector.
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.0), facecolor=CREAM)
    for ax, (title, fn) in zip(axes.flat, cases):
        train, val = fn(n, rng)
        ax.set_facecolor(CREAM)
        ax.semilogy(train, label="train", color=TRAIN, linewidth=1.6)
        ax.semilogy(val, label="val", color=VAL, linewidth=1.6, linestyle="--")
        ax.set_title(title, color=INK, fontsize=13.5)
        ax.set_xlabel("epoch", color=INK, fontsize=11)
        ax.set_ylabel("loss", color=INK, fontsize=11)
        ax.tick_params(colors=INK, labelsize=10)
        ax.set_xticks([0, 250, 500])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_ylim(1e-4, 1e1)
        ax.grid(True, alpha=0.6, which="both", color=GRID, linewidth=0.5)

    fig.suptitle("Six canonical loss-curve shapes", fontsize=15, y=1.05, color=INK)
    # One legend for the figure, not six. Six copies of the same two-entry key ate
    # a quarter of every panel.
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.995, 1.06), ncol=2, fontsize=11,
               facecolor=CREAM, edgecolor=GRID, labelcolor=INK, frameon=False)
    fig.tight_layout()
    # .pdf out => vector, which is what the Beamer deck embeds; .png keeps dpi.
    fig.savefig(args.out, dpi=120, bbox_inches="tight", facecolor=CREAM)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
