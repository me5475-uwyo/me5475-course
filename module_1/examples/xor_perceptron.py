"""
xor_perceptron.py
=================

Module 1 / Lecture 7 live demo — show that a single linear unit CANNOT
solve XOR, but adding one hidden layer CAN.

What this actually trains
-------------------------
Both models are trained the SAME way: Adam on binary cross-entropy. So the
single-unit model here is a LOGISTIC LINEAR CLASSIFIER, not Rosenblatt's 1958
perceptron update (which changes weights only on a misclassification and has no
loss function). The point being demonstrated is about the LINEAR DECISION
BOUNDARY, which both share — not about the training rule.

Two things the printed numbers do not say
-----------------------------------------
* The 50% accuracy below is what THIS run converges to, not the ceiling for a
  linear classifier on XOR. A line can always get 3 of the 4 points right (75%);
  symmetric BCE simply prefers the symmetric all-0.5 solution here. Either way a
  line cannot get all four, which is the actual claim.
* Minsky and Papert (1969) documented this limitation. Avoid the tidy slogan about
  one paper starting, or one paper ending, a decade of reduced funding; the bounded
  claim is the one this script demonstrates -- a single linear unit cannot represent
  XOR, and one hidden layer removes that limitation. (For the history: backpropagation
  for multilayer networks reached a broad audience in 1986, Rumelhart, Hinton &
  Williams.)

Usage
-----
    python xor_perceptron.py

Produces:
  - xor_single_layer.png : decision boundary of a single linear unit (fails)
  - xor_mlp.png          : decision boundary of an MLP with one hidden layer (succeeds)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # save-only: no display needed, so this works over SSH
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# --- The XOR dataset ---
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


def train_model(model: nn.Module, lr: float = 0.1, epochs: int = 5000) -> list[float]:
    """Train the given model on XOR and return the loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def plot_decision_boundary(model: nn.Module, title: str, out_path: str) -> None:
    """Plot the model's prediction on a fine grid + the four XOR points."""
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Fixed 0-1 levels. With `levels=20` matplotlib autoscales to the data range,
    # so a model sitting flat at P=0.5 gets its floating-point noise stretched
    # into a full-contrast picture that looks like a decision boundary.
    cs = ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 21),
                     cmap="RdBu_r", alpha=0.85, vmin=0.0, vmax=1.0)
    # Only draw the P=0.5 contour if the model actually commits to a side
    # somewhere on the grid; otherwise there is no boundary to draw.
    if probs.max() - 0.5 > 0.01 and 0.5 - probs.min() > 0.01:
        ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
    else:
        ax.text(0.5, 1.42, "model is flat at P = 0.5 — no boundary",
                ha="center", va="top", fontsize=9)
    # Markers must read on THREE different grounds: the pale flat field of the
    # failing model, and the deep red / deep blue of the MLP's two half-planes.
    # So each marker carries a dark face, a white ring and a thin dark rim --
    # that sandwich stays visible whatever is behind it -- and the shape alone
    # (circle vs square) still distinguishes the classes in greyscale or print.
    STYLE = {1.0: dict(marker="o", label="y = 1  (inputs differ)"),
             0.0: dict(marker="s", label="y = 0  (inputs agree)")}
    labelled = set()
    for i in range(4):
        cls = Y[i].item()
        st = STYLE[cls]
        ax.plot(X[i, 0].item(), X[i, 1].item(), st["marker"], markersize=15,
                markerfacecolor="#1A1A1A", markeredgecolor="white", markeredgewidth=3.0,
                zorder=5, label=st["label"] if cls not in labelled else None)
        ax.plot(X[i, 0].item(), X[i, 1].item(), st["marker"], markersize=17,
                markerfacecolor="none", markeredgecolor="#1A1A1A", markeredgewidth=1.2,
                zorder=6)
        labelled.add(cls)
    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.92,
                    facecolor="white", edgecolor="#999999")
    leg.set_zorder(10)
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax, label="predicted P(y=1)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    torch.manual_seed(0)

    # --- Single linear unit (perceptron) ---
    print("=== Single linear unit (logistic, no hidden layer) ===")
    perceptron = nn.Linear(2, 1)
    losses_perc = train_model(perceptron)
    final_loss_perc = losses_perc[-1]
    with torch.no_grad():
        preds_perc = (torch.sigmoid(perceptron(X)) > 0.5).float()
        acc_perc = (preds_perc == Y).float().mean().item()
    print(f"Final loss: {final_loss_perc:.4f}; accuracy on XOR: {acc_perc * 100:.1f}%")
    plot_decision_boundary(perceptron, "Single linear unit -- fails XOR",
                            "xor_single_layer.png")

    # --- 1-hidden-layer MLP with tanh ---
    print("\n=== 1-hidden-layer MLP, width 4, tanh ===")
    mlp = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )
    losses_mlp = train_model(mlp)
    final_loss_mlp = losses_mlp[-1]
    with torch.no_grad():
        preds_mlp = (torch.sigmoid(mlp(X)) > 0.5).float()
        acc_mlp = (preds_mlp == Y).float().mean().item()
    print(f"Final loss: {final_loss_mlp:.4f}; accuracy on XOR: {acc_mlp * 100:.1f}%")
    plot_decision_boundary(mlp, "MLP (1 hidden layer, width 4, tanh) -- solves XOR",
                            "xor_mlp.png")

    # --- Summary ---
    print("\nSummary:")
    print(f"  Single linear unit: loss {final_loss_perc:.4f}  accuracy: {acc_perc * 100:.0f}%")
    print(f"  + 1 hidden layer:   loss {final_loss_mlp:.4f}  accuracy: {acc_mlp * 100:.0f}%")
    print("\nNo straight line can separate XOR, so the single unit cannot get all")
    print("four points right (Minsky-Papert 1969). The 50% above is this run's")
    print("symmetric solution; a line can reach 75%, never 100%. One hidden layer")
    print("removes the restriction entirely.")


if __name__ == "__main__":
    main()
