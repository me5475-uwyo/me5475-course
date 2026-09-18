"""
five_lines_trace.py
===================

Module 1 / Lecture 8 — what each of the five lines actually changes.

The training loop is five lines and they look interchangeable:

    optimizer.zero_grad()
    y_pred = model(x)
    loss   = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

They are not. Two of them do all the work, and they do *different* work:

    backward()   computes gradients, writes them into .grad, weights untouched
    step()       reads .grad, moves the weights, .grad untouched

This script prints one weight and its gradient after each line so you can see
exactly where each thing happens. Nothing here is new code — it is sine_mlp.py's
network, watched one value at a time.

Usage
-----
    source /project/me5475/setup5475.sh      # on ARCC
    python five_lines_trace.py

Runs in about a second. No figures, no files written.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build():
    """sine_mlp.py's network and data, exactly."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    x = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)
    y = torch.sin(torch.pi * x)
    return model, x, y


def part1_what_each_line_changes() -> None:
    """Walk the five lines once, printing one weight and its gradient."""
    model, x, y = build()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    w = model[0].weight          # the first layer's weight matrix; watch entry [0,0]

    def show(label: str) -> None:
        g = w.grad
        grad = "None" if g is None else f"{g[0, 0].item():+.8f}"
        print(f"  {label:<24} w[0,0] = {w[0,0].item():+.8f}   w.grad[0,0] = {grad}")

    print("=" * 74)
    print("PART 1 — one pass through the five lines, watching a single weight")
    print("=" * 74)
    show("start")
    optimizer.zero_grad();          show("1. zero_grad()")
    y_pred = model(x);              show("2. forward")
    loss = loss_fn(y_pred, y);      show("3. loss")
    loss.backward();                show("4. backward()   <<<")
    optimizer.step();               show("5. step()       <<<")

    print()
    print("  Line 4 created the gradient and did NOT move the weight.")
    print("  Line 5 moved the weight and did NOT touch the gradient.")
    print("  Lines 2 and 3 only computed -- nothing was stored on the parameter.")
    print()
    print("  Note the last row: .grad is STILL THERE after step(). Nothing clears")
    print("  it for you. That is what line 1 is for -- see part 2.")


def part2_what_forgetting_zero_grad_does() -> None:
    """Run three backward() calls with no zero_grad() between them."""
    model, x, y = build()
    loss_fn = nn.MSELoss()
    w = model[0].weight

    print()
    print("=" * 74)
    print("PART 2 — three backward() calls, no zero_grad() between them")
    print("=" * 74)
    print("  Same data every time, so each call computes the SAME gradient.")
    print("  backward() *adds* to .grad rather than replacing it:")
    print()

    first = None
    for i in range(1, 4):
        loss = loss_fn(model(x), y)
        loss.backward()
        g = w.grad[0, 0].item()
        if first is None:
            first = g
        print(f"    after backward() #{i}:  .grad[0,0] = {g:+.8f}   "
              f"({g / first:.1f}x the first)")

    print()
    print("  The gradient is three times too large, pointing the same way. Train")
    print("  like this and the model still improves -- just with an effective")
    print("  learning rate that grows every step. No error. No warning. This is")
    print("  the single most common bug in a hand-written training loop.")


if __name__ == "__main__":
    part1_what_each_line_changes()
    part2_what_forgetting_zero_grad_does()
