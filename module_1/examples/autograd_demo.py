"""L5 · PyTorch autograd — the five lines on the projector, then a tiny regression.

Run it:
    python autograd_demo.py

Act 1 reproduces the slide exactly: build a graph, call backward(), read .grad.
Act 2 uses the same machinery to actually fit something, which is the whole of L8's
training loop in miniature.
"""
import torch


def act1_scalar_graph() -> None:
    """The slide verbatim: f = (x + y) * x, differentiated by reverse-mode autodiff."""
    print("=== Act 1 — one graph, one backward pass ===")

    # Build the computation graph.
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    f = (x + y) * x          # f = x^2 + xy

    # Reverse-mode pass.
    f.backward()

    print(f"  f            = {f.item():.1f}        # (2+3)*2")
    print(f"  x.grad       = {x.grad.item():.1f}        # df/dx = 2x + y = 7")
    print(f"  y.grad       = {y.grad.item():.1f}        # df/dy = x     = 2")

    # f is the ROOT of the graph — it carries the grad_fn that backward() walks.
    # x and y are the LEAVES — they are what .grad gets filled in on.
    print(f"  f.grad_fn    = {type(f.grad_fn).__name__}   (f is the root)")
    print(f"  x.is_leaf    = {x.is_leaf}, y.is_leaf = {y.is_leaf}")


def act2_linear_regression() -> None:
    """Same three idioms, now fitting y = w*x + b. This is L8's training loop, by hand."""
    print("\n=== Act 2 — the same machinery, fitting a line ===")

    torch.manual_seed(0)
    x_data = torch.linspace(-1, 1, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn(50)   # truth: w = 2, b = 1

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for step in range(200):
        y_pred = w * x_data + b
        loss = ((y_pred - y_data) ** 2).mean()
        loss.backward()                  # fills w.grad, b.grad

        with torch.no_grad():            # updates must NOT be tracked by autograd
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
            w.grad.zero_()               # gradients accumulate — clear them
            b.grad.zero_()

        if step % 50 == 0:
            print(f"  step {step:>3}: loss = {loss.item():.4f}   w = {w.item():.3f}  b = {b.item():.3f}")

    print(f"  final:    w = {w.item():.3f}  b = {b.item():.3f}   (truth: 2.000, 1.000)")

    print("\nThree idioms to carry into L8:")
    print("  requires_grad=True   opt a tensor into the graph")
    print("  with torch.no_grad() parameter updates that should not be tracked")
    print("  .grad.zero_()        gradients accumulate; forgetting this is the #1 PyTorch bug")


if __name__ == "__main__":
    act1_scalar_graph()
    act2_linear_regression()
