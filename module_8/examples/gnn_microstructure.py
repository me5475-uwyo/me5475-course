"""
gnn_microstructure.py
======================

Module 8 / Lab 8 -- train a graph neural network (GIN or GATv2) on the microstructure graphs from
microstructure_to_graph.py to predict the apparent modulus E_app, and evaluate it exactly as Lab 6 evaluates its CNN.

Comparable with module_6/examples/cnn_microstructure_to_modulus.py by construction:
  - the same split: split_data(seed=--split-seed) over the dataset's rows, 70 % train / 15 % validation / 15 % test,
    identical to the CNN's (each graph carries its Lab 6 row number); --seed changes only initialization and order
  - the target standardized with training-set statistics; MSE loss; AdamW; 200 epochs; batch 32
  - the weights of the best validation epoch are kept (patience 30, as in the CNN); the test set is used once, at the
    end, never for selection
  - the same metrics in physical units: test MAE of E_app and MAE / mean(E_app)
  - the checkpoint stores the split (seed and test rows), so comparison.py can check it against the CNN's

Node features (7) and edge features (4) are standardized with training-set statistics. Both models use the edge
features: GINEConv (GIN with edge features) and GATv2Conv with edge_dim. Runs on the CPU (the graphs are small).

Usage
-----
    python gnn_microstructure.py --graphs microstructure_graphs.pt --architecture gin   # -> checkpoints/gin_modulus.pt
    python gnn_microstructure.py --graphs microstructure_graphs.pt --architecture gat   # -> checkpoints/gat_modulus.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GATv2Conv, global_add_pool, global_mean_pool


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", type=Path, required=True, help="output of microstructure_to_graph.py")
    p.add_argument("--architecture", choices=["gin", "gat"], default="gin")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--pooling", choices=["sum", "mean"], default="sum")
    p.add_argument("--out", type=Path, default=None, help="default: checkpoints/<architecture>_modulus.pt")
    p.add_argument("--seed", type=int, default=42, help="model initialization and minibatch order only")
    p.add_argument("--split-seed", type=int, default=42,
                   help="the train/val/test partition; use the same value as your Lab 6 CNN")
    args = p.parse_args()
    if args.architecture == "gat" and args.hidden % 4 != 0:
        p.error(f"--hidden {args.hidden}: GATv2 uses 4 attention heads of hidden/4 channels each, so --hidden must be "
                "divisible by 4 (e.g. 32 or 64)")
    return args


def split_data(n, seed=42, val_frac=0.15, test_frac=0.15):
    """Row indices exactly as split_data in cnn_microstructure_to_modulus.py (same seed -> same rows)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return {"test": idx[:n_test], "val": idx[n_test : n_test + n_val], "train": idx[n_test + n_val:]}


class MicrostructureGIN(nn.Module):
    """GIN with edge features (GINEConv), sum or mean readout, MLP head."""

    def __init__(self, in_dim=7, edge_dim=4, hidden=64, n_layers=4, pooling="sum"):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     train_eps=True, edge_dim=edge_dim)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.pool = global_add_pool if pooling == "sum" else global_mean_pool

    def forward(self, data):
        x = self.embed(data.x).relu()
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr).relu()
        return self.head(self.pool(x, data.batch)).squeeze(-1)


class MicrostructureGAT(nn.Module):
    """GATv2 (Brody et al. 2022) with edge features, sum or mean readout, MLP head."""

    def __init__(self, in_dim=7, edge_dim=4, hidden=64, n_layers=4, pooling="sum", heads=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=edge_dim)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.pool = global_add_pool if pooling == "sum" else global_mean_pool

    def forward(self, data):
        x = self.embed(data.x).relu()
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr).relu()
        return self.head(self.pool(x, data.batch)).squeeze(-1)


def standardized(graphs, xm, xs, em, es, ym, ys):
    out = []
    for g in graphs:
        h = g.clone()
        h.x = (g.x - xm) / xs
        h.edge_attr = (g.edge_attr - em) / es
        h.y = (g.y - ym) / ys
        out.append(h)
    return out


def evaluate(model, loader):
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch))
            truth.append(batch.y)
    return torch.cat(preds), torch.cat(truth)


def main():
    args = parse_args()
    if args.out is None:
        args.out = Path(f"checkpoints/{args.architecture}_modulus.pt")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graphs = torch.load(args.graphs, weights_only=False)
    graphs = sorted(graphs, key=lambda g: int(g.row))
    rows = [int(g.row) for g in graphs]
    if rows != list(range(len(graphs))):
        raise RuntimeError("the graph file must hold exactly one graph per Lab 6 row (0..N-1); rerun the converter")
    print(f"Loaded {len(graphs)} graphs (one per Lab 6 row)")

    split = split_data(len(graphs), seed=args.split_seed)
    tr = [graphs[i] for i in split["train"]]
    va = [graphs[i] for i in split["val"]]
    te = [graphs[i] for i in split["test"]]
    print(f"Train: {len(tr)}  Val: {len(va)}  Test: {len(te)}  (split seed {args.split_seed}, the Lab 6 CNN's split)")
    print(f"Test rows: first five {split['test'][:5].tolist()}")

    # standardize with training-set statistics only
    x_tr = torch.cat([g.x for g in tr]); e_tr = torch.cat([g.edge_attr for g in tr])
    y_tr = torch.cat([g.y for g in tr])
    xm, xs = x_tr.mean(0), x_tr.std(0).clamp_min(1e-8)
    em, es = e_tr.mean(0), e_tr.std(0).clamp_min(1e-8)
    ym, ys = y_tr.mean(), y_tr.std()
    tr_n, va_n, te_n = (standardized(s, xm, xs, em, es, ym, ys) for s in (tr, va, te))

    train_loader = DataLoader(tr_n, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(va_n, batch_size=256)
    test_loader = DataLoader(te_n, batch_size=256)

    Model = MicrostructureGIN if args.architecture == "gin" else MicrostructureGAT
    model = Model(hidden=args.hidden, n_layers=args.n_layers, pooling=args.pooling)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.architecture.upper()} parameters: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    best_val, best_state, patience, bad = float("inf"), None, 30, 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:                       # one epoch = one shuffled pass over the training graphs
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        epoch_loss /= len(tr_n)
        p_va, y_va = evaluate(model, val_loader)
        val_loss = loss_fn(p_va, y_va).item()
        history["train"].append(epoch_loss); history["val"].append(val_loss)
        if val_loss < best_val:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train={epoch_loss:.4e}  val={val_loss:.4e}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # test, once, in physical units (the same definitions as the Lab 6 CNN)
    p_te, y_te = evaluate(model, test_loader)
    pred = (p_te * ys + ym).numpy()
    truth = (y_te * ys + ym).numpy()
    mae = float(np.abs(pred - truth).mean())
    rel_mae = mae / float(truth.mean())
    print(f"\nTest set MAE:           {mae:.4e}")
    print(f"Test set MAE / mean(E): {rel_mae:.4%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": {"seed": args.split_seed, "test_idx": split["test"].tolist()},
        "scalers": {"x_mean": xm.tolist(), "x_std": xs.tolist(), "e_mean": em.tolist(), "e_std": es.tolist(),
                    "mu_y": float(ym), "sd_y": float(ys)},
        "history": history,
        "test_pred": pred.tolist(),
        "test_mae": mae,
        "test_relative_mae": rel_mae,
    }, args.out)
    print(f"Saved {args.out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(truth, pred, s=20, alpha=0.6)
    lo, hi = truth.min(), truth.max()
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("MOOSE E_app")
    axes[0].set_ylabel(f"{args.architecture.upper()}-predicted E_app")
    axes[0].set_title(f"Test parity (MAE = {mae:.4f}, rel = {rel_mae:.2%})")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(history["train"], label="train (epoch mean)")
    axes[1].semilogy(history["val"], label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE (standardized)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, which="both"); axes[1].set_title("Training history")
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=120)
    print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
