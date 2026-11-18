"""
meshgraphnet_demo.py
=====================

Module 8 / Lab 8 stretch (Task 5) -- a small MeshGraphNet-style simulator (Pfaff et al., ICLR 2021) for a toy
dynamic problem: a mass-spring membrane, NOT a finite-element model. It shows the MeshGraphNet recipe --
encode, process with message passing, decode, then roll out step by step -- on data that takes seconds to make.

The simulation: an N x N grid of unit masses joined by springs (horizontal, vertical and both diagonals) with
linear damping. The bottom row is fixed; the top row is pulled sideways at a constant rate to a random final
displacement (and held at y = 1). Time stepping is semi-implicit (symplectic) Euler:
    v_{t+1} = v_t + dt * a_t,    x_{t+1} = x_t + dt * v_{t+1}.

What the network learns: the acceleration a_t of every free node from the current state, as in MeshGraphNets
(next-step supervised regression on the simulator's states, then integrate). Inputs:
  - node features: velocity (vx, vy) and node type (free, fixed, driven)
  - edge features: current relative position x_dst - x_src (2), its length, and the rest length
    (relative positions only, no absolute coordinates)
Inputs and targets are standardized with training statistics; the loss covers free nodes only (fixed and driven
nodes are set by the boundary conditions, not predicted).

Model: node and edge encoders (MLP + LayerNorm), --n-message-passing processor steps, each with an edge update
e' = e + MLP([e, h_src, h_dst]) and a node update h' = h + MLP([h, sum of incoming e']), and a node decoder.

Data split by trajectory (80 % train, 20 % test), so the test rollouts are trajectories the model never saw.
Outputs: one-step test error, a full rollout of every test trajectory from its initial state (the model's own
predictions fed back, boundary conditions imposed), the position error along the rollout, and a figure.

Usage
-----
    python meshgraphnet_demo.py                      # defaults below; CPU, a few minutes
    python meshgraphnet_demo.py --noise-std 0.003    # ablation: noisy input velocities, targets unchanged (see --help)
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

FREE, FIXED, DRIVEN = 0, 1, 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trajectories", type=int, default=100)
    p.add_argument("--n-timesteps", type=int, default=50)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30, help="passes over all training states")
    p.add_argument("--batch-size", type=int, default=16, help="graphs (states) per minibatch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-message-passing", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="std of Gaussian noise added to training input velocities (0 = none). A simplified "
                        "augmentation ablation: the acceleration targets are NOT adjusted, unlike the noise-correcting "
                        "training of Pfaff et al. 2021, App. A.2.2; it may help or hurt rollouts")
    p.add_argument("--out", type=Path, default=Path("meshgraphnet_demo.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Toy simulator: mass-spring membrane
# -----------------------------------------------------------------------------
K_SPRING, DAMPING, DT = 100.0, 0.5, 0.01


def build_grid_mesh(n):
    """n x n grid on [0, 1]^2; springs along rows, columns and both diagonals (both directions listed)."""
    xs = np.linspace(0.0, 1.0, n)
    pos = np.array([[xs[j], xs[i]] for i in range(n) for j in range(n)], dtype=np.float64)   # node = i*n + j
    pairs = []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            if j + 1 < n:
                pairs.append((k, k + 1))
            if i + 1 < n:
                pairs.append((k, k + n))
            if i + 1 < n and j + 1 < n:
                pairs.append((k, k + n + 1))
            if i + 1 < n and j > 0:
                pairs.append((k, k + n - 1))
    pairs = np.array(pairs, dtype=np.int64)
    edge_index = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T          # (2, E), both directions
    node_type = np.full(len(pos), FREE, dtype=np.int64)
    node_type[pos[:, 1] < 1e-9] = FIXED                                     # bottom row
    node_type[pos[:, 1] > 1 - 1e-9] = DRIVEN                                # top row
    return pos, edge_index, node_type


def spring_acceleration(pos, vel, edge_index, rest):
    src, dst = edge_index
    d = pos[dst] - pos[src]
    dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    f = K_SPRING * (dist - rest[:, None]) * d / dist                        # force on src from each spring
    acc = np.zeros_like(pos)
    np.add.at(acc, src, f)
    return acc - DAMPING * vel                                              # unit masses


def simulate(pos0, edge_index, node_type, top_disp, n_steps):
    """Return positions (T+1, n, 2) and velocities (T+1, n, 2)."""
    rest = np.linalg.norm(pos0[edge_index[1]] - pos0[edge_index[0]], axis=1)
    drive_v = top_disp / (n_steps * DT)
    pos, vel = pos0.copy(), np.zeros_like(pos0)
    vel[node_type == DRIVEN, 0] = drive_v
    P, V = [pos.copy()], [vel.copy()]
    for step in range(n_steps):
        acc = spring_acceleration(pos, vel, edge_index, rest)
        vel = vel + DT * acc
        pos = pos + DT * vel
        apply_bc(pos, vel, pos0, node_type, drive_v, step + 1)
        P.append(pos.copy()); V.append(vel.copy())
    return np.array(P), np.array(V)


def apply_bc(pos, vel, pos0, node_type, drive_v, step):
    fixed, driven = node_type == FIXED, node_type == DRIVEN
    pos[fixed], vel[fixed] = pos0[fixed], 0.0
    pos[driven, 0] = pos0[driven, 0] + drive_v * DT * step
    pos[driven, 1] = pos0[driven, 1]
    vel[driven] = [drive_v, 0.0]


# -----------------------------------------------------------------------------
# Features, normalization, batching
# -----------------------------------------------------------------------------
def features(pos, vel, node_type, edge_index, rest):
    src, dst = edge_index
    d = pos[dst] - pos[src]
    edge = np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True), rest[:, None]], axis=1)
    node = np.concatenate([vel, np.eye(3)[node_type]], axis=1)
    return node.astype(np.float32), edge.astype(np.float32)


class Normalizer:
    def __init__(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0) + 1e-8

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def batch_graphs(nodes, edges, edge_index, n_nodes):
    """Stack B graphs with identical connectivity into one disconnected graph."""
    B = len(nodes)
    offsets = torch.arange(B).repeat_interleave(edge_index.shape[1]) * n_nodes
    ei = edge_index.repeat(1, B) + offsets
    return torch.cat(nodes), torch.cat(edges), ei


# -----------------------------------------------------------------------------
# MeshGraphNet-style model: encode, process (edge + node updates), decode
# -----------------------------------------------------------------------------
def mlp(i, h, o, norm=True):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*layers, nn.LayerNorm(o)) if norm else nn.Sequential(*layers)


class MeshGraphNet(nn.Module):
    def __init__(self, node_in=5, edge_in=4, hidden=64, n_message_passing=10, out=2):
        super().__init__()
        self.node_enc = mlp(node_in, hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.edge_mlps = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(n_message_passing)])
        self.node_mlps = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(n_message_passing)])
        self.decoder = mlp(hidden, hidden, out, norm=False)

    def forward(self, node, edge, edge_index):
        h, e = self.node_enc(node), self.edge_enc(edge)
        src, dst = edge_index
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e = e + edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))          # edge update (residual)
            agg = torch.zeros_like(h).index_add_(0, dst, e)                    # sum of incoming edges
            h = h + node_mlp(torch.cat([h, agg], dim=-1))                      # node update (residual)
        return self.decoder(h)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pos0, edge_index_np, node_type = build_grid_mesh(args.grid_size)
    n_nodes = len(pos0)
    rest = np.linalg.norm(pos0[edge_index_np[1]] - pos0[edge_index_np[0]], axis=1)
    print(f"Mass-spring membrane: {n_nodes} nodes, {edge_index_np.shape[1] // 2} springs; "
          f"{args.n_trajectories} trajectories x {args.n_timesteps} steps")

    disps = rng.uniform(0.05, 0.3, args.n_trajectories)
    trajs = [simulate(pos0, edge_index_np, node_type, d, args.n_timesteps) for d in disps]
    order = rng.permutation(args.n_trajectories)
    n_test = int(0.2 * args.n_trajectories)
    test_ids, train_ids = order[:n_test], order[n_test:]
    print(f"Split by trajectory: {len(train_ids)} train, {len(test_ids)} test")

    def states(ids):
        X_node, X_edge, Y = [], [], []
        for t_id in ids:
            P, V = trajs[t_id]
            for t in range(args.n_timesteps):
                acc = (V[t + 1] - V[t]) / DT                                   # the simulator's acceleration
                nf, ef = features(P[t], V[t], node_type, edge_index_np, rest)
                X_node.append(nf); X_edge.append(ef); Y.append(acc.astype(np.float32))
        return np.array(X_node), np.array(X_edge), np.array(Y)

    tr_node, tr_edge, tr_y = states(train_ids)
    te_node, te_edge, te_y = states(test_ids)
    free = node_type == FREE
    node_norm = Normalizer(tr_node.reshape(-1, tr_node.shape[-1]))
    node_norm.mean[2:], node_norm.std[2:] = 0.0, 1.0                           # leave the one-hot node type as is
    edge_norm = Normalizer(tr_edge.reshape(-1, tr_edge.shape[-1]))
    y_norm = Normalizer(tr_y[:, free].reshape(-1, 2))

    model = MeshGraphNet(hidden=args.hidden, n_message_passing=args.n_message_passing)
    print(f"MeshGraphNet parameters: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ei = torch.tensor(edge_index_np)
    free_t = torch.tensor(free)

    def predict(node, edge):
        """node, edge: (B, n, .) arrays already including any noise; returns standardized accel (B*n, 2)."""
        nodes = [torch.tensor(node_norm(n_)) for n_ in node]
        edges = [torch.tensor(edge_norm(e_)) for e_ in edge]
        x, e, eib = batch_graphs(nodes, edges, ei, n_nodes)
        return model(x, e, eib)

    n_tr = len(tr_node)
    history = {"train": [], "test": []}
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n_tr)
        total = 0.0
        for s in range(0, n_tr, args.batch_size):                              # one epoch = one pass over all states
            b = perm[s : s + args.batch_size]
            node = tr_node[b].copy()
            if args.noise_std > 0:                                             # ablation: targets stay clean
                node[:, :, :2] += rng.normal(0.0, args.noise_std, node[:, :, :2].shape)
            pred = predict(node, tr_edge[b])
            target = torch.tensor(y_norm(tr_y[b].reshape(-1, 2)))
            mask = free_t.repeat(len(b))
            loss = ((pred - target)[mask] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = predict(te_node, te_edge)
            target = torch.tensor(y_norm(te_y.reshape(-1, 2)))
            mask = free_t.repeat(len(te_node))
            test_loss = ((pred - target)[mask] ** 2).mean().item()
        history["train"].append(total / n_tr); history["test"].append(test_loss)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train={total / n_tr:.4e}  one-step test={test_loss:.4e} (standardized MSE)")

    # Rollout every test trajectory from its initial state with the model's own predictions
    model.eval()
    errs = []
    with torch.no_grad():
        for t_id in test_ids:
            P, V = trajs[t_id]
            drive_v = disps[t_id] / (args.n_timesteps * DT)
            pos, vel = P[0].copy(), V[0].copy()
            err_t = [0.0]
            for t in range(args.n_timesteps):
                nf, ef = features(pos, vel, node_type, edge_index_np, rest)
                acc = y_norm.inverse(predict(nf[None], ef[None]).numpy())
                vel = vel + DT * np.where(free[:, None], acc, 0.0)
                pos = pos + DT * vel
                apply_bc(pos, vel, pos0, node_type, drive_v, t + 1)
                # error relative to the size of this trajectory's final displacement
                err_t.append(np.linalg.norm(pos - P[t + 1]) / np.linalg.norm(P[-1] - pos0))
            errs.append(err_t)
    errs = np.array(errs)
    print(f"Rollout over {args.n_timesteps} steps, {len(test_ids)} unseen trajectories: relative displacement "
          f"error at the last step, median {np.median(errs[:, -1]):.3%}, worst {errs[:, -1].max():.3%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "edge_index": torch.tensor(edge_index_np),
        "normalizers": {name: {"mean": torch.tensor(nz.mean), "std": torch.tensor(nz.std)}
                        for name, nz in (("node", node_norm), ("edge", edge_norm), ("y", y_norm))},
        "history": history,
        "rollout_relative_error": torch.tensor(errs),
        "test_trajectories": torch.tensor(test_ids),
    }, args.out)
    print(f"Saved {args.out}")

    # Figure: final shapes of one test trajectory, and rollout error vs step for all test trajectories
    t_id = test_ids[0]
    P, V = trajs[t_id]
    drive_v = disps[t_id] / (args.n_timesteps * DT)
    pos, vel = P[0].copy(), V[0].copy()
    with torch.no_grad():
        for t in range(args.n_timesteps):
            nf, ef = features(pos, vel, node_type, edge_index_np, rest)
            acc = y_norm.inverse(predict(nf[None], ef[None]).numpy())
            vel = vel + DT * np.where(free[:, None], acc, 0.0)
            pos = pos + DT * vel
            apply_bc(pos, vel, pos0, node_type, drive_v, t + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (u, v) in edge_index_np.T[: edge_index_np.shape[1] // 2]:
        axes[0].plot(*P[-1][[u, v]].T, color="0.6", lw=0.6)
        axes[0].plot(*pos[[u, v]].T, color="tab:red", lw=0.6, ls="--")
    axes[0].plot([], [], color="0.6", label="simulator"); axes[0].plot([], [], "r--", label="MeshGraphNet rollout")
    axes[0].set_aspect("equal"); axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title(f"Test trajectory {t_id}: shape after {args.n_timesteps} steps")
    for e_ in errs:
        axes[1].plot(e_, color="tab:blue", alpha=0.4)
    axes[1].set_xlabel("rollout step"); axes[1].set_ylabel("relative displacement error")
    axes[1].set_title("Rollout error, all test trajectories"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=120)
    print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
