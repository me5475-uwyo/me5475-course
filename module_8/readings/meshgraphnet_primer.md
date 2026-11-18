# MeshGraphNet Primer

Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia (DeepMind), *Learning Mesh-Based Simulation with Graph Networks*, ICLR
2021. The canonical "GNN for simulation meshes" architecture.

---

## The premise

A mesh-based simulation (finite elements, finite volumes, cloth) produces a time series of states on a mesh:

```
t = 0:  positions p(0), velocities v(0), features f(0)
t = 1:  positions p(1), velocities v(1), features f(1)
...
```

Each time step computes a local update — typically based on stress, contact, or other local information. A *neural* model can be trained to predict the next time step given the current state, mimicking what the classical solver does, and then be rolled out step by step as a surrogate. Pfaff et al. report their trained models running "1-2 orders of magnitude faster than the simulation on which it is trained" (their abstract; per-step timings against their ground-truth solvers in their Table 1). Whether a surrogate is faster for *your* problem is something to measure.

MeshGraphNet is the architecture that does this on irregular meshes.

---

## The architecture

```
Input: mesh state at time t
   |
   v
+----------+
| Encoder  |    -- per-node MLP, per-edge MLP
+----------+    -- maps raw features to latent representations
   |
   v
+--------------------+
| Processor (K steps)|   -- K steps of message passing on the mesh graph
|                    |   -- updates BOTH node and edge features
+--------------------+
   |
   v
+----------+
| Decoder  |    -- per-node MLP
+----------+    -- maps latent representation to a time derivative per node
   |
   v
Output: e.g. acceleration per node, integrated to give the state at time t+1
```

The key features:

1. **Both node and edge features.** Unlike GCN or plain GIN, which update node features only, MeshGraphNet keeps a latent vector on every edge and updates it at every step. Edge inputs are relative quantities (relative positions and their lengths), not absolute positions, so the model does not depend on where the mesh sits in space.
2. **Many message-passing steps.** Pfaff et al. use K = 15 steps per time step. Each step reaches one hop further, so K steps reach K hops (L34). The demo uses 10 (`--n-message-passing`).
3. **Predicts a time derivative, then integrates.** The decoder outputs, per node, what a solver would update: acceleration for cloth (integrated twice), a first-order rate of change for the plate and the flows (integrated once), with forward Euler (Pfaff et al., Section 3). The demo predicts acceleration and integrates with semi-implicit Euler.
4. **Supervised next-step regression.** The targets come from the simulator's own trajectories, so no extra labeling is needed, but it is ordinary supervised learning: the state at t in, the derivative at t as the target, mean-squared error. In Pfaff et al.'s words, "we only supervise on the next step in sequence."

---

## The model in `meshgraphnet_demo.py`

Lab 8's stretch demo (Task 5) is a **toy mass-spring membrane, not a finite-element model**: a 10 × 10 grid of unit
masses joined by springs (along rows, columns and both diagonals) with linear damping, stepped with semi-implicit
Euler. The bottom row is fixed; the top row is pulled sideways at a constant rate. The model, unchanged from the script:

```python
import torch
import torch.nn as nn

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
```

- **Node inputs (5):** velocity (vx, vy) and a one-hot node type (free, fixed, driven).
- **Edge inputs (4):** the current relative position x_dst − x_src (2), its length, and the spring's rest length. No
  absolute positions.
- **Output (2):** the standardized acceleration of each node. The loss covers the free nodes only; the fixed and
  driven nodes are set by the boundary conditions.

Each of the K steps first updates every edge from its own latent and its two end nodes, e' = e + MLP([e, h_src, h_dst]),
then updates every node from the sum of its incoming updated edges, h' = h + MLP([h, Σ e'_in]). Both updates are
residual, every MLP except the decoder ends in a LayerNorm, and each step has its own MLPs. Pfaff et al.'s processor has
the same structure (their Eq. 1; a separate set of parameters per step), plus the extra "world" edges they add for
contact in some Lagrangian cases.

The script standardizes inputs and targets with training statistics, splits the data **by trajectory** (80 % train,
20 % test), rolls out every test trajectory from its initial state with the model's own predictions fed back and the
boundary conditions imposed, and plots the error against the step. `--noise-std` adds Gaussian noise to the training
input velocities and leaves the acceleration targets unchanged: a simplified augmentation ablation, not the papers'
noise-correcting training (below).

---

## What Pfaff et al. demonstrated

| Domain in Pfaff et al. | Mesh | Notes |
|--------------|-------------|-----|
| Cloth: FlagSimple, FlagDynamic, SphereDynamic | Lagrangian (moves with the material) | Acceleration predicted. FlagSimple keeps a fixed mesh; FlagDynamic and SphereDynamic remesh adaptively, with the network predicting the sizing field, and add world edges for contact |
| Hyperelastic plate deformed by an actuator: DeformingPlate | Lagrangian | Velocity predicted (first order), plus stress; world edges for contact |
| Incompressible flow around a cylinder: CylinderFlow | Eulerian (fixed in space) | First-order output |
| Compressible flow around an airfoil: Airfoil | Eulerian | First-order output |

Two cases that are often confused with these:

- **Granular media and other particle systems** are not meshes. The particle-based predecessor, GNS
  (Graph Network-based Simulators; Sanchez-Gonzalez et al., ICML 2020), makes each particle a node and connects the
  particles near each other, rebuilding those edges as they move; it covers fluids, rigid solids and deformable
  materials. MeshGraphNet builds on it and adds the mesh.
- **Incompressible flow** is not made divergence-free by this architecture. Incompressibility is the *local*
  differential condition ∇·u = 0. In a projection solver, enforcing it requires a pressure solve that couples the whole
  domain; a finite-hop, unconstrained network does not automatically reproduce that coupling or enforce zero divergence
  (architectures constrained to be divergence-free by construction exist, but MeshGraphNet is not one). Check
  conservation in a learned flow; do not assume it.

The paper's solids are cloth and a hyperelastic plate. History-dependent materials (plasticity) are not among its
domains; a next-step model for them would need the internal state variables as inputs.

---

## Limitations

**Error compounding over long rollouts.** Each step's small error becomes part of the next step's input, so errors compound along a rollout. Training on one-step targets from the simulator's own states (teacher forcing) can contribute to the drift: the model never sees its own imperfect states during training, so a rollout can wander into states unlike its training data. Two mitigations from the literature: noise-correcting training, which perturbs the training inputs *and adjusts the targets* so the model learns to undo the perturbation (Sanchez-Gonzalez et al. 2020; Pfaff et al. 2021, Appendix A.2.2); or training on unrolled multi-step rollouts with the loss summed over several predicted steps. The demo's `--noise-std` is simpler: noisy input velocities with **unchanged** targets. With this integrator, if a free node's input velocity is v + η, the target that would recover the clean next velocity is a − η/Δt; keeping the clean a leaves η in the next velocity even if the network predicts its target perfectly. So treat it as an experiment whose effect on rollouts you measure, not as the published remedy. How fast the demo's error grows is what Task 5 measures.

**Runs on any mesh, but reaches only K hops per step.** The weights are shared by every node and every edge, like a convolution kernel, so a trained model runs on a mesh of any size or shape. Pfaff et al. train on rectangular flags (FlagDynamic) and roll out on fish-shaped flags and a windsock whose mesh "averages at 20k nodes, an order of magnitude more than seen in training." The limit is reach: K message-passing steps carry information only K hops per time step, so on a finer mesh the same K covers less physical distance (L34). Multi-scale message passing is one remedy (Fortunato et al. 2022).

**No conservation laws baked in.** Energy and momentum are not conserved by construction and may drift over a rollout.

**Cost per step grows with K and the mesh.** Each time step runs K edge and node MLP passes over every edge and node of the mesh.

---

## When to use MeshGraphNet for mechanics

- You have a *dataset of mesh-time-series* from a classical simulator.
- You need many rollouts (design loops, parameter sweeps, control), so the cost of generating data and training is paid back; check the speedup on your own problem rather than assume it.
- The simulation has *local* dynamics — interactions among neighboring elements.
- The mesh may differ from run to run (the weights are shared per node and edge). Remeshing during a run is possible: Pfaff et al. train the network to predict the sizing field that a standard remesher then applies.

If those apply, MeshGraphNet is a reasonable candidate.

---

## When NOT to use MeshGraphNet

- One-time solve of a single FE problem (use the classical solver: generating the training data alone takes many such solves).
- Long rollouts where errors are costly and cannot be checked: drift has to be measured (as in Task 5) and controlled (noise-correcting training, unrolled training).
- Problems dominated by long-range coupling on fine meshes, where K hops per step reach too little (multi-scale variants are an active research topic).
- Very large 3-D meshes, where memory for K message-passing steps over every edge becomes the limit.

Lab 8's demo is deliberately small (a 10 × 10 toy membrane, CPU only). How well it rolls out is the stretch task's measurement.

---

## Further reading

- Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia, *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021. The original.
- Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec & Battaglia, *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020. GNS, the particle-based predecessor (fluids, rigid solids, deformable materials).
- Fortunato, Pfaff, Wirnsberger, Pritzel & Battaglia, *MultiScale MeshGraphNets*, 2nd AI4Science Workshop at ICML 2022 (arXiv 2210.00612). Message passing on a coarser mesh as well, for high-resolution meshes.
- Allen, Rubanova, Lopez-Guevara, Whitney, Sanchez-Gonzalez, Battaglia & Pfaff, *Learning Rigid Dynamics with Face Interaction Graph Networks*, ICLR 2023.
