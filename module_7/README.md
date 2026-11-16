# Module 7 — Operator Learning (DeepONet + FNO)

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 3 lectures (Wk 11 Fri → Wk 12 Wed: Fri Nov 13 → Wed Nov 18)
**Status:** Audited Nov 13, 2026 (pending Reviewer)

---

## What this module does

The conceptual leap from "models that take a fixed-shape input" to "models that take a *function* as input and output another function." This is the unifying view that ties together M3 (PINN, which can be reframed as learning the parametric solution operator) and M6 (U-Net, which is a specific discretized operator).

Two architectures:

- **DeepONet** (Lu et al., 2021) — branch + trunk + dot product.
- **Fourier Neural Operator (FNO)** (Li et al., ICLR 2021) — spectral convolution in Fourier space; *designed* to run on grids of any resolution (whether accuracy carries over is what Lab 7 Task 4 tests).

**Lab 7 deliverable** (released Mon Nov 16, due Mon Nov 23, 11:59 PM MT). Train and compare DeepONet and FNO on the 1-D Burgers' equation, a standard benchmark from the FNO paper. Cross-resolution test: both models are trained at N = 128 and evaluated on fresh test sets at N = 64, 128, 256 and 512; the lab measures how each method holds up.

**Stretch task.** Apply FNO to Lab 6's microstructure-to-stress dataset and compare to the U-Net result.

---

## Where this module fits

**Back to M3:** the parametric PINN (M3 L16) is an operator learning special case where the input "function" is a finite-dimensional vector of material parameters. M7 generalizes.

**Back to M6:** the U-Net is a *discretized* operator. M7's FNO generalizes it to an operator designed to work across grid resolutions.

**Forward to M8:** graph neural networks handle the same operator-learning problem on irregular meshes. The conceptual chain: CNN → U-Net → FNO → GNN → MeshGraphNet.

**Forward to final project:** operator learning is an active area of ML for mechanics; DeepONet and FNO are natural starting points for final projects that map one field to another.

---

## File index

```
module_7/
├── README.md
├── lectures/
│   ├── L31_deeponet.md                          (Fri Nov 13) branch + trunk + dot product
│   ├── L32_fno.md                                (Mon Nov 16) Fourier spectral convolution
│   └── L33_pinn_vs_operator_learning.md          (Wed Nov 18) when to use which
├── examples/
│   ├── generate_burgers_data.py                  1-D Burgers spectral solver (no MOOSE)
│   ├── deeponet_burgers.py                       DeepONet on 1-D Burgers
│   ├── fno_burgers.py                            FNO on 1-D Burgers
│   ├── heat_fourier_example.py                   L32: heat equation by hand and by FFT (NumPy only)
│   ├── fft_basics_examples.py                    L32: five FFT examples (fft_primer.md; NumPy only)
│   ├── fno_microstructure.py                     stretch: FNO on M6 dataset
│   ├── burgers_data.sbatch                       ARCC SLURM: Burgers data at any N
│   ├── deeponet_demo.sbatch                      ARCC SLURM: the L31 demo (data + DeepONet)
│   └── operator_train.sbatch                     ARCC SLURM: train deeponet | fno
├── homework/
│   ├── lab_7.md
│   └── starter_prompts.md
└── readings/
    ├── reading_list.md
    ├── deeponet_primer.md                        branch/trunk architecture explained
    ├── fft_primer.md                             FFT fundamentals, five worked examples (L32)
    ├── fno_primer.md                             Fourier neural operator explained
    └── measured_results.md                       source of every number quoted in M7
```

---

## A note on data generation

Lab 7 uses **numerically generated data** (a Fourier pseudo-spectral 1-D Burgers solver in NumPy), not MOOSE. This keeps the lab tractable in 1 week — Lab 6's MOOSE dataset generation was the heavy lift; M7 focuses on the *ML methods*.

The stretch task does use M6's MOOSE-generated dataset; if you saved `rve_dataset.npz` with full stress fields, FNO loads it natively.

---

## Open items

- 1-D Burgers' spectral solver: done (NumPy only). Dealiasing fix Nov 12; automatic dt reduction on grids N ≥ 256 Nov 13; validation recorded in `readings/measured_results.md` §1.
