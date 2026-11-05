# Module 6 — CNNs for Field Prediction

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 3 lectures: L28 Fri Nov 6, L29 Mon Nov 9, L30 Wed Nov 11
**Lab 6:** released Thu Nov 5, due Mon Nov 16, 11:59 PM MT

---

## What this module does

The course pivots from constitutive modeling (M4–M5) to *image-based* mechanics surrogates. CNNs take 2-D images as input — e.g., a binary microstructure of matrix + inclusions — and produce either scalar properties (effective modulus) or full fields (stress map).

By the end of M6 students can:

- Build a convolutional neural network from scratch in PyTorch.
- Generate a dataset of random RVE microstructures and their apparent moduli E_app with MOOSE on ARCC (a mixed-BC tension
  test of each cell; not periodic homogenization: see the template's header).
- Train a CNN regressor that predicts E_app from a microstructure image.
- (Stretch) train a U-Net that predicts the full stress field from a microstructure image.

**Module deliverable (Lab 6).** A trained CNN regressor on a student-generated dataset of 500 random RVE microstructures; validated on a held-out test set; bonus task extending to U-Net.

---

## Where this module fits

**Back to M1 and M2:** Lab 6's CNN training reuses the M1 PyTorch idioms and the M2 Optuna pattern.

**Back to M0:** Lab 6's data generation runs MOOSE 500 times on ARCC — students reuse the M0 SLURM array pattern.

**Forward to M7:** Operator learning (DeepONet, FNO) generalizes CNN image-to-field surrogates to function-space inputs. The same RVE problem returns in M7's first lecture as a comparison.

**Forward to M8:** Graph neural networks handle non-image (irregular mesh) inputs to similar effects.

**Forward to final project:** CNN-based microstructure surrogates are a natural midterm-extension or final-project choice.

---

## File index

```
module_6/
├── README.md                                this file
├── lectures/
│   ├── L28_cnn_basics.md                    (Fri Nov 6)  convolutions, pooling, the LeNet→ResNet arc
│   ├── L29_image_to_field_mechanics.md      (Mon Nov 9)  microstructure to property: the Lab 6 pipeline
│   └── L30_unet_full_field.md               (Wed Nov 11) U-Net for image-to-image; full stress field
├── examples/
│   ├── generate_microstructures.py          random ellipsoidal-inclusion RVE generator + MOOSE driver
│   ├── rve_homogenization.i.template        MOOSE input template with spatially-varying E(x,y)
│   ├── cnn_microstructure_to_modulus.py     CNN regressor (Lab 6 main)
│   ├── unet_microstructure_to_stress.py     U-Net (Lab 6 stretch)
│   ├── cnn_optuna_sweep.py                  HP sweep (one Optuna worker)
│   ├── cnn_optuna_sweep.sbatch              ten sweep workers on ARCC
│   ├── cnn_train.sbatch                     training job (GPU): modulus or unet
│   ├── data_generation.sbatch               array job: 10 workers × 50 RVEs
│   ├── aggregate_parts.py                   combines the workers' parts into rve_dataset.npz
│   ├── rve_homogenization.i                 one rendered example of the template
├── homework/
│   ├── lab_6.md
│   └── starter_prompts.md
└── readings/
    ├── reading_list.md
    ├── cnn_architecture_primer.md           conv, pool, receptive field in 2 pages
    └── unet_for_mechanics.md                U-Net architecture + skip-connection rationale
```

---

## ARCC notes

- Data generation uses the course MOOSE build `/project/me5475/software/rom_opt_arcc/rom_opt-opt` with the course
  runtime modules (gcc 14 libstdc++ first), as set in `data_generation.sbatch`.
- Training uses the course environment `/project/me5475/envs/ml4sm` (PyTorch 2.5.1 with CUDA) on the GPU partitions;
  both training scripts print the device they ran on. The Optuna sweep runs on CPU nodes.
- Validated on MedicineBow on 2026-11-05 (`module_6/readings/measured_results.md`): the handout's commands for data
  generation, aggregation, training, the sweep and retraining, end to end; the U-Net stretch path on a 100-RVE field
  set only.
