# Module 7 — Reading List

---

## Before Lecture 32 (assigned in Lecture 31, Fri Nov 13)

### [Primary] Lu, Jin, Pang, Zhang & Karniadakis, *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, Nature Machine Intelligence 3, 218–229, 2021.

DeepONet paper. Read Sections 1–3. Free version: arXiv 1910.03193.

### [Primary] `module_7/readings/deeponet_primer.md`

Companion reference; released with Lab 7, Mon Nov 16.

### [Optional] Chen & Chen, *Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems*, IEEE Trans. Neural Networks 6(4), 911–917, 1995.

The original universal-approximation theorem for operators (an existence result).

---

## Before Lecture 33 (assigned in Lecture 32, Mon Nov 16)

### [Primary] Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar, *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.

FNO paper. Read Sections 1–4 (Section 4 defines the Fourier layer). Free version: arXiv 2010.08895.

### [Primary] `module_7/readings/fno_primer.md`

Companion reference; released with Lab 7, Mon Nov 16.

### [Optional] Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs*, JMLR 24(89), 1–97, 2023. The unified theory.

---

## During Lab 7 (assigned in L33, Wed Nov 18; Lab 7 due Mon Nov 23, 11:59 PM MT)

### [Primary] Wang, Wang & Perdikaris, *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets*, Science Advances 7(40), eabi8605, 2021.

The physics-informed extension: the PDE residual plus the initial/boundary-condition losses go into the training loss.

### [Primary] Lu, Meng, Cai, Mao, Goswami, Zhang & Karniadakis, *A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data*, CMAME 393, 114778, 2022.

DeepONet vs FNO on a range of benchmarks. Also the source of POD-DeepONet (Section 3.1.4: the trunk replaced by POD modes of the training outputs).

### [Optional] Goswami, Bora, Yu & Karniadakis, *Physics-informed deep neural operator networks*, arXiv 2207.05748, 2022. Review.

### [Optional] Boullé & Townsend, *A mathematical guide to operator learning*, Handbook of Numerical Analysis 25, 83–125, 2024 (arXiv 2312.14688). A broad survey.

---

## Reference

### Mechanics applications of operator learning

- Mao, Lu, Marxen, Zaki & Karniadakis, *DeepM&Mnet for hypersonics: Predicting the coupled flow and finite-rate chemistry behind a normal shock using neural-network approximation of operators*, J. Comput. Phys. 447, 110698, 2021.

### Modern neural operators (for final project)

- Tripura & Chakraborty, *Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems*, CMAME 404, 115783, 2023. Wavelets in place of the Fourier basis.
- Li, Huang, Liu & Anandkumar, *Fourier neural operator with learned deformations for PDEs on general geometries* (Geo-FNO), JMLR 24(388), 1–26, 2023. FNO on irregular domains, via a learned deformation to a regular latent grid.
- Lanthaler, Mishra & Karniadakis, *Error estimates for DeepONets: a deep learning framework in infinite dimensions*, Trans. Math. Appl. 6(1), tnac001, 2022. Theoretical understanding.

### Software

- `deepxde` — supports DeepONet natively.
- `neuraloperator` — FNO and related neural operators, from the FNO authors' group; `pip install neuraloperator`.
- NVIDIA PhysicsNeMo (formerly Modulus) — production-grade operator learning; heavyweight.
