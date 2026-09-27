# Plane Stress and Plane Strain — A Primer

*Module 3. Read it before Lecture 14 (Fri Oct 2) and before Lab 3 Task 1, where you compare Min Lin's notebook with the course's port. Companions: `module_3/examples/min_lin_2D_hole_example.ipynb` (plane stress) and `module_3/examples/plate_with_hole_fixed.py` (plane strain).*

Lab 3's plate is two-dimensional, but every real body is three-dimensional. There are two standard ways to reduce 3-D elasticity to 2-D, and **they are not the same problem**. Min Lin's notebook uses one of them: **plane stress**. The course's PyTorch port, the shared MOOSE reference, and Lab 3's reference and PINN results in `module_3/readings/measured_results.md` (§1–§6) use the other: **plane strain**. This reading shows where the difference lives in the equations and in the code, and how much it changes Lab 3's answer. Measured numbers cite `measured_results.md` by section; the elastic constants in the tables are exact arithmetic from the formulas shown.

---

## 1 · Two ways to be two-dimensional

Both idealisations keep every field a function of (x₁, x₂) only. They differ in what happens in the third direction, x₃.

- **Plane stress: a thin plate loaded in its own plane.** Its faces are free, so nothing pushes on them: σ₃₃ = σ₁₃ = σ₂₃ = 0. The plate is free to get thinner or thicker, so ε₃₃ need not vanish. Examples: a sheet-metal part, a thin test coupon, the web of a beam.
- **Plane strain: deformation out of the plane is suppressed.** ε₃₃ = ε₁₃ = ε₂₃ = 0. Whatever prevents the stretch must push or pull, so σ₃₃ = ν(σ₁₁ + σ₂₂) need not vanish. It fits a long or thick body when the loading is uniform along its length and its ends, or the surrounding material, keep it from stretching: a dam or tunnel cross-section, a pipe whose ends are held, a section deep inside a thick part. **Length alone is not enough:** a long body with free or force-loaded ends can stretch uniformly along x₃, which is *generalized* plane strain.

Everything except Hooke's law is identical in the two:

- **Equilibrium** (Lab 3's two momentum residuals): ∂σ₁₁/∂x₁ + ∂σ₁₂/∂x₂ = 0 and ∂σ₁₂/∂x₁ + ∂σ₂₂/∂x₂ = 0.
- **Kinematics:** ε₁₁ = ∂u₁/∂x₁, ε₂₂ = ∂u₂/∂x₂, ε₁₂ = ½(∂u₁/∂x₂ + ∂u₂/∂x₁).
- **Boundary conditions:** the same ten.

**The difference is entirely in the three constitutive lines** — the three Hooke residuals of the mixed formulation.

## 2 · Where they differ: Hooke's law

Start from 3-D isotropic Hooke's law, written for strains:

    ε₁₁ = [σ₁₁ − ν(σ₂₂ + σ₃₃)] / E
    ε₂₂ = [σ₂₂ − ν(σ₁₁ + σ₃₃)] / E
    ε₃₃ = [σ₃₃ − ν(σ₁₁ + σ₂₂)] / E
    ε₁₂ = (1 + ν) σ₁₂ / E

**Plane stress** sets σ₃₃ = 0. Solving the first two lines for the stresses gives:

    σ₁₁ = E/(1 − ν²) · (ε₁₁ + ν ε₂₂)
    σ₂₂ = E/(1 − ν²) · (ν ε₁₁ + ε₂₂)
    σ₁₂ = E/(1 + ν) · ε₁₂ = 2μ ε₁₂
    ε₃₃ = −ν/(1 − ν) · (ε₁₁ + ε₂₂)        (the plate thins as it stretches)

**Plane strain** sets ε₃₃ = 0. The third line then gives σ₃₃ = ν(σ₁₁ + σ₂₂). Substituting that into the first two lines and solving gives:

    σ₁₁ = (λ + 2μ) ε₁₁ + λ ε₂₂
    σ₂₂ = λ ε₁₁ + (λ + 2μ) ε₂₂
    σ₁₂ = 2μ ε₁₂
    σ₃₃ = ν(σ₁₁ + σ₂₂)                   (the ends push back)

Here λ = Eν / ((1 + ν)(1 − 2ν)) and μ = E / (2(1 + ν)) are the Lamé constants, so λ + 2μ = E(1 − ν) / ((1 + ν)(1 − 2ν)). This λ is a material constant; it is not a loss weight (`team/NOTATION.md` keeps the two apart).

For Lab 3's material, E = 1 and ν = 0.3:

| entry | plane stress (Min's notebook) | plane strain (port, MOOSE, Lab 3) |
|---|---|---|
| C₁₁: σ₁₁ per unit ε₁₁ | E/(1 − ν²) = 1.0989 | λ + 2μ = 1.3462 |
| C₁₂: σ₁₁ per unit ε₂₂ | Eν/(1 − ν²) = 0.3297 | λ = 0.5769 |
| C₃₃: σ₁₂ per unit ε₁₂ | E/(1 + ν) = 0.7692 | 2μ = 0.7692 |
| σ₃₃ | 0 | ν(σ₁₁ + σ₂₂) |
| ε₃₃ | −ν(ε₁₁ + ε₂₂)/(1 − ν) | 0 |

The shear entry is the same in both. **Plane strain is stiffer in the normal entries:** C₁₁ is 22.5 % larger and C₁₂ 75 % larger, because the out-of-plane constraint adds a σ₃₃ that feeds back through ν.

## 3 · One formula, two sets of constants

The two models share one formula; only the constants passed into it change.

- **Plane strain with (E, ν)** is plane stress with **E∗ = E/(1 − ν²)** and **ν∗ = ν/(1 − ν)**. For Lab 3 that is E∗ = 1.0989 and ν∗ = 0.4286. Put them into the plane-stress C₁₁ and C₁₂ and you get exactly 1.3462 and 0.5769.
- **Plane stress with (E, ν)** is plane strain with **E′ = E(1 + 2ν)/(1 + ν)²** and **ν′ = ν/(1 + ν)**. For Lab 3 that is E′ = 0.9467 and ν′ = 0.2308.

The second line is how the course measured Min's plane-stress problem with MOOSE without writing a new input file (`measured_results.md` §8). The shared plane-strain input was run with E′ and ν′ in place of E and ν. The mapping is exact for the in-plane fields: u₁, u₂, σ₁₁, σ₂₂ and σ₁₂. The out-of-plane quantities, σ₃₃ and ε₃₃, and anything built from them, such as the von Mises stress, must be recomputed with the true model.

## 4 · How to recognise each one in code

**Min's notebook (plane stress),** in its `pde` cell:

```python
C11 = E/(1-nu*nu)
C12 = E*nu/(1-nu*nu)
C33 = E/(1+nu)
eq_3 = S11 - C11*E11 - C12*E22
```

**The course's port (plane strain),** `module_3/examples/plate_with_hole_fixed.py`:

```python
LAMBDA_LAME = E_VALUE * NU_VALUE / ((1 + NU_VALUE) * (1 - 2 * NU_VALUE))
MU_LAME = E_VALUE / (2 * (1 + NU_VALUE))
r_cst_1 = S11 - (LAMBDA_LAME * trace_eps + 2 * MU_LAME * eps_11)
```

**The MOOSE reference:** `planar_formulation = PLANE_STRAIN` in `module_3/examples/plate_square_hole_reference.i`.

**The von Mises stress** also depends on the model:
- Plane stress uses σ₃₃ = 0.
- Plane strain uses σ₃₃ = ν(σ₁₁ + σ₂₂). The port's plotting function does this, and so does Lab 3 Task 1d's figure.

**One trap that looks like a plane-stress/plane-strain difference but isn't: tensor vs engineering shear strain.**
- C₃₃ = E/(1 + ν) multiplies the **tensor** shear strain ε₁₂ = ½(∂u₁/∂x₂ + ∂u₂/∂x₁).
- With the **engineering** shear strain γ₁₂ = 2ε₁₂, the factor is μ = E/(2(1 + ν)).
- Both of Lab 3's files use the tensor ε₁₂, which is why their shear lines agree. Check this whenever you read someone else's code.

## 5 · How much it matters on Lab 3's plate

Measured with MOOSE on the same mesh: plane strain in `measured_results.md` §1, plane stress in §8.

| quantity | plane stress (Min's constants) | plane strain (Lab 3) | ratio |
|---|---|---|---|
| σ₁₁ at the hole top, (0, 0.1) | 2.9500 | 3.2418 | 0.9100 |
| reaction force on the loaded edge | 0.9768 | 1.0734 | 0.9100 |
| stress-concentration factor, σ₁₁(0, 0.1) / mean σ₁₁ on the loaded edge | 3.020 | 3.020 | 1 |

On the 200 × 200 reference grid, the relative L2 difference between the two solutions is:

- **9.0 %** in each of σ₁₁, σ₂₂ and σ₁₂;
- **29 %** in u₂, the sideways contraction;
- 0.1 % in u₁, which is pinned at both loaded edges.

**For this plate, each in-plane stress component (σ₁₁, σ₂₂, σ₁₂) is exactly 1 − ν² = 0.91 times its plane-strain value.** So the in-plane stress pattern and the stress-concentration factor are the same in both models; only the scale changes. σ₃₃ is the exception: it is zero in plane stress and generally nonzero in plane strain. Here is why the factor is exact:

- **The problem.** The plate is homogeneous, isotropic elasticity with small strain and no body force. Its left and right edges are frictionless grips at constant u₁ (0 and 1), its bottom edge is a symmetry line, and its top edge and the hole are traction-free.
- **The plane-stress stresses depend on E but not on ν.** In plane stress, the local rotation ω and the stresses are tied by equations that contain no ν: with zero body force, E·ω and σ₁₁ + σ₂₂ are harmonic conjugates. The grips and the symmetry edge each fix ω = 0 along their edge. The one remaining condition, that the right edge moves by 1 relative to the left, can be taken along the traction-free top edge, where σ₂₂ = 0: u₁(1, 1) − u₁(0, 1) = (1/E) ∫₀¹ σ₁₁(x₁, 1) dx₁ = 1. It contains no ν either.
- **From that to the factor.** Plane strain with (E, ν) is plane stress with (E∗, ν∗) (§3). Because this plate's stresses do not depend on ν, changing ν to ν∗ changes nothing; changing E to E∗ = E/(1 − ν²) scales every in-plane stress by 1/(1 − ν²).
- **It depends on these boundary conditions.** It is not a general conversion factor. For example, on a unit square **without a hole**, prescribe u₁ = 0 and 1 on the left and right, u₂ = 0 on the bottom and top, and zero shear on all four edges. The solution u₁ = x₁, u₂ = 0 gives plane-stress stresses σ₁₁ = E/(1 − ν²) and σ₂₂ = Eν/(1 − ν²), which depend on ν.
- **A plain bar with free sides shows the scale:** σ = E·ε in plane stress and σ = E/(1 − ν²)·ε in plane strain.
- **The displacements do change.** The relative L2 difference in u₂ is 29 %, because the sideways contraction follows the effective Poisson ratio: ν = 0.3 in plane stress but ν/(1 − ν) = 0.43 in plane strain.

The MOOSE runs agree with this exact result within numerical error. On the grid, the scaled difference ‖σ_ps − 0.91 σ_pe‖ / ‖σ_pe‖ is between 2e-10 and 1e-8 for each in-plane component (§8).

In plane strain the out-of-plane stress σ₃₃ = ν(σ₁₁ + σ₂₂) is not small either: on the grid its size is about 30 % of σ₁₁ (relative L2), with a maximum of 0.97.

**For scale:** the soft-BC PINN runs in §6 reach a relative L2 error of about 6e-4 in u and 2e-3 in σ₁₁. (Relative L2 means on the reference grid, normalised by the reference field.) The modelling choice is far larger. In σ₁₁ it is 9 %, some 30–60 times the σ₁₁ error of every PINN run in §6. A PINN built with Min's constants and scored against the plane-strain reference would carry that error even when fully converged, because it converges to the wrong model. It is the same lesson as the missing boundary conditions in §2: a tiny loss certifies the problem you posed, not the problem you meant.

## 6 · Which one should you use?

- **Plane stress** when the body is thin in x₃ compared with its in-plane size and its faces are free: sheet metal, thin plate specimens, thin-walled parts loaded in plane.
- **Plane strain** when out-of-plane extension is suppressed. That means a long or thick body whose loading, end restraint and region of interest justify ε₃₃ = 0: dams, tunnels, pipes with held ends, cross-sections deep inside thick parts. A long body with free ends may instead need generalized plane strain, with a uniform ε₃₃. Fracture-toughness tests, for example, are designed so the crack tip is in plane strain.

Neither model is more correct in general; each is an idealisation. What matters in a PINN project is that **the model and its reference solve the same problem**. Lab 3 fixed that problem as plane strain, consistently across the port, the MOOSE input and the measurements. So read Min's notebook knowing that its constants are plane stress, and **keep the port plane strain** (Task 1a).

## Check yourself

1. From ε₃₃ = 0, show that σ₃₃ = ν(σ₁₁ + σ₂₂). *(Set the third line of 3-D Hooke's law to zero.)*
2. Show that plane strain with (E, ν) is plane stress with (E/(1 − ν²), ν/(1 − ν)). *(Put E∗ and ν∗ into E∗/(1 − ν∗²) and simplify to E(1 − ν)/((1 + ν)(1 − 2ν)).)*
3. Which lines of Min's `pde` cell would you change to make it plane strain? *(The two that define C11 and C12; C21 and C22 follow from them. C33 stays.)*
4. A bar with free sides is pulled to strain ε. What stress does each model give, and what is their ratio? *(E·ε and E/(1 − ν²)·ε; ratio 1 − ν². Section 5 measures the same ratio on Lab 3's plate.)*

**Further reading.** Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. (McGraw-Hill, 1970), Chapter 2, "Plane Stress and Plane Strain". Any elasticity text's chapter on two-dimensional problems covers the same ground.
