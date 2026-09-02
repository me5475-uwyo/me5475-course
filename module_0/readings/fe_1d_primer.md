# A 1-D Finite Element Primer

*ME-5475 · Module 0. For students who want to refresh FEA — or who have never taken an FEA course and need the gist before we meet MOOSE (the MOOSE cantilever reading, Lecture 3) and the 2-D weak form (the weak-form reading).*

This reading walks the **entire finite element machine** — strong form → weak form → discretization → element matrices → assembly → boundary conditions → solve → post-process → convergence — in the simplest setting where every step fits on one line: a 1-D elastic bar. The runnable counterpart is `module_0/examples/fe1d_reference.py` (~55 lines; numpy for the solver, matplotlib for the figure), which produces every number below — rerun the convergence table yourself.

**The model problem, used throughout.** A bar on [0, 1] with axial stiffness EA = 1 carries a distributed body load b(x) = x (force per unit length), is fixed at the left end, u(0) = 0, and is pulled at the right end with traction F = 0.3.

---

## 1 · The strong form, and the two kinds of boundary condition

Force balance on a differential slice of the bar gives the **strong form**:

    −(EA u′)′ = b(x)   on (0, 1),
    u(0) = 0,          EA u′(1) = F = 0.3.

Here u(x) is axial displacement, and σ(x) = EA u′(x) is the axial force (call it stress; with unit area they coincide). The two boundary conditions are of fundamentally different types:

- **u(0) = 0** is an **essential** (Dirichlet) condition: it constrains u itself, and must be *built into the solution space by hand*. It is "essential" because the method cannot proceed without imposing it explicitly.
- **EA u′(1) = F** is a **natural** (Neumann) condition: it constrains the derivative, i.e. the stress. It is "natural" because the weak form absorbs it *automatically*, through the boundary term of an integration by parts — we never enforce it ourselves.

**The exact solution** (derive it yourself — it's two integrations). From −u″ = x: u″ = −x, so u′ = −x²/2 + C₁ and u = −x³/6 + C₁x + C₂. The essential BC u(0) = 0 gives C₂ = 0; the natural BC u′(1) = 0.3 gives −1/2 + C₁ = 0.3, so C₁ = 0.8. Therefore

    u(x) = 0.8x − x³/6,    σ(x) = EA u′(x) = 0.8 − x²/2.

Keep these; we measure every FE answer against them.

## 2 · The weak form

Multiply the strong form by a **test function** v(x) satisfying v(0) = 0 (test functions vanish wherever essential BCs are imposed), and integrate over the bar:

    ∫₀¹ −(EA u′)′ v dx = ∫₀¹ b v dx.

Integrate the left side by parts once:

    ∫₀¹ EA u′ v′ dx − [EA u′ v]₀¹ = ∫₀¹ b v dx.

The boundary term at x = 0 dies because v(0) = 0. At x = 1, EA u′(1) is exactly the natural BC — so we may *replace it by F*. The **weak form** reads:

> Find u with u(0) = 0 such that, for all v with v(0) = 0,
>
>     ∫₀¹ EA u′ v′ dx = ∫₀¹ b v dx + F·v(1).

Notice what happened: the natural BC entered through the boundary term of the integration by parts, and now lives on the right-hand side as data. This is the 1-D shadow of the weak-form reading, where ∫ σ(u) : ε(v) dV = ∫ b·v dV + ∫ t·v dA — one dimension collapses each piece to the line above.

## 3 · Discretization: mesh, nodes, hat functions

Divide [0, 1] into N equal **elements** of length h = 1/N, with **nodes** x₀ = 0, x₁ = h, …, x_N = 1. Approximate the displacement as a combination of **shape functions**:

    u_h(x) = Σⱼ uⱼ Nⱼ(x),

where the unknowns uⱼ are the *nodal displacements* and Nⱼ(x) is the piecewise-linear **hat function** of node j: it equals 1 at node j, slopes linearly to 0 at the two neighbors, and is zero elsewhere — a triangular tent of height 1 pitched over node j, its base spanning the two elements that touch it:

    N_j:        1
               /\
              /  \
    ____0____/    \____0____
           x_{j−1}  x_{j+1}

Two consequences: (i) u_h *interpolates its own coefficients* — u_h(xⱼ) = uⱼ; (ii) each hat overlaps only its immediate neighbors, which is why the matrices below are sparse. Galerkin's method: use the same hats as test functions v, and require the weak form to hold for each.

## 4 · The element stiffness matrix

Everything is computed element by element. On one element [x₁, x₂] (length h), only two shape functions are alive — the descending half of the left node's hat and the ascending half of the right node's:

    N₁(x) = (x₂ − x)/h,   N₂(x) = (x − x₁)/h,   so   N₁′ = −1/h,   N₂′ = +1/h.

The element's contribution to the stiffness term ∫ EA u′ v′ dx is the 2×2 **element stiffness matrix** with entries k_ab = ∫ₓ₁ˣ² EA N_a′ N_b′ dx. Since the derivatives are constant, each integral is EA · (±1/h)(±1/h) · h = ±EA/h:

    k_e = (EA/h) · [  1  −1 ]
                   [ −1   1 ]

Sanity checks: it is symmetric (the weak form is), and its rows sum to zero — translating both nodes equally stretches nothing and costs no force.

## 5 · The consistent load vector

The element's contribution to ∫ b v dx is the **load vector** f_a = ∫ₓ₁ˣ² b(x) N_a(x) dx. For our b(x) = x, write b along the element via the same shapes, b(x) = x₁N₁ + x₂N₂ (exact here, since b is linear), and use ∫N_aN_b dx = (h/6)·[[2,1],[1,2]]:

    f_e = (h/6) · [ 2x₁ + x₂ ]
                  [ x₁ + 2x₂ ]

This is the **consistent** load: each node receives the share of b that its hat actually "feels." Contrast naive **lumping** — total element force h·(x₁+x₂)/2, split evenly — which ignores that b is bigger toward x₂. Lumping is a legitimate approximation, but it is not what the weak form says, and it costs accuracy. Use the consistent vector.

## 6 · Assembly

Element e connects global nodes e and e+1 (0-based). Assembly **scatters** each 2×2 element matrix into the global (N+1)×(N+1) matrix K at rows/columns (e, e+1), summing overlaps. Concretely, for N = 2 (nodes 0, 1, 2; h = 1/2, EA/h = 2):

    element 0 → rows/cols (0,1)          element 1 → rows/cols (1,2)

    K = 2 · [  1  −1   0 ]
            [ −1  1+1 −1 ]      ← node 1 gets a diagonal contribution from BOTH elements
            [  0  −1   1 ]

The load vector assembles the same way: f₁ receives contributions from both elements touching node 1, and finally `f[N] += F` adds the end traction (that F·v(1) term — the natural BC arriving "naturally," as promised). In the reference script this whole section is four lines inside one loop over elements.

Note that K as assembled is **singular**: every row sums to zero, so the constant vector (1, 1, …, 1) is in its null space. Physically: nothing yet anchors the bar, and a rigid translation costs no energy. This is why we cannot solve yet —

## 7 · Imposing the essential BC

— we must impose u₀ = 0. Simplest correct method: **delete row 0 and column 0** of K, and entry 0 of f, then solve the reduced N×N system. Why is that legitimate? Deleting the *column* substitutes the known u₀ = 0 into every remaining equation. Deleting the *row* discards the equation tested with N₀ — but v(0) = 0 means N₀ was never an admissible test function, so that equation was never part of the weak form; its residual is just the support reaction, recoverable afterwards. The reduced matrix is symmetric positive definite: the anchor removed the rigid mode. (Penalty and Lagrange-multiplier alternatives exist; we introduce them later, when a problem needs one.)

## 8 · Solve, then post-process

Solve Ku = f (in the script, `np.linalg.solve`; at MOOSE scale, PETSc). Post-processing then recovers stress:

    σ_h = EA u_h′ = EA (u_{i+1} − u_i)/h   — constant on each element.

Differentiation *loses one order*: a piecewise-linear displacement has a piecewise-**constant** derivative, so the smooth exact parabola σ(x) = 0.8 − x²/2 becomes a **staircase** that jumps at every node. See `figures/fe1d_N4.png`: the left panel shows the N = 4 piecewise-linear displacement passing *through* the exact curve at the nodes; the right panel shows the four-step staircase stress straddling the exact parabola. Section 9 measures exactly what that lost order costs.

## 9 · Convergence: the measured table

Errors measured on ARCC by `fe1d_reference.py`. N = number of elements; **nodal** = max |u_h − u| at the nodes; **L2** = displacement error in the L² norm; **midpoint stress** = stress error sampled at element midpoints; **true energy norm** = (σ_h − σ)² integrated over each element by 20-point Gauss quadrature. The script prints both stress columns, so you can reproduce both rates yourself:

| N  | nodal error | L2 (displacement) | midpoint stress | true energy norm |
|----|-------------|-------------------|-----------------|------------------|
| 2  | 1.110e-16 | 1.278e-02 | 1.042e-02 | 8.122e-02 |
| 4  | 4.441e-16 | 3.269e-03 | 2.604e-03 | 4.141e-02 |
| 8  | 1.110e-16 | 8.220e-04 | 6.510e-04 | 2.080e-02 |
| 16 | 1.110e-16 | 2.058e-04 | 1.628e-04 | 1.041e-02 |
| 32 | 9.881e-15 | 5.146e-05 | 4.069e-05 | 5.208e-03 |
| 64 | 8.327e-15 | 1.287e-05 | 1.017e-05 | 2.604e-03 |

Measured ratios per mesh halving: **L2 displacement 3.97×**, **midpoint stress 4.00×**, **true energy norm 2.00×**. Two facts deserve honest explanation.

**Fact 1: the nodal errors are machine zero** (10⁻¹⁶ to 10⁻¹⁴ — floating-point round-off, with *no trend in N*). Linear FE is **nodally exact** here: with EA constant in 1-D, the Green's function of −(EA u′)′ (the response to a point load) is piecewise linear — a triangle — and so lies *inside* the FE space. Galerkin orthogonality then forces the error to vanish at every node: the FE solution *interpolates* the exact solution there. **Warning:** this is a special gift of 1-D problems of this form. In 2-D the Green's function is not piecewise polynomial and nodal exactness is gone — do not expect your MOOSE cantilever to hit exact nodal values.

**Fact 2: displacement and stress converge at *different rates*.** This is the single most useful lesson in the reading:

- **Displacement, O(h²).** The L2 error falls 3.97× per halving — halve h, quarter the error. The textbook rate for linear elements, now measured.
- **Stress in the true energy norm, O(h).** It falls only 2.00× per halving — halve h, *halve* the error. This, not O(h²), is the honest convergence rate of FE stress.
- **Midpoint stress, O(h²) — a superconvergent exception, not the general rate.** Sampled exactly at element midpoints, the constant element stress matches the exact stress to O(h²) and so falls 4.00× per halving. But the staircase is only O(h) accurate away from those points, which is what the energy norm measures. Never quote the midpoint rate as "the" stress convergence rate.

The reason is the one from step 8: **stress is one derivative down from displacement, so it converges one order slower.** Differentiating a piecewise-linear u_h throws away an order of accuracy, and no amount of post-processing puts it back. This is precisely why FE stress is less trustworthy than FE displacement — and it is why the MOOSE cantilever reading validates the MOOSE cantilever against beam theory on **tip deflection** rather than on peak stress. Deflection is what the method computes best, peak stress at a clamped corner what it computes worst; when you judge a simulation, know which of the two you are looking at.

## 10 · The map to MOOSE

Every block of a MOOSE input file (the MOOSE cantilever reading) is one of the steps above, industrialized:

| This reading | MOOSE block |
|---|---|
| Step 3 — mesh, nodes, shape functions | `[Mesh]` (+ element order in `[Variables]`) |
| Steps 4–6 — element matrices, loads, assembly | `[Kernels]` / TensorMechanics — you state the PDE; MOOSE builds k_e, f_e and assembles |
| Steps 1 & 7 — essential and natural BCs | `[BCs]` — `DirichletBC` vs `NeumannBC`/`Pressure`, same distinction, same names |
| Step 8 — solve Ku = f | `[Executioner]` (PETSc underneath) |
| Step 8 — post-processing σ_h | `[AuxKernels]` / `[Postprocessors]` / `[Outputs]` |

When you open `cantilever_beam.i` in Lecture 3, read it structurally: it is this document, written in input-file syntax, in 2-D.

---

## Questions for the instructor

None open. Both earlier questions were resolved by the instructor on 2026-09-01: essential BCs stay elimination-only, and the energy-norm query was confirmed as a real defect in the reference script — script, `measured_results.md` and section 9 now carry both stress measures and both rates.
