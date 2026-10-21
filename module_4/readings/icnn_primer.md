# Input-Convex Neural Networks (ICNN): a Primer

*Course reading · ME 5475 · Module 4 · **[Primary]**, after L21 (Wed Oct 21), before L22. Companion code:
`module_4/examples/icnn_hyperelastic.py` and `module_4/examples/icnn_rank_one_check.py`. Every computed number
below is recorded in M4 measured results, §4.*

**The one sentence to keep:** an ICNN is convex in its **inputs**, for every weight. That is not the same as a
**polyconvex** energy ψ(F), and the course's ICNN is not polyconvex by construction. You check it.

---

## What an ICNN is

A neural network architecture provably convex in its inputs, for any choice of weights satisfying simple constraints.

**Why it matters.** A convex function has no spurious local minima: every local minimum is a global one. For hyperelasticity the relevant property is **polyconvexity** (Ball 1977): ψ(F) = Φ(F, cof F, det F) with Φ convex. Polyconvexity implies rank-one convexity, i.e. ellipticity in the non-strict sense, and, **with appropriate coercivity/growth hypotheses and admissible boundary data**, the existence of energy minimizers. It does **not** imply strict ellipticity or uniqueness, and the growth condition (ψ → ∞ as J → 0⁺) is an extra assumption, not a consequence (L21 slides 3–5; the next section works it out).

---

## Ellipticity and polyconvexity, in more detail (L21 slides 3–5)

**The tangent and the acoustic tensor.** The tangent is ℂ = ∂P/∂F = ∂²ψ/∂F∂F, with components ℂ_ijkl = ∂P_ij/∂F_kl. For a unit direction n, the acoustic tensor is

A_ik(n) = n_j ℂ_ijkl n_l.

**Why it is called that.** Linearize the equations of motion, Div P = ρ₀ ü (ρ₀ > 0 the reference mass density), about a homogeneous deformed state (or freeze the tangent locally), and look for a small plane-wave displacement u = a f(n · X − c t) with an arbitrary profile f. Substituting gives ρ₀ c² a = A(n) a. So a wave can travel in direction n only with a polarization a that is an eigenvector of A(n), and ρ₀ c² is the matching eigenvalue: **the squared wave speeds are the eigenvalues of A(n)/ρ₀.** (They are speeds in reference coordinates, since the phase is written in X.)

**Strong ellipticity** means A(n) is positive definite for every unit n, at every F. Then every wave speed is real and positive. Since a · A(n) a = (a⊗n) : ℂ : (a⊗n), this is the same as the **strict Legendre–Hadamard condition**, positive rank-one curvature: (a⊗n) : ℂ : (a⊗n) > 0 for all nonzero a and n. (Some mechanics texts call it strict rank-one convexity; strict convexity of each rank-one restriction in the ordinary sense is weaker, e.g. Σ F_ij⁴ at F = I.) The non-strict version, ≥ 0, is the Legendre–Hadamard condition.

**What losing it means.** If an eigenvalue of A(n) reaches zero, a wave speed goes to zero. A discontinuity in the strain across a plane with normal n can then stand still instead of propagating. That **can permit** constitutive-level localization (shear bands, kink bands) and can lead to mesh dependence in an unregularized local finite element model. A zero eigenvalue does not by itself guarantee that a band forms, and it is a local condition, distinct from buckling or other global instabilities. Strong ellipticity rules the local mode out.

**A check you can do by hand.** In small-strain isotropic elasticity, ℂ = λ I⊗I + 2μ I^sym, where I^sym is the fourth-order symmetric identity (I^sym : ε = ε for symmetric ε), and

A(n) = (λ + μ) n⊗n + μ I.

Its eigenvalues are λ + 2μ, with a parallel to n (the pressure wave), and μ, twice, with a perpendicular to n (the two shear waves), so c_P² = (λ + 2μ)/ρ₀ and c_S² = μ/ρ₀. Linear isotropic elasticity is strongly elliptic **if and only if μ > 0 and λ + 2μ > 0**. That is weaker than positive definiteness of ℂ on symmetric strains, which needs μ > 0 and 3λ + 2μ > 0: for example λ = −1, μ = 1 gives A = I, strongly elliptic, while 3λ + 2μ = −1.

**Why it is hard to build into a network.** Ellipticity constrains the **second** derivative of ψ, for every F and every direction n. A network can be checked for it at sample points (Way 1), but nothing in an ordinary architecture guarantees it. That is why the field aims for a condition one level up, which a network can be built to satisfy.

**Polyconvexity** (Ball 1977). ψ(F) is polyconvex if

ψ(F) = Φ(F, cof F, det F) with Φ convex in its three arguments jointly. (Φ is a scalar function, not a network weight. The literature usually writes W; this primer uses Φ because W_y and W_z are the ICNN's weights.)

The three arguments are what F does to line, area and volume elements:

| argument | maps | formula |
|---|---|---|
| F | line elements | dx = F dX |
| cof F = J F⁻ᵀ | area elements | n da = (cof F) N dA (Nanson) |
| det F = J | volume elements | dv = J dV |

**Why not simply ask for convexity in F?** The set {F : J > 0} is not convex. diag(1, 1, 1) and diag(−1, −1, 1) both have J = 1. On the segment F(t) = diag(1 − 2t, 1 − 2t, 1) between them, J(t) = (1 − 2t)² → 0⁺ as t → ½. A function convex in F is bounded above on that segment by its finite end values, so it cannot satisfy the physical requirement ψ → ∞ as J → 0⁺. Convexity in F is too strong for elasticity. Polyconvexity is Ball's relaxation: convexity in a larger set of arguments, in which the volume enters on its own.

**The chain of conditions.**

convex ⇒ polyconvex ⇒ quasiconvex ⇒ rank-one convex, which is equivalent (for C² energies) to Legendre–Hadamard (A(n) positive semidefinite).

The first three arrows are one-way in general. The last link is an equivalence for twice continuously differentiable energies: along any rank-one segment, the Legendre–Hadamard condition says exactly that the second derivative is nonnegative. **Strictness is not inherited:** ψ = (det F − 1)² is polyconvex (a convex function of det F), but along the volume-preserving simple shear F(t) = I + t e₁⊗e₂ it is constant. Its curvature there is zero, so it fails the strict Legendre–Hadamard condition: it is not strongly elliptic.

**What polyconvexity buys.** With appropriate coercivity/growth hypotheses and admissible boundary data, the total potential energy has a **minimizer** (Ball 1977): the boundary-value problem has a solution. It does **not** give uniqueness, and the growth condition (for example ψ → ∞ as J → 0⁺) is an extra assumption, not a consequence.

**Examples.**

- **Compressible neo-Hookean** with μ, λ > 0: polyconvex on the course's sampling window 0.5 < J < 1.8. Its volumetric part h(J) = −μ ln J + (λ/2)(ln J)² has h″(J) = (μ + λ(1 − ln J))/J², positive there. It is **not** polyconvex for all J > 0: h″ < 0 once ln J > 1 + μ/λ (M4 ledger §4).
- **St. Venant–Kirchhoff**, ψ = (λ/2)(tr E)² + μ tr E², is **not** polyconvex (Raoult 1986), and it loses ellipticity under strong compression. This is one reason it is avoided for large strains.

**One sufficient route with invariants.** I₁ = |F|² is convex in F, I₂ = |cof F|² is convex in cof F, and −ln J is convex in J. A function convex in (I₁, I₂, J, −ln J) and **non-decreasing in I₁, I₂ and −ln J** is a convex function of (F, cof F, J), so it is polyconvex. (The composition rule: a convex, non-decreasing function of convex functions is convex; J enters affinely, so it may enter either way.) This is the route of Klein et al. 2022, and the reason the course's ICNN, which is convex but not monotone in its inputs, is not polyconvex by construction.

---

## The construction

A 3-hidden-layer ICNN ψ(y) mapping y ∈ ℝⁿ → ℝ:

```
   y --(W_y_first)--> z_1 = softplus(W_y_first y)
   y --(W_y_skip_1)----+
                       v
   z_1 --(W_z_1)--> z_2 = softplus(W_z_1 z_1 + W_y_skip_1 y)
                       |
   y --(W_y_skip_2)----+
                       v
   z_2 --(W_z_2)--> z_3 = softplus(W_z_2 z_2 + W_y_skip_2 y)
                       |
   y --(W_y_out)-------+
                       v
                 psi = W_z_out z_3 + W_y_out y
```

**The constraints:**

- **W_z_i (every "internal" weight matrix that takes a hidden state as input) has non-negative entries.**
- **W_y_skip_i and W_y_first (the input-to-hidden weights) can be arbitrary.**
- **Activations must be convex and non-decreasing.** Softplus is the standard choice: it keeps the proof and is smooth, so the stress and tangent are classical everywhere. ReLU also keeps convexity, but has kinks, which matters when a globally smooth tangent is needed. tanh and GELU are **not** convex and non-decreasing, so they break the proof, even though `sobolev_training_primer.md` lists them for ordinary (non-ICNN) Sobolev networks.

**Why this gives convexity.**

The proof is short, by induction on layer index.

*Base case (layer 1).* z_1 = softplus(W_y_first y). softplus is convex; W_y_first y is linear (so convex). Composition of convex with linear gives convex (this is the "affine in" case). z_1 is convex in y.

*Inductive step.* Suppose z_{l-1} is convex in y. Then:

```
z_l = softplus(W_z_{l-1} z_{l-1} + W_y_skip_{l-1} y)
```

- W_z_{l-1} z_{l-1} is a NON-NEGATIVE weighted sum of convex functions z_{l-1}. Non-negative-weighted sums of convex functions are convex.
- W_y_skip_{l-1} y is linear (so convex).
- The sum of two convex functions is convex.
- softplus is non-decreasing and convex. A non-decreasing convex function of a convex function is convex (Boyd & Vandenberghe 2004, §3.2.4, composition rules).

Therefore z_l is convex in y. By induction, the final z_L and thus ψ = W_z_out z_L + W_y_out y (sum of non-negative-weighted convex + linear) is convex in y.

---

## PyTorch implementation

Simplified from `icnn_hyperelastic.py`, where the skip layers are called `W_y`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonNegLinear(nn.Module):
    """Linear with non-negative weights via exp parameterization.
    Bias is unconstrained (an arbitrary scalar shift doesn't break convexity)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # log_w lives on the reals; exp(log_w) is always positive.
        self.log_w = nn.Parameter(torch.randn(out_features, in_features) - 2.0)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, torch.exp(self.log_w), self.bias)


class ICNN(nn.Module):
    def __init__(self, n_in, hidden=32, depth=3):
        super().__init__()
        self.W_y_first = nn.Linear(n_in, hidden)
        self.W_z = nn.ModuleList([NonNegLinear(hidden, hidden) for _ in range(depth - 1)])
        self.W_y_skip = nn.ModuleList([nn.Linear(n_in, hidden) for _ in range(depth - 1)])
        self.W_z_out = NonNegLinear(hidden, 1)
        self.W_y_out = nn.Linear(n_in, 1)

    def forward(self, y):
        z = F.softplus(self.W_y_first(y))
        for W_z, W_y in zip(self.W_z, self.W_y_skip):
            z = F.softplus(W_z(z) + W_y(y))
        return (self.W_z_out(z) + self.W_y_out(y)).squeeze(-1)
```

That's the whole architecture in ~25 lines.

---

## Verifying convexity numerically

A useful sanity check after construction:

```python
def is_convex(model, n_pairs=100, n_lambdas=20):
    """Sample pairs (y1, y2) and check convexity along the line segment."""
    n_in = next(model.parameters()).shape[1]    # input dimension
    y1 = torch.randn(n_pairs, n_in)
    y2 = torch.randn(n_pairs, n_in)
    psi1 = model(y1)
    psi2 = model(y2)

    for lam in torch.linspace(0, 1, n_lambdas)[1:-1]:
        y_lam = lam * y1 + (1 - lam) * y2
        psi_lam = model(y_lam)
        # Convexity: psi(lam y1 + (1-lam) y2) <= lam psi(y1) + (1-lam) psi(y2)
        lhs = psi_lam
        rhs = lam * psi1 + (1 - lam) * psi2
        violations = (lhs > rhs + 1e-5).sum()
        if violations > 0:
            print(f"WARNING: {violations} convexity violations at lambda = {lam:.2f}")
            return False
    return True
```

For a correctly constructed ICNN, no violations should occur (within floating-point tolerance). The induction above **proves** convexity in y; this random test only checks the inequality at sampled points. **Either way it is about y, the inputs, and says nothing about polyconvexity in F.** For that, see the rank-one check below.

---

## Applying ICNN to hyperelasticity (the recipe used in `icnn_hyperelastic.py`)

**Choice of inputs.** Use centered invariants:

```
y = (I_1 - 3, I_2 - 3, J - 1, -ln J)
```

- All four are **frame-indifferent** (functions of C), so ψ(QF) = ψ(F) for every weight: L20's Way 3.
- All four vanish at F = I. That does **not** make ψ(I) = 0 or the reference state stress-free, because the biases are free.
- −ln J grows without bound as J → 0⁺. The energy blows up there only if ψ increases with −ln J at large −ln J, which this network does not enforce.

**Compose:** ψ(F) = ICNN([I_1 − 3, I_2 − 3, J − 1, −ln J]).

**Is this ψ polyconvex? Not by construction.** It is convex in y. A sufficient route to polyconvexity in F (Klein et al. 2022) also needs ψ to be **non-decreasing** in I₁, I₂ and −ln J. These are convex in F, in cof F and in J respectively; J − 1 may enter either way. The unconstrained skip weights W_y let ψ decrease in those inputs, and nothing forbids it. The condition is sufficient, not necessary: on the course window 0.5 < J < 1.8 the neo-Hookean truth is polyconvex, yet it decreases in −ln J where J > e^(μ/λ) ≈ 1.65. (It is not polyconvex for all J > 0: its volumetric term stops being convex in J once ln J > 1 + μ/λ.) To get polyconvexity by construction, make the skip weights (and the first layer) non-negative on those three inputs (L21 slide 17). The course does not implement that.

**Check, don't trust the name.** Polyconvexity implies rank-one convexity: d²/dt² ψ(F + t a⊗b) ≥ 0 for every F, a and b. `icnn_rank_one_check.py` tests this at random samples. It found these results for the same architecture under three trainings (M4 ledger §4):

| trained model | test P relative error | rank-one curvature < 0 |
|---|---|---|
| H¹, 20 000 epochs (the default) | 10.2 % | 0 of 2 540 samples |
| energy only, 20 000 epochs | 42.6 % | 245 of 2 540 |
| H¹, 5 000 epochs | 657 % | 1 530 of 2 540 |
| neo-Hookean truth | — | 0 of 2 540 |

Counts are curvatures below −10⁻⁶. A value that close to zero can change sign between machines, so the check reports those separately. A clearly negative curvature **proves** a model is not polyconvex. Finding none proves nothing, because the check only looks where it samples. After H¹ training this sample found no violation; the energy-only and undertrained models have substantial ones. **The architecture alone guarantees neither outcome.**

**Stress.** P = ∂ψ/∂F, the first Piola–Kirchhoff stress, by autograd. It rotates correctly, P(QF) = Q P(F), because ψ is frame-indifferent.

**Tangent.** ∂²ψ/∂F∂F = ∂P/∂F by autograd-on-autograd. Convexity in y does **not** make this tensor positive semi-definite. ψ is convex in the invariants, not in F, and even a genuinely polyconvex energy is generally not convex in F (Ball 1977 explains why convexity in F is too strong for elasticity). What polyconvexity does give is rank-one convexity: (a⊗b) : ∂²ψ/∂F∂F : (a⊗b) ≥ 0, the quantity the check measures.

---

## ICNN vs alternative approaches

| Approach | Polyconvex? | Frame-indifferent? | Cost |
|----------|-------------|---------------------|------|
| MLP on F | No | No | Cheap |
| MLP on invariants | No | Yes | Cheap |
| ICNN on invariants (this course) | Not guaranteed; check it | Yes | About an MLP |
| ICNN, monotone skips (Klein et al.) | By construction | Yes | About an MLP |
| CANN (Linka & Kuhl 2023) | By design | Yes | Cheap |
| Reduced-order polynomial | If the polynomial is | Yes | Cheap, limited |

"Monotone skips" means non-negative skip weights on I₁, I₂ and −ln J. The network is then less flexible, because it can no longer represent an energy that decreases in those inputs.

For mechanics, an ICNN on invariants is a common choice for hyperelasticity with physics constraints, with the monotone skips if polyconvexity must be guaranteed. CANN is a strong alternative if you want a more interpretable architecture.

---

## Limitations of ICNN

- An ICNN is a *sufficient* way to get a convex function; not every convex function can be expressed by a small ICNN. Wide and deep ICNNs approximate broad classes of convex functions but may overfit.
- For *plasticity*, the stress is not a function of the current strain alone: it depends on the loading history, through internal variables. A single energy of the total strain, convex or not, cannot describe it, so an ICNN on strain is the wrong tool by itself. L22–L24 cover recurrent and yield-surface approaches.
- Polyconvexity by construction needs polyconvex inputs and the monotone construction. Not every constitutive law is polyconvex, and sometimes the right answer is "this law does not admit a polyconvex model; use a different architecture."

---

## Further reading

- Amos, Xu, Kolter, *Input Convex Neural Networks*, ICML 2017. The original paper.
- Klein, Fernandez, Martin, Neff, Weeger, *Polyconvex anisotropic hyperelasticity with neural networks*, JMPS 2022. The mechanics application.
- Linka, Kuhl, *A new family of constitutive artificial neural networks*, CMAME 2023. The CANN alternative.
- Boyd & Vandenberghe, *Convex Optimization*, Cambridge 2004. Background on convex functions; §3.2.4 for the composition rules.
- Ball, *Convexity conditions and existence theorems in nonlinear elasticity*, ARMA 63, 1977. Polyconvexity and existence.
- Raoult, *Non-polyconvexity of the stored energy function of a Saint Venant–Kirchhoff material*, Aplikace Matematiky 31(6), 417–419, 1986.
