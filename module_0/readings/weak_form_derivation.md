# The Weak Form of Linear Elastostatics

*Read before Lecture 2 (Wed Sep 2). In class we will paste the lead prompt at the end of this document into an AI agent, watch it produce this derivation, then route the output to an independent review agent. Your job in that demo is to judge the agents' work — which you can only do if you have done the derivation yourself once, slowly.*

This same boundary-value problem returns in Module 3 as the running example for PINN forward and inverse solves on the plate with a hole, and the 1-D FE primer walks through the 1-D finite-element version of the same machinery.

## 1. The problem

Take a two-dimensional elastic body occupying a domain Ω ⊂ ℝ², with boundary Γ split into two non-overlapping parts, Γ = Γ_D ∪ Γ_N. The **strong form** of linear elastostatics is:

    ∇·σ + b = 0        in Ω                    (1)
    u = ū              on Γ_D                  (2)
    σ·n = t̄            on Γ_N                  (3)
    σ = ℂ:ε,  ε = ½(∇u + ∇uᵀ)                  (4)

Here u(x) is the displacement field (how far each material point moves), ε is the small-strain tensor — the symmetric part of ∇u, the part that deforms material rather than rigidly rotating it — σ is the Cauchy stress (force per unit area across internal surfaces), b is body force per unit volume, and ℂ is the fourth-order elasticity tensor mapping strain to stress. On Γ_D the displacement is prescribed to ū (a clamped or dragged edge); on Γ_N the traction σ·n, with n the outward unit normal, is prescribed to t̄ (an applied load, or zero on a free edge).

Equation (1) is **balance of linear momentum** for a body at rest: ∇·σ is the net internal force per unit volume from neighboring material, and it must cancel the body force everywhere so nothing accelerates. In index notation (repeated indices summed, comma meaning partial derivative): σ_ij,j + b_i = 0.

## 2. Why bother with a weak form?

Three honest reasons. First, the strong form demands second derivatives of u pointwise everywhere; real problems (corners, interfaces, point loads) do not oblige. The weak form asks only for first derivatives, in an integral sense, which admits the solutions nature actually produces. Second, the weak form is *the thing the finite element method discretizes*: restrict u and v below to finite-dimensional spaces and equation (9) becomes K d = f. Third, the traction condition (3) is never imposed on the discrete solution at all — it enters by substitution and is satisfied automatically in the limit. That is why Neumann conditions are called **natural**, and Dirichlet conditions, built into the space, **essential**.

## 3. The derivation, in six steps

These are exactly the six steps the lead prompt requests, so you can check the agent line against line.

**Step 1 — state the strong form.** Equations (1)–(4) above.

**Step 2 — multiply by an admissible test function.** Take a vector field v, arbitrary except for two requirements: smooth enough for the integrals below to exist, and **v = 0 on Γ_D** (Section 4 explains both). Dot it with the residual of (1):

    v·(∇·σ + b) = 0    for every admissible v.         (5)

**Step 3 — integrate over Ω.**

    ∫_Ω v·(∇·σ) dΩ + ∫_Ω v·b dΩ = 0.                   (6)

Nothing is lost: if (6) holds for *all* admissible v, the integrand of (1) must vanish wherever it is continuous — pointwise satisfaction is traded for satisfaction against every test function.

**Step 4 — integrate by parts.** This is where sign errors live, so watch it in both notations. In index notation, the product rule gives (v_i σ_ij),_j = v_i,_j σ_ij + v_i σ_ij,_j, and the divergence theorem converts the total-divergence term to a boundary integral:

    ∫_Ω v_i σ_ij,_j dΩ = ∫_Γ v_i σ_ij n_j dΓ − ∫_Ω v_i,_j σ_ij dΩ.     (7)

Symbolically: ∫_Ω v·(∇·σ) dΩ = ∫_Γ v·(σ·n) dΓ − ∫_Ω ∇v:σ dΩ. Note the **minus sign** on the volume term — the derivative moved off σ and onto v, at the price of a sign and a boundary term. One more simplification: σ is symmetric, so in the contraction ∇v:σ only the symmetric part of ∇v survives, i.e. ∇v:σ = ε(v):σ. Substituting (7) into (6):

    ∫_Ω ε(v):σ dΩ = ∫_Ω v·b dΩ + ∫_Γ v·(σ·n) dΓ.       (8)

**Step 5 — substitute the boundary conditions.** Split the boundary integral over Γ = Γ_D ∪ Γ_N. On Γ_D the test function vanishes, so that piece is zero — this is precisely why v was required to vanish there. On Γ_N, the strong form says σ·n = t̄, so the unknown stress on the boundary is replaced by the known data. The boundary term becomes ∫_{Γ_N} v·t̄ dΓ.

**Step 6 — state the weak form.** Insert the constitutive law σ = ℂ:ε(u). The problem reads: find u with u = ū on Γ_D such that

    a(u, v) = ℓ(v)    for all v ∈ V,                   (9)

where

    a(u, v) = ∫_Ω ε(v) : ℂ : ε(u) dΩ,
    ℓ(v)   = ∫_Ω v·b dΩ + ∫_{Γ_N} v·t̄ dΓ.

a(·,·) is bilinear (linear in each argument separately) and ℓ(·) is linear. Everything known — loads and tractions — sits in ℓ; everything involving the unknown sits in a.

## 4. Why v must vanish on Γ_D, and what space it lives in

The boundary integral in (8) runs over all of Γ, but on Γ_D we know u, not σ·n — the traction there is a *reaction*, unknown until the problem is solved. If v were nonzero on Γ_D, that unknown would pollute the right-hand side. Requiring v = 0 on Γ_D deletes the term cleanly; nothing is lost, because the displacement condition (2) is enforced directly on the trial function instead. The test space is

    V = { v ∈ H¹(Ω) : v = 0 on Γ_D },

and H¹(Ω), in one plain sentence, is the space of functions that are square-integrable with square-integrable first derivatives — exactly the regularity needed for every integral in (9) to be finite, and no more. (The trial function lives in the shifted set of H¹ functions equal to ū on Γ_D.) We develop no functional analysis beyond this; the sentence is enough to grade an agent with.

## 5. Symmetry of a(u, v)

The elasticity tensor has the major symmetry ℂ_ijkl = ℂ_klij (it comes from the existence of a stored-energy density). Then

    a(u, v) = ∫_Ω ε_ij(v) ℂ_ijkl ε_kl(u) dΩ = ∫_Ω ε_kl(u) ℂ_klij ε_ij(v) dΩ = a(v, u).

This matters twice. Discretely, it makes the stiffness matrix K symmetric — halving storage and unlocking solvers (Cholesky, conjugate gradients) unavailable to unsymmetric systems. Physically, it makes (9) equivalent to minimizing the potential energy ½a(u,u) − ℓ(u), the variational structure that PINN losses in Module 3 will imitate.

## 6. What to check — the five review items

The review prompt below asks a second agent to verify five things. Here is what each guards against, concretely.

**Dimensional consistency.** Every term in (9) must be an energy (force × length in 2-D, per unit thickness). A boundary term written with dΩ instead of dΓ, or a stress multiplying a displacement with no integral measure, silently stops being physics. Check units at the *end*, not just the start — errors accumulate at substitution steps.

**v vanishing on Γ_D, used correctly.** The usual failure is not forgetting to state the condition — it is stating it, then keeping ∫_Γ over the whole boundary in the final form, or splitting Γ and "dropping" the Γ_D term with no reason given. The drop must be explicitly tied to v|_{Γ_D} = 0.

**The Γ_N integral matches t̄.** The final linear form must contain ∫_{Γ_N} v·t̄ dΓ — the *given* traction, over Γ_N only. Common corruptions: integrating over all of Γ, writing σ·n instead of t̄ (leaving an unknown in ℓ), or a stray normal n alongside t̄.

**Signs after integration by parts.** The classic error: writing (7) without the minus sign on ∫_Ω ε(v):σ dΩ, which propagates to a(u,v) + ℓ(v) = 0 and a negative-definite stiffness. A subtler one: writing the operator as ∇·(ℂ:∇u) — full gradient, no symmetrization — where ∇·σ = ∇·(ℂ:ε(u)) belongs. With ℂ's minor symmetries the two coincide, but a derivation that swaps them without saying so has not earned the step, and for anisotropic or learned constitutive models the distinction is real.

**Symmetry of a(u,v).** The final bilinear form must treat u and v identically, as in Section 5. An asymmetric result usually traces to pairing ∇v with ε(u) without invoking the symmetry of σ, or a constitutive substitution done on one argument only.

## 7. What agents typically miss here

In practice both the lead *and* the review handle the algebra well, and both hand-wave the same thing: the **regularity requirement on v**. The lead says "let v be a test function vanishing on Γ_D" without stating the space; the review, primed by its five-item checklist, verifies the items and never asks whether the integrals in (9) are even defined. Neither states that v ∈ H¹ is what makes ∫ ε(v):ℂ:ε(u) dΩ finite, or that "v = 0 on Γ_D" for an H¹ function is a statement about boundary traces, not pointwise values. This is not consequence-free pedantry — the same looseness reappears in Module 3, where an ansatz's smoothness quietly decides which formulation a PINN can legitimately discretize. Expect to be the one who catches it: the checklist covers what checklists cover; the human's job is to notice what the checklist itself omitted.

## The two prompts (verbatim)

**Lead prompt** (paste into a fresh agent session):

```
Derive the weak form of the equations of linear elastostatics on a 2-D domain Ω
with displacement BC u = ū on Γ_D and traction BC σ·n = t̄ on Γ_N. Assume small
strain and linear elasticity σ = ℂ:ε. Show every step:

1. State the strong form.
2. Multiply by an admissible test function v that vanishes on Γ_D.
3. Integrate over Ω.
4. Apply integration by parts (divergence theorem) on the σ term.
5. Substitute the Neumann BC into the boundary integral.
6. State the final weak form clearly, identifying the bilinear form a(u, v)
   and the linear form ℓ(v).
```

**Review prompt** (a second, fresh session — no shared context with the lead):

```
Read the following derivation and check the following items independently:

1. Dimensional consistency at every step.
2. That the test function v vanishes on Γ_D (and that this is used correctly when
   the boundary integral over Γ_D drops out).
3. That the surface integral over Γ_N matches the traction BC t̄.
4. Sign conventions throughout (especially after integration by parts).
5. Whether the final bilinear form a(u, v) is symmetric.

Reply with PASS / FAIL on each item, plus one sentence of justification per item.
If FAIL, explain what the correct version should be. Do not rewrite the entire
derivation — just point to the specific step.

Derivation to review:
<paste lead's output here>
```
