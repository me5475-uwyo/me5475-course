# Plasticity and the Return Mapping

*Course reading · ME 5475 · Module 4 · **[Optional]**, after L23 (Mon Oct 26), before L24 (Wed Oct 28). Companion
code: `module_4/examples/plot_yield_surfaces.py` draws every figure below and checks the formulas it draws or uses (the
curves lie on f = 0; the 1-D return mapping and the combined isotropic/kinematic J2 return satisfy their own equations). No computed result appears here: the
figures use illustrative parameters in normalized units.*

**The one sentence to keep:** a plasticity model is a yield function, a flow rule, a hardening law and the loading
conditions. A finite element code integrates it one strain increment at a time with a **return mapping**: predict
elastically, check the yield function, and if the stress is outside, return it to the surface.

L23 used both ideas: the classical four ingredients (slide 3) and the return-mapping slide after the divider. This
reading fills in the background: the one-dimensional model, the yield surfaces you will meet, isotropic and kinematic
hardening, and the J2 return mapping exactly as MOOSE runs it for Lab 4's data.

**Notation:**
- **Stress and invariants:** stress σ, tension positive. I₁ = tr σ. Deviator s = σ − (I₁/3) **I**. J₂ = ½ s:s. Von Mises stress q = √(3J₂). π-plane radius ρ = √(2J₂) = |s|.
- **Plasticity variables:** n = ∂f/∂σ, the normal. Δγ, the plastic multiplier per step (not Lab 4's shear strain γ_xy). ξ, the accumulated plastic multiplier. σ_y, the initial yield stress. h, the isotropic hardening modulus (MOOSE's `hardening_constant`). R, the hardening so far. X, the back stress.
- **Elasticity:** ℂ, the elastic stiffness; G, the shear modulus; E, Young's modulus.
- **Clashes with the MOOSE walkthrough:** it writes s − X as ξ, and uses γ for a recovery parameter. Here s − X is η, and that recovery parameter does not appear.

---

## 1 · One dimension: a spring and a slider

![A one-dimensional rheological model. On the left, a fixed wall, then a spring labeled E (elastic) carrying the elastic strain. It connects to two branches in parallel: on top, a friction slider that slips when the absolute value of sigma minus X reaches sigma_y plus H_iso times xi, carrying the plastic strain; below it, a spring H_kin that carries the back stress X. On the right, an arrow pulls with stress sigma.](figures/plasticity_1d_spring_slider_model.png)

Pull on the model with stress σ:
- **The spring E stretches** by the elastic strain ε^e = σ/E.
- **The slider does not move** until the stress it carries, σ − X, reaches its threshold. Once it slips, its displacement is the plastic strain ε^p.
- **The parallel spring H_kin** carries X = H_kin ε^p. That is the back stress.
- **Isotropic hardening** raises the slider's threshold as it slips.

**The four ingredients, in one dimension:**

```text
strain split:      ε = ε^e + ε^p,      σ = E (ε − ε^p)
1 yield function:  f(σ, X, ξ) = |σ − X| − (σ_y + H_iso ξ) ≤ 0
2 flow rule:       ε̇^p = γ̇ sign(σ − X)                       (= γ̇ ∂f/∂σ: associative)
3 hardening:       ξ̇ = γ̇   (isotropic: the elastic range grows)
                   Ẋ = H_kin ε̇^p   (kinematic: the elastic range moves)
4 loading:         γ̇ ≥ 0,   f ≤ 0,   γ̇ f = 0;   and γ̇ ḟ = 0 while it flows
```

The loading conditions (Kuhn–Tucker, card 4 of L23 slide 3) say that the slider slips only when the stress is on the
surface, and that the stress never leaves it. While the model flows, the stress–strain slope is the elastoplastic
modulus E (H_iso + H_kin)/(E + H_iso + H_kin).

**Isotropic or kinematic?** Load in tension to a peak stress σ_max, then reverse.
- **Isotropic hardening:** the elastic range grows symmetrically, so reverse yielding starts at σ = −σ_max.
- **Kinematic hardening:** the elastic range keeps its width, 2σ_y, but its center moves to X. Reverse yielding starts at σ = σ_max − 2σ_y, earlier than −σ_max. That is the **Bauschinger effect**, which metals show under cyclic loading.
- **Real metals** show a mix. MOOSE's `CombinedNonlinearHardeningPlasticity` (walkthrough §3) combines both.

![Two stress-strain panels, normalized by the yield stress and yield strain, for the same strain history: load to plus four, reverse to minus four, reload to plus four. Left, isotropic hardening (solid line): after the first peak, marked sigma_max, the unloading line runs elastically all the way down to minus sigma_max, where a red dot marks reverse yield, and every later yield point is higher. Right, kinematic hardening (dashed line): after the same kind of peak, reverse yield, marked by a red dot, starts much earlier, at sigma_max minus two sigma_y, and the loop is a parallelogram whose elastic range keeps the same width.](figures/plasticity_1d_isotropic_vs_kinematic_loops.png)

**The 1-D return mapping.** The FE code supplies a strain increment Δε. The material point stores ε^p_n, ξ_n and
X_n from the last step.

```text
1 predict:  σ^tr = E (ε_n + Δε − ε^p_n),     η^tr = σ^tr − X_n
2 check:    f^tr = |η^tr| − (σ_y + H_iso ξ_n).   If f^tr ≤ 0: elastic, σ_{n+1} = σ^tr, done.
3 return:   Δγ = f^tr / (E + H_iso + H_kin),     σ_{n+1} = σ^tr − E Δγ sign(η^tr)
4 update:   ε^p_{n+1} = ε^p_n + Δγ sign(η^tr),   X_{n+1} = X_n + H_kin Δγ sign(η^tr),   ξ_{n+1} = ξ_n + Δγ
```

**Why step 3 is one line.** After the return, η_{n+1} = η^tr − (E + H_kin) Δγ sign(η^tr). That has the same sign
as η^tr, so f_{n+1} = f^tr − (E + H_iso + H_kin) Δγ, and setting it to zero gives Δγ. Everything is linear, so the
answer is exact in one step. The companion script runs this loop to draw the stress–strain figure, and asserts f ≤ 0
at every step and f = 0 after every plastic one.

---

## 2 · Typical yield surfaces

A yield surface lives in a six-dimensional stress space. For an isotropic material it depends only on the principal
stresses, so it can be drawn in their three-dimensional space and viewed in two cuts:
- **The π-plane:** looking straight down the hydrostatic axis σ₁ = σ₂ = σ₃, as on L23's return-mapping slide. The projected principal axes appear 120° apart. A point's distance from the center is ρ = √(2J₂), and its direction around the center is the **Lode angle** θ.
- **The meridian plane:** the size of the deviator, q, against the pressure.

| criterion | yield function (tension positive) | suits | shape |
|---|---|---|---|
| **von Mises (J2)** | f = q − σ_y | metals: yield does not depend on pressure | a cylinder around the hydrostatic axis; on the π-plane, a circle of radius √(2/3) σ_y |
| **Tresca** | f = (σ_max − σ_min) − σ_y | metals: yield at a critical maximum shear stress | a hexagonal prism; on the π-plane, a regular hexagon |
| **Drucker–Prager** | f = √J₂ + A I₁ − B | soils, concrete, rock: a smooth stand-in for Mohr–Coulomb | a cone; on the π-plane, circles that shrink as I₁ grows (tension) |
| **Mohr–Coulomb** | f = (σ_max − σ_min) + (σ_max + σ_min) sin ω − 2c cos ω | soils and rock: friction angle ω and cohesion c | a cone with a hexagonal section; on the π-plane, an irregular hexagon |
| **modified Cam-Clay** | f = q² + M² p (p − p_c), with p = −I₁/3 | clays: critical-state soil mechanics | an ellipse in the (p, q) plane: a closed cap |

σ_max and σ_min are the largest and smallest principal stresses. The friction angle is usually written φ; here it is
ω, because φ is the level-set function in L23. A and B are the Drucker–Prager constants, as on L23 slide 4. **Hill's
criterion** generalizes von Mises to orthotropic materials, with a different yield stress in each material direction.
The course's MOOSE example is the walkthrough's §5.

![Two pi-plane panels, each with the projected principal axes sigma 1 (up), sigma 2 (lower left) and sigma 3 (lower right), 120 degrees apart, and a cross at the hydrostatic axis. Panel (a), metals with the same uniaxial yield stress: the von Mises circle (solid) and the Tresca regular hexagon (dashed) inside it, touching the circle at its six corners, which lie on and between the axes. Panel (b), frictional materials at I1 = 0: the Mohr-Coulomb irregular hexagon (solid), with three outer corners and three inner corners; a dashed Drucker-Prager circle through the outer corners, enclosing the hexagon; and a dash-dot Drucker-Prager circle through the inner corners, which the hexagon's edges cross slightly.](figures/yield_surfaces_pi_plane.png)

**Read the π-plane panels:**
- **Panel (a), von Mises and Tresca:** matched to the same uniaxial yield stress, the Tresca hexagon is **inscribed** in the von Mises circle. They agree under uniaxial stress and differ most in pure shear (below).
- **Panel (b), Mohr–Coulomb:** its outer corners lie on the **compression meridians** (σ_max = σ_mid > σ_min, i.e. axisymmetric compression). Its inner corners lie on the **extension meridians** (σ_max > σ_mid = σ_min).

**Matching Drucker–Prager to Mohr–Coulomb.** There are two classical choices. The **outer cone** passes through the
compression corners and circumscribes the hexagon:

```text
A = 2 sin ω / (√3 (3 − sin ω)),     B = 6 c cos ω / (√3 (3 − sin ω))
```

The **inner cone** passes through the extension corners: replace 3 − sin ω by 3 + sin ω. It meets the hexagon at its
inner corners but is not inscribed, because the hexagon's edges dip slightly inside it. The script checks both
statements.

![Plane stress, sigma 3 = 0, axes sigma 1 over sigma_y and sigma 2 over sigma_y. The von Mises ellipse (solid), tilted along the line sigma 1 = sigma 2, and the Tresca hexagon (dashed) inside it, with corners at (1, 0), (1, 1), (0, 1) and their negatives. Labeled points: uniaxial at (1, 0) and equibiaxial at (1, 1), where both criteria agree; and the pure-shear direction, where Tresca yields at tau = sigma_y / 2 and von Mises at tau = sigma_y / root 3, slightly farther out.](figures/yield_surfaces_plane_stress.png)

**Plane stress** (σ₃ = 0, as in a thin sheet) is the classic engineering picture. Von Mises becomes the ellipse
σ₁² − σ₁σ₂ + σ₂² = σ_y², where σ₁ and σ₂ are the two in-plane principal stresses. Tresca becomes a hexagon inside it.
**In pure shear** (σ₁ = τ, σ₂ = −τ), Tresca yields at τ = σ_y/2 and von Mises at τ = σ_y/√3.

![Two meridian-plane panels with pressure p = minus I1 over 3 on the horizontal axis, positive in compression, and q = root of 3 J2 on the vertical axis, in illustrative units. Left: the Mohr-Coulomb compression meridian (solid), which the outer Drucker-Prager cone coincides with; the extension meridian (dashed), lower; both rise linearly from a common apex on the tension side, at negative p; and the von Mises surface, a flat dotted line. Right: the modified Cam-Clay ellipse, passing through the origin and through p_c on the pressure axis, with the critical state line q = M p (dash-dot) passing through the ellipse's top; the inside is labeled elastic.](figures/yield_surfaces_meridian_plane.png)

**The meridian plane** is where pressure sensitivity shows. The pressure is p = −I₁/3, positive in compression, as in
soil mechanics and in the Cam-Clay formula.
- **Von Mises is flat:** pressure does not matter.
- **Drucker–Prager is one straight line.** Mohr–Coulomb gives two lines, its compression and extension meridians, that meet the p axis at the apex p = −c cot ω, on the tension side. A frictional material is stronger the more it is squeezed.
- **Modified Cam-Clay closes the cone with a cap:** an ellipse through the origin and through the preconsolidation pressure p_c, with its top on the critical state line q = Mp. p_c grows when the soil compacts plastically. That is Cam-Clay's hardening law, and it is why a clay can yield under hydrostatic compression, where an uncapped Drucker–Prager or Mohr–Coulomb cone would not.

---

## 3 · Hardening in stress space

![Two pi-plane panels with the projected principal axes. Left, isotropic hardening: a dashed circle (before) and a larger solid concentric circle (after), with red arrows pointing radially outward from the old circle to the new one; caption: same center, larger radius. Right, kinematic hardening: a dashed circle centered at the origin (before) and a solid circle of the same radius whose center has moved along a red arrow labeled X, the back stress; caption: same radius, center moved to X.](figures/hardening_isotropic_vs_kinematic_pi_plane.png)

In three dimensions the same two mechanisms move the von Mises surface:
- **Isotropic:** f = q − (σ_y + R), with R growing as plastic flow accumulates. Lab 4's MOOSE data use this, with linear hardening R = h ξ (`IsotropicPlasticityStressUpdate`, `hardening_constant`). Voce hardening, in walkthrough §4, saturates instead of growing without bound.
- **Kinematic:** f = √(3/2 η:η) − σ_y, with η = s − X. The deviatoric back stress X moves the circle's center. The linear (Prager) law is Ẋ = (2/3) h_kin ε̇^p. Walkthrough §3 adds the Armstrong–Frederick recovery term.

This is the moving surface of L23 slide 9 in classical form. Isotropic hardening expands the von Mises surface at the
same normal speed everywhere. Kinematic hardening translates it with normal speed **v**·n̂, where n̂ = n/|n| and **v** is
the back-stress velocity in stress-space coordinates.

---

## 4 · The return mapping for J2, as MOOSE runs it

The four steps of L23's return-mapping slide, at one material point for one strain increment Δε:

```text
1 predict:  σ^tr = σ_n + ℂ:Δε                                  (freeze plastic flow)
2 check:    f(σ^tr, state_n) ≤ 0 ?  then σ_{n+1} = σ^tr, done
3 return:   find Δγ > 0 with  σ_{n+1} = σ^tr − Δγ ℂ:n_{n+1}  and  f(σ_{n+1}, state_{n+1}) = 0
4 update:   ε^p_{n+1} = ε^p_n + Δγ n_{n+1};  harden
```

The normal is evaluated at the **end** of the step: backward Euler. After convergence, the returned stress satisfies
the yield condition to solver tolerance. Step size still controls accuracy, and nonlinear models may need substepping.

**Radial return (isotropic hardening).** For von Mises, n = (3/2) s/q is deviatoric. Isotropic elasticity gives
ℂ:n = 2G n, so the correction changes only the deviator and leaves the pressure untouched:

```text
s_{n+1} = s^tr − 2G Δγ (3/2) s_{n+1}/q_{n+1}   ⇒   (1 + 3G Δγ / q_{n+1}) s_{n+1} = s^tr
```

- **The direction is known before solving.** The bracket is a positive number, so s_{n+1} points the same way as s^tr, and the end normal equals the trial normal.
- **Only the size is unknown.** Taking magnitudes gives q_{n+1} = q^tr − 3G Δγ. With linear hardening, R = R_n + h Δγ, the condition f = 0 is one linear equation:

```text
Δγ = (q^tr − σ_y − R_n) / (3G + h)
```

- **Δγ is the effective plastic strain step,** √(2/3 Δε^p:Δε^p), the scalar MOOSE solves for. (L23's slide calls it Δp.)
- **3G plays the role E played in one dimension.**

**The same steps in MOOSE's source** (`modules/solid_mechanics/src/materials/`). These files are identical at commit
`437fbe5082`, the build that generated Lab 4's data (`measured_results.md`, Environments), and in a 2026-06-28
checkout:

| step | file : line | what it does |
|---|---|---|
| 1 predict | `ComputeMultipleInelasticStress.C` 79–80 | trial stress = ℂ : (old elastic strain + strain increment) |
| q^tr | `RadialReturnStressUpdate.C` 257–258 | `sqrt(3.0 / 2.0 * dev_trial_stress_squared)` |
| 2 check | `IsotropicPlasticityStressUpdate.C` 116 | yield condition q^tr − R_n − σ_y |
| 3 residual | `IsotropicPlasticityStressUpdate.C` 134 | (q^tr − R − σ_y)/(3G) − Δγ, the equation above divided by 3G |
| 3 solve | `RadialReturnStressUpdate.C` 271 | scalar Newton solve (one iteration is exact for linear hardening) |
| 4 direction | `RadialReturnStressUpdate.C` 276 | plastic strain step = (3/2) Δγ s^tr / q^tr, along the trial deviator |
| 4 stress | `RadialReturnStressUpdate.C` 291 | new stress = ℂ : (elastic strain) |
| 4 hardening | `IsotropicPlasticityStressUpdate.C` 180 | R = R_n + h Δγ |

**With linear kinematic hardening,** the return is radial from the back stress. Let η = s − X. The yield function is
f = √(3/2 η:η) − (σ_y + h ξ), and the back stress moves as ΔX = (2/3) h_kin Δγ n. Subtracting the back-stress update
from the stress update gives

```text
η_{n+1} = η^tr − (2G + (2/3) h_kin) Δγ (3/2) η_{n+1}/q_η,n+1   ⇒   η_{n+1} ∥ η^tr
```

So the deviatoric stress returns along the line from the current center X_n through the trial stress, and

```text
Δγ = (q_η^tr − σ_y − h ξ_n) / (3G + h + h_kin),      q_η^tr = √(3/2 η^tr:η^tr)
```

It is the 3-D twin of the 1-D formula in §1. The companion script checks the combined J₂ formula (setting h_kin = 0
recovers the isotropic formula) on random trial stresses:
- the returned stress lies on the updated surface;
- the end-of-step normal equals the trial normal;
- the backward-Euler equation holds with that normal;
- the pressure is unchanged.

**Beyond the J₂ shortcut.** The pressure-preserving radial return derived here uses a von Mises surface and isotropic
elasticity. Other models may need a coupled solve for stress and internal variables, but some retain special
reductions: a smooth Drucker–Prager return with constant friction and dilation parameters keeps its deviatoric
direction, while corners and apices require special handling. For a general smooth learned yield function, the end
normal can change during return, and Newton's Jacobian includes its stress derivative. L24 writes out that general
system.

---

## 5 · Loading conditions: two forms of one statement

Classical plasticity texts often state loading in stress form. On the surface (f = 0):

```text
n : dσ > 0   plastic loading,      n : dσ = 0   neutral,      n : dσ < 0   elastic unloading
```

**For the isotropically hardening J₂ model below (h > 0), this is the Kuhn–Tucker statement in disguise.** For f = q − σ_y − R with Ṙ = h γ̇, the
consistency condition ḟ = 0 during flow gives n : σ̇ − h γ̇ = 0, so

```text
γ̇ = (n : σ̇) / h
```

With h > 0, γ̇ ≥ 0 exactly when n : σ̇ ≥ 0.

**Where the stress form fails:**
- **Perfect plasticity (h = 0):** during flow n : σ̇ = 0 always, because the stress cannot leave a fixed surface. The test cannot tell flow from neutral loading.
- **Softening (h < 0):** during plastic flow n : σ̇ < 0, so the test reads it as unloading.

**The strain form works for both.** On the active plastic branch, with σ̇ = ℂ:(ε̇ − γ̇ n),

```text
γ̇ = (n : ℂ : ε̇) / (n : ℂ : n + h)
```

During elastic unloading, γ̇ = 0.

For J2, n : ℂ : n = 3G, the same denominator as the return mapping's 3G + h. The denominator stays positive for h = 0
and for mild softening, which is why computational plasticity is **strain-driven**: the FE code supplies Δε, not Δσ.
The return mapping's check is the discrete version of this test: starting on the surface,
f(σ^tr) ≈ n : ℂ : Δε to first order.

---

## 6 · Where a learned yield function plugs in

Replace f by a network φ, as in L23. The return mapping keeps its four steps and requests three kinds of quantities from the network:
- **its value,** in step 2's check and step 3's f = 0;
- **its normal** n = ∂φ/∂σ, by one backward pass, in step 3 (with associative flow, it is also the flow direction);
- **its curvature** ∂²φ/∂σ∂σ, inside Newton's Jacobian.

MOOSE's general plastic-model interface asks for the same three things:
`modules/solid_mechanics/include/userobjects/SolidMechanicsPlasticModel.h` declares
- `yieldFunction` (line 303);
- `dyieldFunction_dstress` (311);
- `flowPotential` (327), which with associative flow is that same normal;
- `dflowPotential_dstress` (335), which is then the Hessian.

The file is identical at the Lab 4 build. A learned φ would have to implement those functions. **The course does not
build this:** `module_4/examples/` stops at training φ, and Module 5 does not deploy a yield-surface network.

---

## 7 · Further reading

- **"Plasticity models in MOOSE"** (`module_4/readings/constitutive_models_moose_walkthrough.md`; optional, after L19): §3 kinematic hardening and §4 Voce hardening, each with a runnable MOOSE input validated on ARCC.
- **Belytschko, Liu, Moran & Elkhodary,** *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., 2014, Chapter 5, §5.1, 5.2 and 5.6 (assigned before L20).
- **Simo & Hughes,** *Computational Inelasticity*, Springer 1998: the classical reference on rate-independent plasticity, return mapping and algorithmic tangents.
- **de Souza Neto, Perić & Owen,** *Computational Methods for Plasticity: Theory and Applications*, Wiley 2008: the chapters "The mathematical theory of plasticity" and "Finite elements in small-strain plasticity problems".
- **Borja,** *Plasticity: Modeling and Computation*, Springer 2013: a modern, friendly alternative to Simo & Hughes.
