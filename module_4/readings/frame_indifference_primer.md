# Rotations, Tensors and Frame Indifference — A Primer for L20

*Course reading · ME 5475 · Module 4 · **[Primary]**, background for L20 (Mon Oct 19), slides 3–6; read it before L21
(Wed Oct 21). Runnable companion: `module_4/examples/frame_indifference_checks.py`, which recomputes every number below
(NumPy only; worked arithmetic, not a measurement). Its output is recorded in M4 measured results
(`module_4/readings/measured_results.md`), §8.*

**After reading this, you should be able to:**

- apply the M1 change-of-basis rule to strain without the shear-factor error, and to the 4th-order stiffness;
- tell a **change of basis** (rotate the axes) from a **superposed rigid rotation** (rotate the deformed body), and say which one frame indifference is about;
- state how F, C, E, b, J, σ, P, S and the small strain ε each behave under a superposed rotation;
- say what makes a material hyperelastic, derive the stress from a stored energy, and work the neo-Hookean example;
- read L20 slide 6, ψ(QF) = ψ(F), and explain why an energy model fed C (or its invariants) satisfies it for any weights.

---

## 1 · What you already have

- **Reading · Index notation and coordinate transformation** (Module 1). Part 2 derived the change-of-basis rule and Part 3 introduced the word "frame indifference". This primer does not repeat them; section 3 recaps them in four lines and adds only the two cases M1 did not work: strain and stiffness.
- **L6** introduced the deformation gradient F, its polar decomposition, the strain measures ε and E, and the three stress measures σ, P and S. Section 4 collects them in one table.
- **L6 and L19** named hyperelasticity (S = 2 ∂ψ/∂C, P = ∂ψ/∂F) without working it. Section 5 does.
- **L19** used "the invariants of C" without saying why C. Sections 6–7 answer that.

The one new idea is in section 2: there are two different operations that both use an orthogonal matrix, and they are easy to confuse.

## 2 · Two kinds of rotation

**Passive: rotate the axes.** Nothing physical changes. You describe the *same* vector or tensor in a new basis, so its components change. This is the M1 rule.

**Active: rotate the body.** After the body has deformed, rotate the whole deformed body rigidly (or, equivalently, let a second observer watch it while rotating). The new configuration is physically different, but the *material has done nothing new*: every fiber has the same stretch as before. Frame indifference is a statement about this operation.

| | passive (change of basis) | active (superposed rigid rotation) |
|---|---|---|
| what moves | the coordinate axes | the deformed body (or the observer) |
| what changes | the components of one tensor | the tensors, in the same basis |
| the formula | v' = Q v, T' = Q T Qᵀ | x* = Q x + c, so F* = Q F |
| question it answers | components in this frame? | does the response depend on the observer? |

In both, Q is orthogonal (QᵀQ = I), and for a rotation det Q = +1. In this primer Q always means such a rotation.

## 3 · Passive: strain and stiffness in a rotated coordinate system

**Recap of M1 Part 2** (read it there if any line is unfamiliar):

- Q_ij = e'_i · e_j, the direction cosines; in 2-D, Q = [[cos θ, sin θ], [−sin θ, cos θ]], and QᵀQ = I.
- One Q per index: v' = Q v; T' = Q T Qᵀ (T'_ij = Q_ip Q_jq T_pq); C'_ijkl = Q_ip Q_jq Q_kr Q_ls C_pqrs.
- The trace, determinant and principal values of a 2nd-order tensor are **invariants**, the same in every basis.
- The companion script's items 1–2 check the vector and stress rules in numbers (θ = 30°), if you want to see them.

**New here:** the 4th-order rule keeps Hooke's law consistent. If σ = C : ε in the old basis, then σ' = C' : ε' in the new one, because each side picks up the same Q's. Three worked cases at θ = 30° (companion script, items 3–5):

1. **Strain: transform the tensor shear, not γ.** ε = [[1, 0.5], [0.5, 0]] × 10⁻³ (tensor shear ε₁₂ = 0.5 × 10⁻³) becomes ε' = [[1.183, −0.183], [−0.183, −0.183]] × 10⁻³, so the engineering shear is γ' = 2ε'₁₂ = −0.366 × 10⁻³. Putting γ into the tensor rule gives a wrong answer. The factor-of-2 trap of `voigt_notation_reference.md` is why the Voigt stress and strain vectors need different transformation matrices.
2. **Isotropic stiffness** (plane strain, E = 1, ν = 0.3, C₁₁₁₁ = 1.346): unchanged by the rotation. For an isotropic material every rotation is a symmetry, so C' = C.
3. **Orthotropic stiffness:** C₁₁₁₁ = 10, C₂₂₂₂ = 1, C₁₁₂₂ = 0.3 and C₁₂₁₂ = 0.5, rotated by 30°, become C'₁₁₁₁ = 6.175, C'₂₂₂₂ = 1.675, C'₁₁₂₂ = 1.875 and C'₁₂₁₂ = 2.075, plus a coupling term C'₁₁₁₂ = −2.858 that was zero before. The material is the same; its stiffness components depend on the axes you use. The script also checks that σ' = C' : ε' agrees with Q (C : ε) Qᵀ.

## 4 · Deformation and stress measures, in one table

The motion maps a reference point X to x = φ(X). The deformation gradient is F = ∂x/∂X, J = det F > 0, and the polar decomposition is F = R U = V R (R a rotation, U and V symmetric positive definite stretches).

| measure | definition | lives in | what it is |
|---|---|---|---|
| F | ∂x/∂X | two-point | maps reference line elements to current ones |
| C | Fᵀ F | reference | right Cauchy–Green: squared stretches of reference fibers |
| E | ½ (C − I) | reference | Green–Lagrange strain |
| b | F Fᵀ | current | left Cauchy–Green |
| ε | ½ (∇u + ∇uᵀ) | (linearized) | small strain, valid only for small displacement gradients |
| σ | force / current area | current | Cauchy ("true") stress, symmetric |
| P | J σ F⁻ᵀ | two-point | first Piola–Kirchhoff: force / reference area |
| S | F⁻¹ P = J F⁻¹ σ F⁻ᵀ | reference | second Piola–Kirchhoff, symmetric |

The work-conjugate pairs are P with Ḟ and S with Ė (the stress power per reference volume is P : Ḟ = S : Ė). For a hyperelastic material with stored energy ψ, P = ∂ψ/∂F, and, writing ψ(F) = ψ̃(C), S = 2 ∂ψ̃/∂C (section 5). In small strain all three stresses coincide to first order, which is why Modules 0–3 used σ only.

## 5 · Hyperelasticity: stress from a stored energy

L6 and L19 named hyperelasticity. This section works it, because the rest of this primer, and L20–L21, are about the constraints a stored energy must satisfy.

**What it is.** A material is **hyperelastic** if there is a stored-energy function ψ(F), the energy per unit reference volume, from which the stress follows by differentiation. Loading and unloading then follow the same path, and the work done on the material is stored, not dissipated: around any closed deformation cycle the net work is zero. Rubber, approximately many soft tissues, and every linear-elastic model in Labs 0–3 are hyperelastic. **Lab 4's J₂ plasticity is not:** its response depends on the loading history through internal variables (L22–L24).

**Why the stress is a derivative.** The stress power per unit reference volume is P : Ḟ (section 4). If all of it is stored, then ψ̇ = P : Ḟ for every deformation rate Ḟ. The chain rule gives ψ̇ = (∂ψ/∂F) : Ḟ, and since Ḟ is arbitrary,

P = ∂ψ/∂F.

If ψ depends on F only through C (section 7 shows that frame indifference requires exactly this), write ψ(F) = ψ̃(C). Then dC = dFᵀ F + Fᵀ dF, and because ∂ψ̃/∂C is symmetric the chain rule gives P = 2 F ∂ψ̃/∂C. So S = F⁻¹ P = 2 ∂ψ̃/∂C, and σ = J⁻¹ P Fᵀ.

**Small strain is the special case you already know.** With ψ = ½ ε : C : ε, the derivative is σ = ∂ψ/∂ε = C : ε, Hooke's law. Every linear-elastic computation in Labs 0–3 was hyperelastic in this sense.

**Isotropic energies** depend only on the invariants of C (section 8): I₁ = tr C, I₂ = ½[(tr C)² − tr C²] and J = det F (the third invariant is I₃ = det C = J²).

**The running example: compressible neo-Hookean.**

ψ = ½μ (I₁ − 3) − μ ln J + ½λ (ln J)².

Using ∂I₁/∂F = 2F and ∂J/∂F = J F⁻ᵀ:

P = μ (F − F⁻ᵀ) + λ (ln J) F⁻ᵀ, and σ = J⁻¹ [μ (b − I) + λ (ln J) I].

- **At F = I,** ψ = 0 and P = 0: the reference state is stress-free.
- **In small strain,** F = I + H with H small, F⁻ᵀ ≈ I − Hᵀ and ln J ≈ tr H, so P ≈ μ (H + Hᵀ) + λ (tr H) I = 2μ ε + λ (tr ε) I. That is Hooke's law with Lamé constants μ and λ: the two constants mean what they meant in Labs 0–3.
- **For μ > 0 and λ ≥ 0, as in our course examples,** ψ → +∞ as J → 0⁺: compressing to zero volume costs infinite stored energy. Hyperelasticity alone does not guarantee this barrier.

Mooney–Rivlin adds an I₂ term; Arruda–Boyce and Ogden are other common isotropic forms (L19 lists them). They follow the same pattern.

**Checked in numbers** (companion script, items 10–11; 2-D plane strain, E = 1, ν = 0.3, where the 2-D energy uses tr C − 2). At F = [[1.1, 0.2], [0, 0.95]], the formula for P agrees with a central finite difference of ψ to 3 × 10⁻¹¹ and equals J σ F⁻ᵀ. For F = I + δH, the relative gap between P and C : ε is 1.0 × 10⁻² at δ = 10⁻² and 1.0 × 10⁻⁴ at δ = 10⁻⁴: it shrinks in proportion to the strain, as first-order agreement should.

**What this means for a network.** A neural hyperelastic model learns one scalar ψ_NN and gets the stress by automatic differentiation, which is P = ∂ψ/∂F. That is L21's demo, `icnn_hyperelastic.py`, trained on the neo-Hookean energy above. Two things come for free: the stress is path-independent and dissipation-free by construction, and every requirement on the stress becomes a requirement on one scalar function. The rest of this primer asks which requirements: frame indifference (section 7) and material symmetry (section 8). L21 adds polyconvexity.

## 6 · Active: what each measure does under a superposed rotation

Rotate the deformed body: x* = Q x + c, with Q a rotation. Then F* = Q F, and each measure follows from its definition:

| measure | under F → Q F | why |
|---|---|---|
| F | Q F | definition |
| J | J | det Q = 1 |
| C | **C** | (Q F)ᵀ (Q F) = Fᵀ Qᵀ Q F = Fᵀ F |
| E, U | **unchanged** | functions of C |
| R | Q R | the rotation part absorbs Q |
| b, V | Q b Qᵀ, Q V Qᵀ | (Q F)(Q F)ᵀ = Q F Fᵀ Qᵀ |
| σ | Q σ Qᵀ | a physically objective stress rotates with the body |
| P | Q P | J (Q σ Qᵀ)(Q F)⁻ᵀ = J Q σ Qᵀ Q F⁻ᵀ = Q P |
| S | **S** | (Q F)⁻¹ (Q P) = F⁻¹ Qᵀ Q P = S |
| ε (small strain) | **neither** (see below) | ε* = sym(Q F − I): rotation read as strain |

**"Neither"** means the small strain follows neither clean rule: it is not unchanged (like C), and it does not become Q ε Qᵀ (like σ). It mistakes rotation for strain: a pure 45° rotation, F = Q with no deformation at all, gives ε₁₁ = ε₂₂ = −0.293.

The pattern is that reference-configuration measures (C, E, U, S) do not see the rotation. Current-configuration ones (b, V, σ) rotate with it, and two-point ones (F, R, P) pick up one Q on the left. "C = FᵀF is where the rotation dies" (L20 slide 11) is the third row.

**Worked examples** (companion script, items 6, 7 and 9):

- **C is unchanged.** Take F = [[1.1, 0.2], [0, 0.95]] and rotate it by 45°. C = [[1.21, 0.22], [0.22, 0.9425]] before and after.
- **A neo-Hookean model passes every check.** With ψ = ½μ (tr C − 2) − μ ln J + ½λ (ln J)², E = 1 and ν = 0.3, the energy is 0.012956 for both F and Q F. The script confirms σ* = Q σ Qᵀ, P* = Q P and S* = S.
- **Small strain is not rotation-invariant.** A pure rigid rotation should produce no strain, and E is exactly zero for any rotation. The small strain ε is not: a 45° rotation gives −0.293, and even 1° gives ε₁₁ = −1.5 × 10⁻⁴. Small-strain models, such as Lab 4's J₂ plasticity, are valid only for small rotations as well as small strains.

## 7 · Frame indifference: the requirement on a constitutive model

A constitutive law is **frame-indifferent** (objective) if the response to the rotated deformation is the rotated response:

σ̂(Q F) = Q σ̂(F) Qᵀ for every rotation Q and every F.

For a hyperelastic material this is equivalent to

ψ(Q F) = ψ(F) for every rotation Q,

and it holds exactly when ψ depends on F only through C: ψ(F) = ψ̃(C). The "if" direction is section 6's third row. The "only if" uses the polar decomposition: choose Q = Rᵀ, so ψ(F) = ψ(Rᵀ F) = ψ(U), and U is a function of C.

This is L20 slide 6, line by line:

- **"ψ(QF) = ψ(F)"** is the requirement above;
- **"P(QF) = Q P(F); Cauchy goes to Q σ Qᵀ"** is section 6's table;
- **"ψ sees F only through C = FᵀF"** is the equivalence just shown.

**Why a network breaks it, in numbers** (companion script, item 8). The energy ψ = ½μ |F − I|², a plausible-looking formula, gives 0.010 for F but 0.187 for the same deformation rotated by 45°. It is not frame-indifferent: it treats rotation as strain. A network fed the raw components of F has no reason to do better, unless it is trained or built not to.

**Why feeding C works for any weights.** If an energy network's input is C (or its invariants), a superposed rotation does not change the input, so no choice of weights can change the energy. The stresses then come from the energy (P = ∂ψ/∂F, S = 2 ∂ψ̃/∂C) and transform as section 6 says. This argument is about a **scalar energy**: an invariant output is not automatically an objective stress. A network that mapped the invariants of C straight to Cauchy-stress components would return the same σ for F and Q F, when σ should become Q σ Qᵀ. That is L20's "Way 3", and route 1 of the three that M1 Part 3 listed; this section supplies the proof M1 did not give. L20 slide 11 quotes `icnn_hyperelastic.py`, whose input is the invariants of C.

## 8 · Material symmetry is a different statement

Frame indifference multiplies F on the **left** (the observer, or the rotated body). Material symmetry multiplies on the **right** (the material's own reference directions):

ψ(F Q_sym) = ψ(F) for every Q_sym in the material's symmetry group G_sym.

**What Q_sym is.** A rotation of the material's own structure (its fibers, crystal lattice or texture), made in the reference configuration before any deformation, **that leaves the material looking exactly the same**. All such rotations form the symmetry group G_sym ⊂ SO(3) (some texts write H for its members):

- **isotropic:** every rotation;
- **fibers along e₁** (a unidirectional composite, transversely isotropic): every rotation about e₁, and the 180° rotations about any axis perpendicular to e₁, which reverse the fiber direction (a straight fiber looks the same either way);
- **cubic crystal:** the 24 rotations that map a cube onto itself;
- **section 3's orthotropic stiffness** (2-D): a 180° rotation is a symmetry (in 2-D it is Q = −I, and its four factors cancel), but 30° is not, since case 3 showed that it changes the components; nor is 90°, which would swap C₁₁₁₁ = 10 and C₂₂₂₂ = 1.

**Why the right side: the two ends of F.** F = ∂x/∂X maps a line element in the reference body to the same line element after deformation, dx = F dX. Its **right** side takes in reference vectors (dX, directions drawn in the undeformed material); its **left** side puts out current vectors (dx, directions in the deformed body, as an observer sees them). In index notation `F_iJ` has a spatial index i and a material index J, and a rotation can act on either one:

| | Q on the left | Q on the right |
|---|---|---|
| in indices | `(QF)_iJ = Q_ik F_kJ` | `(FQ)_iJ = F_iK Q_KJ` |
| rotates | the deformed body, or the observer | the reference directions |
| happens | after the deformation | before it |

A material's own directions (a fiber direction a, a lattice vector, a rolling direction) are drawn in the undeformed material, so they are reference vectors and attach to the right side of F. A statement about them, which is what a symmetry is, therefore acts on the right.

What the material feels is the stretch of its directions: the squared stretch of a **unit** material direction a is a · C a (for a non-unit a, divide by a · a). On the left, C does not change, (Q F)ᵀ(Q F) = C, so every direction keeps its stretch and no material of any kind can detect the rotation. On the right, C becomes Qᵀ C Q, so a · C a becomes (Q a) · C (Q a): the material now feels the deformation along the rotated direction Q a. An isotropic material treats all directions alike and cannot tell. A fiber material can, unless Q a = ±a.

**A two-line example** (2-D; companion script, item 12). Take F = diag(1.1, 1), so C = diag(1.21, 1), with fibers along the unit direction a = e₁.

| | C | fiber stretch² a · C a | I₁ = tr C |
|---|---|---|---|
| original F | diag(1.21, 1) | 1.21 | 2.21 |
| Q F, any rotation Q | diag(1.21, 1) | 1.21 | 2.21 |
| F Q, Q = 90° | diag(1, 1.21) | 1.00 | 2.21 |
| F Q, Q = 180° (Q = −I) | diag(1.21, 1) | 1.21 | 2.21 |

On the left nothing changes, for any material. On the right, a 90° rotation moves the stretch from the fiber to the transverse direction: a fiber composite would store a different energy, so 90° is not its symmetry, while an isotropic material, which sees only I₁, I₂ and J, cannot tell. A 180° rotation only reverses the fiber, and is a symmetry of both. (In 2-D, −I is the 180° rotation; in 3-D it is not a rotation, and the fiber-reversing half-turns are rotations about axes perpendicular to a.)

Two descriptions of "the same rotation on the right" appear below, and they differ by a transpose. **Relabeling** the reference by X' = Q X turns F into F Qᵀ (step 3). **Composing** F with a reference-side rotation gives F Q. Both act on the right; step 6 shows why the symmetry condition can be written either way.

**Where the equation comes from, step by step.**

*Step 1: an energy function belongs to a reference configuration.* ψ(F) is measured from a particular undeformed placement of the body, its reference configuration κ: F = ∂x/∂X, where X labels each particle by its position in κ. Write the energy function as ψ_κ when the reference matters.

*Step 2: rotate the reference.* Take the same undeformed body and rotate the material in it rigidly, before any loading, so that the particle that sat at X now sits at X' = Q X. Call that placement κ'. Its particles are the same particles, unstressed, merely turned.

*Step 3: the same deformation, described from κ'.* Now deform the body so that every particle ends up exactly where it did before, at x = φ(X). In terms of the new labels X = Qᵀ X', so x = φ(Qᵀ X'), and by the chain rule

∂x/∂X' = (∂x/∂X)(∂X/∂X') = F Qᵀ.

So the same physical deformation has gradient F from κ and F Qᵀ from κ'. The rotation multiplies F **from the right**, because it acts on the reference side, X, of F = ∂x/∂X.

*Step 4: the stored energy is a physical quantity.* The energy stored in the deformed body does not depend on which description we use, so ψ_κ'(F Qᵀ) = ψ_κ(F) for every F. This holds for **any** rotation Q. It only says how an energy function changes when the reference is relabeled.

*Step 5: symmetry means the two energy functions are the same function.* If the rotated material in κ' cannot be told apart from the original in κ (same fibers, same lattice, in the same directions), then any experiment from κ' gives the same results as from κ, so ψ_κ' = ψ_κ. Rotations with this property are the material's symmetries. Substituting into step 4 gives

ψ(F Qᵀ) = ψ(F) for every Q in G_sym.

*Step 6: the group property.* If Q₁ and Q₂ are symmetries, so is Q₁Q₂ (do one, then the other), and so is Q₁ᵀ = Q₁⁻¹ (undo it). So the symmetries form a group. Because Qᵀ is in the group whenever Q is, step 5 can be written with Q in place of Qᵀ, which is the form on L20 slide 7: ψ(F Q_sym) = ψ(F).

**In terms of C.** Since C(F Q) = Qᵀ C Q, the condition reads ψ̃(Qᵀ C Q) = ψ̃(C) for every Q in G_sym. The energy cannot tell C apart from C written in rotated material axes.

**In small strain, and the link to section 3.** For ψ = ½ ε : C : ε the condition becomes ψ(Qᵀ ε Q) = ψ(ε) for every ε, which holds exactly when the rotated stiffness equals the original, C' = C. That is what section 3 computed. Case 2 (isotropic, C' = C for any angle) says every rotation is a symmetry. Case 3 (orthotropic, C' ≠ C at 30°) says 30° is not.

**What it means for a model.** An isotropic energy may use only I₁, I₂ and J. A transversely isotropic one with fiber direction a also needs invariants that contain a, for example I₄ = a · C a (the squared fiber stretch) and I₅ = a · C² a. These are unchanged by rotations about a and by a → −a (because Q a = ±a), and in general they change under other rotations. Building such inputs is Way 3 for an anisotropic material (Heider, Wang & Sun 2020).

Compare frame indifference: there Q acts **after** the deformation, holds for **every** rotation, and holds for **every** material. Material symmetry acts **before** it, and holds only for the rotations in this material's group.

- **Isotropic:** every rotation is in the group, so ψ depends only on the invariants of C: I₁ = tr C, I₂ = ½[(tr C)² − tr C²] and J. Case 2 in section 3 is the small-strain version of this: C' = C.
- **Orthotropic or anisotropic:** only some rotations are in the group, so ψ needs extra structural directions. Case 3 is what happens otherwise: rotating the material directions relative to the load changes the stiffness components.

For an isotropic material the two statements lead to the same remedy, the invariants of C. That is why they are easy to merge, and why anisotropy (L20 slide 7) separates them.

## 9 · Where this goes

- **L20:** the three ways to enforce frame indifference (loss, augmentation, architecture).
- **L21:** polyconvexity and ICNNs. Invariant inputs make the energy frame-indifferent (section 7); they do not by themselves make it polyconvex. A suitable construction can guarantee polyconvexity: polyconvex inputs and a network that is convex **and nondecreasing** in them. L21's demo is convex in its invariant inputs but does not enforce the monotonicity, so its polyconvexity is checked, not assumed.
- **Lab 4** is small-strain J₂ plasticity. The rotation caveat in section 6 is one reason the course stays at small strain there.

## Self-check

1. Rotate the orthotropic C of section 3 by 90° instead of 30°. Which components change? Why does the 30° rotation produce a C'₁₁₁₂ term and the 90° rotation not?
2. Why is it wrong to transform the Voigt strain vector [ε₁₁, ε₂₂, γ₁₂] with the same matrix as [σ₁₁, σ₂₂, σ₁₂]?
3. Under x* = Q x + c, which of F, C, b, σ, P and S change, and how? Derive P* from P = J σ F⁻ᵀ.
4. Show that ψ = ½μ (tr C − 3) is frame-indifferent and that ψ = ½μ |F − I|² is not, for a rotation Q ≠ I and F = I.
5. Explain the difference between ψ(Q F) = ψ(F) and ψ(F H) = ψ(F) in one sentence each.
6. A 2° rigid rotation is applied to a small-strain model. Estimate the spurious strain ε₁₁ = cos θ − 1.
7. Derive P = μ (F − F⁻ᵀ) + λ (ln J) F⁻ᵀ from the neo-Hookean ψ, and show that it reduces to σ = 2μ ε + λ (tr ε) I in small strain.
8. Why is Lab 4's J₂ plasticity model not hyperelastic? What would a closed loading cycle show?

## Sources

- Belytschko, Liu, Moran & Elkhodary, *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., 2014. Chapter 5, already assigned before L20; it treats the stress measures and objectivity in the course's notation.
- Holzapfel, *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*, Wiley, 2000. The kinematics, stress and objectivity chapters.
- Gurtin, Fried & Anand, *The Mechanics and Thermodynamics of Continua*, Cambridge, 2010. The chapters on change of frame and frame-indifference.
