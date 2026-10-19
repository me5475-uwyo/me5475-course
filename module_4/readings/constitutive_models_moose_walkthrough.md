# Beyond J2 Linear Hardening: Four Constitutive Models in MOOSE

*Module 4. Read after Lecture 19 (Fri Oct 16), especially when considering midterm Flavor B. Companions are in `module_4/examples/`: four `constitutive_*.i` inputs and `run_constitutive_models.sbatch`. For input-file structure, see `module_0/readings/moose_input_file_anatomy.md`.*

**Validation [fallback], October 16.** All four companions were tested on the pinned course build; the Drucker-Prager companion includes the validated object-name correction. Run evidence and limits are recorded in [M4 measured results](https://github.com/me5475-uwyo/me5475-course/blob/main/module_4/readings/measured_results.md), Section 7. Parameter values below are teaching choices, not fitted material properties. This optional reading introduces examples, not a completed midterm solution or a new assignment requirement.

Lab 4 uses J2 plasticity with linear isotropic hardening. These alternatives change either the yield surface or its evolution:

| Model | What changes | Input suffix |
|---|---|---|
| Drucker-Prager | Yield depends on pressure | `drucker_prager` |
| Kinematic hardening | The yield surface translates | `kinematic` |
| Voce hardening | Expansion toward saturation | `voce` |
| Hill plasticity | Yield depends on material direction | `hill` |

The full input filename is `constitutive_<suffix>.i`.

## 1 · The common constitutive step

All examples use small strain, Cauchy stress, isotropic linear elasticity and a plastic-strain tensor. Tension is positive. A colon denotes the tensor inner product: A:B is the sum of A_ij B_ij over all components.

    ε = ε_el + εᵖ,    σ = C : (ε − εᵖ).
    (q_n, ε_n, ε_{n+1}) → (σ_{n+1}, q_{n+1}).

Here ε_el is elastic strain and q is the collection of internal variables. For J2 isotropic hardening it contains εᵖ and the scalar accumulated equivalent plastic strain p, also written ε̄ᵖ:

    dp = √[(2/3) dεᵖ : dεᵖ],    p starts at zero.

Accumulate these nonnegative increments along the plastic path. Under reversal, components of εᵖ can decrease while p increases. Thus p is not generally the magnitude of the current εᵖ tensor. The J2 definition of p is not automatically the hardening-variable normalization of every other model.

Define the deviator s = σ − tr(σ)I/3 and the J2 equivalent stress σ_eq = √[(3/2)s:s]. A yield function f is nonpositive in admissible states. Plastic loading uses a nonnegative multiplier dλ and a flow potential g:

    f ≤ 0,    dλ ≥ 0,    f dλ = 0,    dεᵖ = dλ ∂g/∂σ.

Associated flow means g = f; other choices are nonassociated. An elastic predictor followed by a plastic correction updates the state. The MOOSE input supplies the material objects; it does not replace the return-mapping algorithm with an ML model.

## 2 · Drucker-Prager: pressure-sensitive yield

Let I1 = tr(σ), J2 = s:s/2, and define pressure P = −I1/3, positive in compression. An ideal Drucker-Prager cone can be written

    f = √J2 + A I1 − K = √J2 − 3 A P − K.
    g = √J2 + B I1.

For positive A, compression increases the admissible shear stress. A and B control friction and dilation; K sets cohesion. If B differs from A, the flow is nonassociated and plastic strain need not be volume preserving. This is a different mechanism from J2 metal plasticity.

The companion uses `CappedDruckerPragerStressUpdate`, `SolidMechanicsPlasticDruckerPrager` and `SolidMechanicsHardeningConstant`. The pinned build still accepts the old `TensorMechanics...` names only as expired aliases and emits warnings, so the companion uses the `SolidMechanics...` names. The selected `outer_tip` conversion from Mohr-Coulomb parameters is

    A = 2 sin(φ) / [√3 (3 − sin(φ))].
    K = 6 c cos(φ) / [√3 (3 − sin(φ))].

The flow coefficient uses the corresponding conversion with dilation angle ψ. The chosen values are c = 1, φ = 20 degrees and ψ = 10 degrees. The input explicitly converts angles to radians. **These are illustrative choices, not a calibration.**

The executable model has finite tensile/compressive caps and numerical smoothing. The example puts both strengths at 1e6 stress units and smoothing parameters at 0.001 to target the cone away from its tip and caps. That intent must be checked in the run; it is not an assertion that the numerical surface is exactly the unsmoothed cone. Do not confuse the yield function with the flow potential. See the [MOOSE parameter object](https://mooseframework.inl.gov/source/userobjects/SolidMechanicsPlasticDruckerPrager.html) and [capped stress-update interface](https://mooseframework.inl.gov/source/materials/CappedDruckerPragerStressUpdate.html).

**What to examine:** compare loading in tension and compression and inspect plastic volumetric strain tr(εᵖ). A pressure-sensitive model needs data spanning different pressures. One proportional strain history is only a first check.

## 3 · Kinematic hardening: a moving yield surface

Introduce the deviatoric backstress tensor X. With a fixed initial yield stress σ0,

    ξ = s − X,    f = √[(3/2) ξ:ξ] − σ0.
    dX = (2/3) C_k dεᵖ − γ X dp.

This Armstrong-Frederick evolution translates the surface and includes a recovery term. Setting γ = 0 gives linear kinematic hardening. Translation changes reverse yielding after prior plastic loading: the Bauschinger effect. It does not mean that the elastic modulus changed.

The input uses `CombinedNonlinearHardeningPlasticity`, with `kinematic_hardening_modulus = 10`, `gamma = 20`, and `yield_stress = 1`. Setting its isotropic slope and its Voce parameters `q` and `b` to zero isolates kinematic hardening. Here C_k is a hardening modulus, not the elasticity tensor C. These parameters and their reductions are described in the [MOOSE combined-hardening documentation](https://mooseframework.inl.gov/source/materials/CombinedNonlinearHardeningPlasticity.html).

**What to examine:** follow unloading and reverse loading, not just the first monotonic branch. The state now needs backstress as well as plastic strain. A network trained on current strain alone cannot generally recover that state.

## 4 · Voce hardening: expansion toward saturation

For isotropic hardening the yield surface stays centered but its radius changes:

    f = σ_eq − σ_Y(p),    σ_Y(p) = σ0 + Q [1 − exp(−b p)].

σ0 is the initial yield stress; Q is the additional saturation strength; b controls how quickly hardening approaches saturation. For Q and b positive, the radius grows toward σ0 + Q. This is a statement about the constitutive law, not a measured stress maximum under the example's multiaxial loading.

The companion keeps `IsotropicPlasticityStressUpdate` and supplies a `ParsedFunction` through `hardening_function`:

```text
[voce_yield]
  type = ParsedFunction
  expression = '1.0 + 0.5 * (1.0 - exp(-20.0*t))'
[]
```

**In this function, `t` receives accumulated equivalent plastic strain, not simulation time. Return the total yield stress, including σ0.** MOOSE evaluates the function at the accumulated plastic strain and subtracts the separately supplied initial yield stress internally. Do not pass only the increment Q[1−exp(−bp)], and do not also set `hardening_constant`. See the [stress-update interface](https://mooseframework.inl.gov/source/materials/IsotropicPlasticityStressUpdate.html) and [its hardening-function implementation](https://mooseframework.inl.gov/docs/doxygen/modules/IsotropicPlasticityStressUpdate_8C_source.html).

**What to examine:** the decrease in hardening slope as plastic strain accumulates. The yield surface expands without a backstress translation. A single loading amplitude may explore too little plastic strain to identify the saturation parameters.

## 5 · Hill plasticity: directional yield

In material axes x, y, z, define the quadratic form

    Q_H(σ) = F(σ_yy − σ_zz)² + G(σ_zz − σ_xx)²
             + H(σ_xx − σ_yy)²
             + 2L σ_yz² + 2M σ_xz² + 2N σ_xy².

F, G, H, L, M and N control directional yield. Hydrostatic stress cancels from the differences, so this model is pressure insensitive. Unequal coefficients can make yielding in x differ from yielding in y even with isotropic elasticity.

The companion pairs `HillConstants` with `HillElastoPlasticityStressUpdate`. The material axes align with the global axes. Its parameters isolate yield anisotropy:

```text
hill_constants = '0.6 0.4 0.7 1.5 1.5 1.5'
yield_stress = 1
hardening_constant = 0
```

**Normalization matters.** The cited MOOSE implementation constructs the Hill matrix M_H such that σ_Vᵀ M_H σ_V = Q_H, then uses ω = √(Q_H/2) in its yield test. With zero hardening, yielding occurs at ω = `yield_stress`. Therefore this input parameter is not automatically the uniaxial yield stress in x. For an ideal uniaxial stress state, the threshold is σ_x = √[2/(G+H)] × `yield_stress`. The companion's boundary conditions do not create that uniaxial stress test. See the [Hill model documentation](https://mooseframework.inl.gov/source/materials/HillElastoPlasticityStressUpdate.html), [stress-update source](https://mooseframework.inl.gov/docs/doxygen/modules/HillElastoPlasticityStressUpdate_8C_source.html) and [Hill-matrix construction](https://mooseframework.inl.gov/docs/doxygen/modules/HillConstants_8C_source.html).

**What to examine:** compare separate runs loading along x and y with the material axes fixed. Do not mistake rotating the observer's coordinates for rotating the loading relative to the material. A midterm dataset must cover relevant directions; one x-directed history cannot establish anisotropic behavior.

## 6 · Run the companions as ARCC jobs

Each input is complete: mesh, variables, strain calculation, boundary conditions, material, solver and CSV outputs. The common test is one plane-strain square element with prescribed displacement on every edge. The baseline strain vector is (ε_xx, ε_yy, γ_xy) = (0.02, 0, 0) f(t), where γ_xy = 2ε_xy. All four use E = 100 and ν = 0.3 in consistent stress units. These are strain-controlled tests, **not uniaxial-stress tests**: transverse stresses generally develop, including σ_zz.

f(t) contains three fully reversed cycles, just as in Lab 4. The new examples choose dt = 0.01 over 0 ≤ t ≤ 3: 300 prescribed increments plus the initial state. This is an input choice, not a convergence result; Lab 4's original dt = 0.05 gives 60 increments and 61 recorded states instead.

From your course repository on ARCC:

```bash
cd module_4/examples
sbatch run_constitutive_models.sbatch
squeue -u "$USER"
```

The job uses the Phase 1 course executable:

```text
/project/me5475/software/moose_torch/me5475/me5475-opt
```

Its log is `constitutive-<jobid>.out`; outputs are in a new `constitutive_runs_<jobid>/` directory. It first checks each input, then runs it. A failing command stops the script. The October 16 ARCC validation, job IDs and measured limits are in Section 7 of the M4 measured-results ledger. The supplied Drucker-Prager input already uses the validated object names.

To run only one model or repeat with smaller increments:

```bash
sbatch run_constitutive_models.sbatch kinematic
sbatch run_constitutive_models.sbatch voce 0.005
sbatch run_constitutive_models.sbatch hill 0.005
```

Open the CSV and plot `stress_xx_avg` against `strain_xx_avg`, retaining row order to follow the cycles. The file also contains yy, zz and xy strain/stress/plastic-strain components. CSV shear strains are tensor shear; multiply by two for engineering shear. To inspect pressure or plastic volume change, use all three normal components, including zz. Plane strain sets total ε_zz to zero, not σ_zz or εᵖ_zz.

Before using an input to generate a dataset, check input registration, finite values, completion to t = 3, the imposed strain history, actual plastic straining, and sensitivity to halving dt. A successful solve alone does not validate the constitutive response. Run the diagnostic comparisons suggested above; preserve inputs, logs and CSVs. The October 16 measurements are in the ledger; one dt-halving comparison measures sensitivity but does not prove convergence.

**Measured timestep sensitivity.** Comparing dt = 0.01 with dt = 0.005 at common times, Hill had the largest differences: 0.0805% for stress and 1.31% for plastic strain. For each component, the comparison is max(|coarse − fine|) divided by max(max(|fine|), 1e-12), with the maxima taken over common times; these percentages are the largest component-wise values. They describe sensitivity in these runs, not error relative to an exact solution. See M4 measured results, Section 7.2. Recheck sensitivity for any new loading path or material parameters.

## 7 · From an example to a Flavor B project

Choose a model for its mechanism, then design loading paths that expose that mechanism. These files provide one starting history each, not the required dataset. Change amplitudes and directions; for paths that turn, use distinct time functions for strain components. Split complete trajectories, not neighboring time steps, between training and testing. Keep the reference simulator, a stateless baseline and the physics constraint required by the midterm spec.

The examples and this reading do not release held Lab 4 Task 2–3 results. Numerical claims about these new runs are recorded in the measured-results ledger. The MOOSE documentation linked above can evolve; the course build is pinned to MOOSE `437fbe5082`. Validation showed that this build registers the current `SolidMechanics...` names and retains the old `TensorMechanics...` names only as expired aliases, so the corrected companion uses `SolidMechanics...`.
