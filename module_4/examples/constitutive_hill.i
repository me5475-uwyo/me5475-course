# ME-5475 / after L19: hill single-element example.
# [fallback] Codex temporary Co-Worker, 2026-10-16. ARCC validation recorded in measured_results.md Section 7.
# Read ../readings/constitutive_models_moose_walkthrough.md.
# Derived from the course j2_cyclic_loading.i; standalone plane-strain input.
# Chosen teaching parameters in consistent stress units, not a material fit.
# Use run_constitutive_models.sbatch; do not run on the ARCC login node.
# Target: course MOOSE 437fbe5082, Phase 1 me5475-opt. Validated by ARCC jobs 24624305–24624306 and 24624330; see measured_results.md §7.
# Prescribed strain: (eps_xx,eps_yy,gamma_xy)=(0.02,0,0)*f(t).
# Tensor shear is half engineering shear. This is NOT uniaxial stress.
# Three fully reversed cycles; dt=0.01 is a starting discretization, not a
# convergence claim. Compare a repeat at dt=0.005 before generating data.

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 1
  ny = 1
  xmin = 0.0  xmax = 1.0
  ymin = 0.0  ymax = 1.0
  elem_type = QUAD4
[]

[Variables]
  [disp_x]
    family = LAGRANGE
    order  = FIRST
  []
  [disp_y]
    family = LAGRANGE
    order  = FIRST
  []
[]

# Current syntax; Modules/TensorMechanics/Master is its deprecated alias.
[Physics/SolidMechanics/QuasiStatic]
  displacements = 'disp_x disp_y'
  [all]
    displacements                 = 'disp_x disp_y'
    add_variables                 = false
    strain                        = SMALL
    incremental                   = true                # plasticity needs incremental form
    generate_output               = 'stress_xx stress_yy stress_zz stress_xy strain_xx strain_yy strain_zz strain_xy plastic_strain_xx plastic_strain_yy plastic_strain_zz plastic_strain_xy'
    planar_formulation            = PLANE_STRAIN
  []
[]

# -----------------------------------------------------------------------------
# Cyclic loading: 3 fully reversed cycles over time t in [0, 3], period 1.
# Each cycle: load 0..+1 (t 0 -> 0.25), unload and reverse +1..-1 (0.25 -> 0.75),
# reload -1..0 (0.75 -> 1). Peaks +1 at t = 0.25, 1.25, 2.25; -1 at 0.75,
# 1.75, 2.75; zero at every half-integer and integer t.
# The "cyclic_factor" function returns a value in [-1, 1] that drives the load.
# -----------------------------------------------------------------------------
[Functions]
  [cyclic_factor]
    type = ParsedFunction
    # 3-cycle reversed triangle wave: 0 -> 1 -> -1 -> 0, three times, over t in [0, 3]
    expression = 'if(t-floor(t) < 0.25, 4*(t-floor(t)), if(t-floor(t) < 0.75, 2-4*(t-floor(t)), 4*(t-floor(t))-4))'
  []
  [eps_xx_amp]
    type = ParsedFunction
    expression = '0.02'                     # Overridden via CLI by the driver
  []
  [eps_yy_amp]
    type = ParsedFunction
    expression = '0.0'
  []
  [gamma_xy_amp]
    type = ParsedFunction
    expression = '0.0'
  []
  # Homogeneous strain: every boundary node moves as u = eps(t) . x, i.e.
  #   u_x = (eps_xx x + gamma_xy/2 y) f(t),   u_y = (gamma_xy/2 x + eps_yy y) f(t).
  # One BC per component on all four edges. (Separate per-edge BCs disagree at
  # the shared corner nodes, and the element then carries about half the
  # requested strain plus a spurious shear -- measured 2026-10-03.)
  [u_x_all]
    type = ParsedFunction
    expression = '(eps_xx_amp_val * x + 0.5 * gamma_xy_amp_val * y) * cyclic_factor_val'
    symbol_names = 'eps_xx_amp_val gamma_xy_amp_val cyclic_factor_val'
    symbol_values = 'eps_xx_amp gamma_xy_amp cyclic_factor'
  []
  [u_y_all]
    type = ParsedFunction
    expression = '(0.5 * gamma_xy_amp_val * x + eps_yy_amp_val * y) * cyclic_factor_val'
    symbol_names = 'gamma_xy_amp_val eps_yy_amp_val cyclic_factor_val'
    symbol_values = 'gamma_xy_amp eps_yy_amp cyclic_factor'
  []
[]

[BCs]
  [ux_all]
    type     = FunctionDirichletBC
    variable = disp_x
    boundary = 'left right top bottom'
    function = u_x_all
  []
  [uy_all]
    type     = FunctionDirichletBC
    variable = disp_y
    boundary = 'left right top bottom'
    function = u_y_all
  []
[]

# -----------------------------------------------------------------------------
# Constitutive model: hill.
# -----------------------------------------------------------------------------
[Materials]
  [elasticity_tensor]
    type            = ComputeIsotropicElasticityTensor
    youngs_modulus  = 100.0
    poissons_ratio  = 0.3
  []
  [hill_tensor]
    type = HillConstants
    # F G H L M N: axes aligned with the global axes.
    hill_constants = '0.6 0.4 0.7 1.5 1.5 1.5'
  []
  [plasticity]
    type = HillElastoPlasticityStressUpdate
    yield_stress = 1.0
    hardening_constant = 0.0
    hardening_exponent = 1.0
  []
  [stress]
    type             = ComputeMultipleInelasticStress
    inelastic_models = 'plasticity'
  []
[]

[Postprocessors]
  [stress_zz_avg]
    type = ElementAverageValue
    variable = stress_zz
  []
  [strain_zz_avg]
    type = ElementAverageValue
    variable = strain_zz
  []
  [plastic_strain_zz_avg]
    type = ElementAverageValue
    variable = plastic_strain_zz
  []
  [stress_xx_avg]
    type = ElementAverageValue
    variable = stress_xx
  []
  [stress_yy_avg]
    type = ElementAverageValue
    variable = stress_yy
  []
  [stress_xy_avg]
    type = ElementAverageValue
    variable = stress_xy
  []
  [strain_xx_avg]
    type = ElementAverageValue
    variable = strain_xx
  []
  [strain_yy_avg]
    type = ElementAverageValue
    variable = strain_yy
  []
  [strain_xy_avg]
    type = ElementAverageValue
    variable = strain_xy
  []
  [plastic_strain_xx_avg]
    type = ElementAverageValue
    variable = plastic_strain_xx
  []
  [plastic_strain_yy_avg]
    type = ElementAverageValue
    variable = plastic_strain_yy
  []
  [plastic_strain_xy_avg]
    type = ElementAverageValue
    variable = plastic_strain_xy
  []
[]

[Executioner]
  type       = Transient
  solve_type = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol = 1e-10
  l_max_its  = 50
  start_time = 0.0
  end_time   = 3.0                       # 3 cycles
  dt         = 0.01                       # 300 prescribed increments; initial state gives 301 rows
[]

[Outputs]
  file_base = constitutive_hill
  execute_on = 'INITIAL TIMESTEP_END'
  csv = true
  exodus = false
[]
