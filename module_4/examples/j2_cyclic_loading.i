# =============================================================================
# Module 4 / Example 1: J2 Plasticity Single Element under Cyclic Loading
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  4 (Data-Driven Constitutive Modeling)
# Status:  validated 2026-10-03 with the course MOOSE build on ARCC
#          (module_4/readings/measured_results.md, section 1)
#
# Problem:
#   A single 2-D plane-strain unit-quad element loaded by displacement BCs that
#   follow three fully reversed strain cycles (see Loading below). The driver script
#   generate_plasticity_data.py runs this input file ~200 times with different
#   random load-path parameters and assembles the resulting stress-strain
#   trajectories into a single .npz file.
#
# Material: J2 plasticity with linear isotropic hardening.
#     E = 100,  nu = 0.3,  yield_stress = 1.0,  hardening_constant = 5.0
#
# Loading: displacement u = eps(t) . x on all four edges, so the element carries
# a homogeneous strain, parameterized by three amplitudes (eps_xx_max,
# eps_yy_max, gamma_xy_max). The cyclic time function makes 3 fully reversed
# cycles: 0 -> +amplitude -> -amplitude -> 0, three times. The strain direction
# is fixed (all three components share one time function); only its sign flips.
#
# Output: CSV with time series of the element averages of strain_xx/yy/xy,
#         stress_xx/yy/xy and plastic_strain_xx/yy/xy (tensor shear components;
#         the driver doubles the xy strains to engineering gamma). No scalar
#         accumulated plastic strain (alpha) is written.
#
# Usage (the driver does this hundreds of times):
#   moose-opt -i j2_cyclic_loading.i \
#     Functions/eps_xx_amp/expression=0.01 \
#     Functions/eps_yy_amp/expression=-0.005 \
#     Functions/gamma_xy_amp/expression=0.003 \
#     Outputs/file_base=run_0042
# (On ARCC the executable is /project/me5475/software/moose_torch/me5475/me5475-opt,
#  run inside a job: see generate_data.sbatch.)
# =============================================================================

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
    generate_output               = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy plastic_strain_xx plastic_strain_yy plastic_strain_xy'
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
    expression = '0.01'                     # Overridden via CLI by the driver
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
# J2 plasticity with isotropic linear hardening.
# -----------------------------------------------------------------------------
[Materials]
  [elasticity_tensor]
    type            = ComputeIsotropicElasticityTensor
    youngs_modulus  = 100.0
    poissons_ratio  = 0.3
  []
  [plasticity]
    type                  = IsotropicPlasticityStressUpdate
    yield_stress          = 1.0
    hardening_constant    = 5.0          # linear isotropic hardening
  []
  [stress]
    type             = ComputeMultipleInelasticStress
    inelastic_models = 'plasticity'
  []
[]

[Postprocessors]
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
  dt         = 0.05                       # ~60 timesteps per trajectory
[]

[Outputs]
  csv = true
  exodus = false
[]
