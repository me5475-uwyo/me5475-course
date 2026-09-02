# =============================================================================
# Module 1 / Example 1: Single-element load sweep for constitutive data generation
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  1 (Math + Mechanics Foundations) -- Lab 1 data source
# Status:  Draft v0.1 -- needs one validation pass on a local MOOSE install
#
# Problem:
#   A 2D plane-strain unit-square mesh of 1x1 LINEAR QUAD ELEMENTS (so a single
#   element, after we ensure num elements = 1). Apply a specified strain field
#   eps = (eps_xx, eps_yy, gamma_xy) as Dirichlet BCs and read out the
#   resulting (uniform) stress field sigma = (sigma_xx, sigma_yy, sigma_xy).
#
#   Material: linear isotropic elastic, E = 1.0, nu = 0.3 (matches M0 example).
#
#   Strategy:
#     - Apply prescribed displacement on the four edges to realize a uniform
#       strain state. For epsilon_xx: u_x(x=L) - u_x(x=0) = eps_xx * L.
#     - For gamma_xy (Voigt shear): displacement BCs that produce pure shear.
#     - For a unit cell of side L=1, the imposed BCs are simply numeric values
#       of eps_xx, eps_yy, gamma_xy directly.
#
#   Output: Postprocessors report element-average stress components and the
#   imposed strain components (echoed). The driver script generate_data.py
#   reads these CSVs and builds the training dataset for Lab 1.
#
# Usage (driver does this hundreds of times):
#   moose-opt -i single_element_loadsweep.i \
#     BCs/eps_xx_val=0.01 \
#     BCs/eps_yy_val=-0.005 \
#     BCs/gamma_xy_val=0.003 \
#     Outputs/file_base=run_0042
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

# Free parameters that the driver overrides via CLI:
[BCs]
  # The strain components we want to impose. The driver overrides these.
  active = 'left_x right_x bottom_y top_y left_y right_y bottom_x top_x'

  # u_x(x=0) = 0 ; u_x(x=L) = eps_xx * L. With L = 1, u_x(right) = eps_xx_val
  [left_x]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0.0
  []
  [right_x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = right
    function = right_x_fn       # see [Functions] below
  []

  # u_y(y=0) = 0 ; u_y(y=L) = eps_yy * L
  [bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0.0
  []
  [top_y]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = top_y_fn
  []

  # Shear: u_x(y=L) = gamma_xy * L (using half-half split for symmetric shear).
  # Simplest: apply u_y on the right boundary = gamma_xy/2 ramp, and u_x on the
  # top boundary = gamma_xy/2 ramp. This gives 2*epsilon_xy = gamma_xy.
  [left_y]
    type = DirichletBC
    variable = disp_y
    boundary = left
    value = 0.0
  []
  [right_y]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = right
    function = right_y_fn
  []
  [bottom_x]
    type = DirichletBC
    variable = disp_x
    boundary = bottom
    value = 0.0
  []
  [top_x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = top
    function = top_x_fn
  []
[]

# These functions reference the strain values the driver provides.
[Functions]
  # Linear ramps in the directions where they apply.
  [right_x_fn]
    type = ParsedFunction
    expression = 'eps_xx_val'                 # x = L = 1, so u_x = eps_xx_val
    symbol_names  = 'eps_xx_val'
    symbol_values = '0.001'                   # CLI override
  []
  [top_y_fn]
    type = ParsedFunction
    expression = 'eps_yy_val'
    symbol_names  = 'eps_yy_val'
    symbol_values = '0.0'
  []
  [right_y_fn]
    type = ParsedFunction
    expression = '0.5 * gamma_xy_val * y'     # u_y = (gamma_xy/2) * y along x=L
    symbol_names  = 'gamma_xy_val'
    symbol_values = '0.0'
  []
  [top_x_fn]
    type = ParsedFunction
    expression = '0.5 * gamma_xy_val * x'     # u_x = (gamma_xy/2) * x along y=L
    symbol_names  = 'gamma_xy_val'
    symbol_values = '0.0'
  []
[]

[Variables]
  [disp_x]
    family = LAGRANGE
    order = FIRST
  []
  [disp_y]
    family = LAGRANGE
    order = FIRST
  []
[]

[Modules/TensorMechanics/Master]
  displacements = 'disp_x disp_y'
  [all]
    displacements = 'disp_x disp_y'
    add_variables = false
    strain = SMALL
    incremental = false
    generate_output = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy'
    planar_formulation = PLANE_STRAIN
  []
[]

[Materials]
  [elastic_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1.0
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeLinearElasticStress
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
  # For Voigt: report 2 * strain_xy (since gamma_xy = 2*eps_xy).
  [strain_xy_avg]
    type = ElementAverageValue
    variable = strain_xy
  []
[]

[Executioner]
  type = Steady
  solve_type = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol = 1e-12
  l_max_its = 50
[]

[Outputs]
  csv = true
  exodus = false        # element-only CSV is enough for data generation; keep small
[]
