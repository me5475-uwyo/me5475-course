# =============================================================================
# Module 6 / Lab 6: the RVE's apparent axial response -- ONE RENDERED EXAMPLE of the template
# (a single circular inclusion of radius 0.1 at the center; the driver makes these automatically)
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  6 (CNNs for Field Prediction)
# Status:  run 500 times per student dataset on MedicineBow (module_6/readings/measured_results.md)
#
# This is a TEMPLATE. The driver script generate_microstructures.py substitutes:
#     1.0 + 4.0 * 0.5 * (1 - tanh(50 * ((x-0.5)^2 + (y-0.5)^2 - 0.01)))   ParsedFunction expression for E(x, y)
#     rve_test      output file base name
#     __FIELD_SAMPLER__  (a line of its own) the stress-field sampler, or nothing
#
# Problem:
#   A unit square, 64 x 64 QUAD4 mesh, small strain, PLANE STRAIN (eps_zz = 0).
#   Young's modulus E(x, y) comes from a ParsedFunction: a smoothed indicator (tanh, scale 50) of the
#   inclusions, so E rises from 1 (matrix) to 5 (inclusion) across a thin band at each interface.
#   nu = 0.3 in both phases.
#
#   Boundary conditions (MIXED, not periodic):
#     u_x = 0     on the left edge     (u_y free there)
#     u_x = 0.001 on the right edge    (u_y free there): macroscopic eps_xx = 0.001
#     u_y = 0     on the whole bottom edge (u_x free there)
#     top edge traction-free
#
#   Output: the volume average of sigma_xx. The driver reports E_app = <sigma_xx> / 0.001, the cell's
#   APPARENT AXIAL RESPONSE under these BCs. It is not periodic homogenization, not the constrained stiffness
#   C_11, and not an unconstrained Young's modulus. For a homogeneous cell these BCs give sigma_yy = 0 and
#   eps_zz = 0, so E_app = E / (1 - nu^2) = 1.0989 E: 1.0989 for all matrix and 5.4945 for all inclusion
#   (checked on MedicineBow, Nov 5, 2026). The loading is displacement-controlled, so E_app grows with the
#   stiffness of any region: every composite lies between those two limits.
# =============================================================================

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 64
  ny = 64
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

[Modules/TensorMechanics/Master]
  displacements = 'disp_x disp_y'
  [all]
    displacements       = 'disp_x disp_y'
    add_variables       = false
    strain              = SMALL
    incremental         = false
    generate_output     = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    planar_formulation  = PLANE_STRAIN
  []
[]

# -----------------------------------------------------------------------------
# Spatially-varying E(x, y) via ParsedFunction.
# The template variable 1.0 + 4.0 * 0.5 * (1 - tanh(50 * ((x-0.5)^2 + (y-0.5)^2 - 0.01))) is substituted by the driver script
# with an expression like:
#   1.0 + (5.0 - 1.0) * min(0.5*(1 - tanh(50*(...))) + ..., 1)
# -----------------------------------------------------------------------------
[Functions]
  [E_field]
    type = ParsedFunction
    expression = '1.0 + 4.0 * 0.5 * (1 - tanh(50 * ((x-0.5)^2 + (y-0.5)^2 - 0.01)))'
  []
  [nu_field]
    type = ParsedFunction
    expression = '0.3'
  []
[]

[AuxVariables]
  [E_aux]
    order  = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [E_aux]
    type      = FunctionAux
    variable  = E_aux
    function  = E_field
    execute_on = INITIAL
  []
[]

# -----------------------------------------------------------------------------
# Material: use ComputeVariableIsotropicElasticityTensor to compute the
# spatially-varying stiffness tensor from the E_aux field.
# -----------------------------------------------------------------------------
[Materials]
  [E_material]
    type     = ParsedMaterial
    property_name = 'E_local'
    coupled_variables = 'E_aux'
    expression = 'E_aux'
  []
  [nu_material]
    type     = GenericConstantMaterial
    prop_names  = 'nu_local'
    prop_values = '0.3'
  []
  [elastic_tensor]
    type             = ComputeVariableIsotropicElasticityTensor
    youngs_modulus   = E_local
    poissons_ratio   = nu_local
    args             = 'E_aux'
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]

# -----------------------------------------------------------------------------
# Mixed BCs: prescribed u_x on left and right, u_y = 0 along the whole bottom, top free
# -----------------------------------------------------------------------------
[BCs]
  [fix_left_x]
    type     = DirichletBC
    variable = disp_x
    boundary = left
    value    = 0.0
  []
  [pull_right_x]
    type     = DirichletBC
    variable = disp_x
    boundary = right
    value    = 0.001         # macroscopic eps_xx = 0.001 / L = 0.001
  []
  [fix_bottom_y]
    type     = DirichletBC
    variable = disp_y
    boundary = bottom
    value    = 0.0
  []
[]

# -----------------------------------------------------------------------------
# Postprocessors: volume-averaged stress and strain (the driver uses stress_xx_avg)
# -----------------------------------------------------------------------------
[Postprocessors]
  [stress_xx_avg]
    type     = ElementAverageValue
    variable = stress_xx
  []
  [stress_yy_avg]
    type     = ElementAverageValue
    variable = stress_yy
  []
  [stress_xy_avg]
    type     = ElementAverageValue
    variable = stress_xy
  []
  [strain_xx_avg]
    type     = ElementAverageValue
    variable = strain_xx
  []
  [E_avg]                          # the average of the smooth E(x, y), not the image's area fraction
    type     = ElementAverageValue
    variable = E_aux
  []
[]

# Per-element stress field sampler for the U-Net stretch task.
# The driver substitutes __FIELD_SAMPLER__ with the [VectorPostprocessors] block
# below when run with --save-stress-fields, and with nothing otherwise (so the
# 500 scalar-only runs stay cheap). The CSV it writes
# (<file_base>_stress_field_*.csv, one row per element with x/y centroid +
# stress_xx/yy/xy) is parsed by generate_microstructures.csv_to_stress_field.


[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol          = 1e-10
  l_max_its           = 50
[]

[Outputs]
  csv       = true
  exodus    = false                 # full field for the U-Net comes from the CSV element sampler above (see __FIELD_SAMPLER__), not Exodus
  file_base = rve_test_out
[]
