# =============================================================================
# Module 0 / moose_cantilever_arcc.md: 2-D cantilever beam, clamped at x=0, pressure on top
# =============================================================================
# Geometry: rectangle L x h = 1.0 x 0.1  (slenderness L/h = 10), unit thickness.
# Material: linear isotropic elastic, E = 1, nu = 0.3, PLANE STRAIN
#           (same dimensionless convention as plate_with_hole.i).
# Loading:  uniform pressure q = 1e-6 on the top face (traction t_y = -q).
# BCs:      left edge fully clamped (u_x = u_y = 0); all other edges natural.
# Check:    tip deflection at (L, h/2) vs Timoshenko beam theory with the
#           plane-strain bending modulus E* = E/(1-nu^2):
#              delta = q L^4 / (8 E* I)  +  q L^2 / (2 kappa G A),  I = h^3/12
# =============================================================================

[Mesh]
  [beam]
    type = GeneratedMeshGenerator
    dim  = 2
    nx   = 100
    ny   = 10
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 0.1
  []
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
    displacements      = 'disp_x disp_y'
    add_variables      = false
    strain             = SMALL
    incremental        = false
    generate_output    = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    planar_formulation = PLANE_STRAIN
  []
[]

[BCs]
  [clamp_x]
    type     = DirichletBC
    variable = disp_x
    boundary = 'left'
    value    = 0.0
  []
  [clamp_y]
    type     = DirichletBC
    variable = disp_y
    boundary = 'left'
    value    = 0.0
  []
  [top_pressure]
    # pressure q pushing DOWN on the top face: traction t_y = -q  (natural BC)
    type     = NeumannBC
    variable = disp_y
    boundary = 'top'
    value    = -1.0e-6
  []
[]

[Materials]
  [elasticity_tensor]
    type           = ComputeIsotropicElasticityTensor
    youngs_modulus = 1.0
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]

[Postprocessors]
  [tip_disp_y]
    type     = PointValue
    variable = disp_y
    point    = '1.0 0.05 0'      # mid-height of the free end = beam axis
  []
  [max_vonmises_stress]
    type       = ElementExtremeValue
    variable   = vonmises_stress
    value_type = max
  []
  [num_elements]
    type = NumElements
  []
[]

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol          = 1e-10
  l_max_its           = 50
[]

[Outputs]
  exodus = true
  csv    = true
[]
