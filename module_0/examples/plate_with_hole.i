# =============================================================================
# Module 0 / Example 1: 2D Plate with Circular Hole, Uniaxial Tension
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  0 (Setup, Agentic Workflow, ARCC + MOOSE Orientation)
# Status:  Base solve validated 2026-05-20; two-metric ARCC run validated 2026-08-25
#
# Problem:
#   A quarter-annulus model (0 <= theta <= 90°) of a large plate (R = 1)
#   with a centered circular hole (r = 0.1, R/r = 10) under uniaxial
#   far-field tension sigma_inf = 1 in the x-direction.
#   Symmetry BCs eliminate rigid-body motion; the exact far-field traction
#   t_x = cos(theta) is applied on the outer arc.
#
# Material: linear isotropic elastic, E = 1, nu = 0.3 (matches Min Lin's PINN
# notebook in PINN_Example/2D-Hole-Fix-E-Nu, so MOOSE and PINN solutions can
# be compared directly in Module 3).
#
# Mesh: AnnularMeshGenerator produces a conforming quad mesh with a smooth
# circular inner boundary.  This is a MOOSE framework object -- no Reactor
# module required.  AnnularMeshGenerator names its arc sidesets 'rmin' and
# 'rmax' and its straight radial edges 'dmin' and 'dmax'.
#
# Homework deliverable (Lab 0):
#   Run with uniform_refine in {0, 1, 2, 3, 4}, collect the
#   max_vonmises_stress, max_stress_xx, and num_elements postprocessor values;
#   plot both peak-stress metrics vs element count and compare with the
#   infinite-plate references: hoop sigma_xx ~ 3*sigma_inf and, for
#   PLANE_STRAIN, von Mises ~ sqrt(1 - nu + nu^2)*3 ~ 2.67 (nu=0.3).
#   At refine 4, expect measured peaks ~2.88 (hoop) and ~2.48 (von Mises).
# =============================================================================

[Mesh]
  # Quarter annulus: 0 <= theta <= 90 deg, rmin=0.1 (hole), rmax=1.0 (plate edge).
  # nt elements span the 90-degree arc; nr elements span the radial direction.
  # Sidesets created: 'rmin' (hole), 'rmax' (outer), 'dmin' (y=0), 'dmax' (x=0).
  [annulus]
    type = AnnularMeshGenerator
    nr   = 8
    nt   = 24
    rmin = 0.1
    rmax = 1.0
    dmin = 0
    dmax = 90
  []

  uniform_refine = 0                        # <-- HOMEWORK: students sweep 0..4
[]

# -----------------------------------------------------------------------------
# Variables: displacement components
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Use the TensorMechanics Master action to auto-generate the momentum kernels.
# This expands to StressDivergenceTensors for each displacement component.
# -----------------------------------------------------------------------------
[Modules/TensorMechanics/Master]
  displacements = 'disp_x disp_y'
  [all]
    displacements                 = 'disp_x disp_y'
    add_variables                 = false
    strain                        = SMALL
    incremental                   = false
    generate_output               = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    planar_formulation            = PLANE_STRAIN
  []
[]

# -----------------------------------------------------------------------------
# Boundary conditions
#   'dmin' (y=0 edge):  u_y = 0  -- horizontal symmetry plane
#   'dmax' (x=0 edge):  u_x = 0  -- vertical symmetry plane
#   'rmax' (outer arc): t_x = cos(theta) = x/sqrt(x^2+y^2)  (far-field tension)
#   'rmin' (hole arc):  traction-free (natural BC, no entry needed)
# The two symmetry BCs eliminate all rigid-body modes.
# t_y on 'rmax' is zero (natural BC) since the far-field loading is purely in x.
# -----------------------------------------------------------------------------
[BCs]
  [sym_bottom_y]
    type     = DirichletBC
    variable = disp_y
    boundary = 'dmin'
    value    = 0.0
  []
  [sym_left_x]
    type     = DirichletBC
    variable = disp_x
    boundary = 'dmax'
    value    = 0.0
  []
  [tension_outer_x]
    type     = FunctionNeumannBC
    variable = disp_x
    boundary = 'rmax'
    function = 'x / sqrt(x*x + y*y)'   # = cos(theta) for sigma_inf = 1
  []
[]

# -----------------------------------------------------------------------------
# Material: linear isotropic elasticity, E = 1, nu = 0.3
# (matches Min Lin's PINN setup so we can compare in Module 3)
# -----------------------------------------------------------------------------
[Materials]
  [elasticity_tensor]
    type            = ComputeIsotropicElasticityTensor
    youngs_modulus  = 1.0
    poissons_ratio  = 0.3
  []
  [stress]
    type   = ComputeLinearElasticStress
  []
[]

# -----------------------------------------------------------------------------
# Postprocessors: report the three scalars the homework needs.
#   max_vonmises_stress -- peak vM; max_stress_xx -- peak hoop stress at top
#   num_elements        -- mesh size (grows ~4x per uniform_refine level)
# -----------------------------------------------------------------------------
[Postprocessors]
  [max_vonmises_stress]
    type     = ElementExtremeValue
    variable = vonmises_stress
    value_type = max
  []
  [max_stress_xx]
    type     = ElementExtremeValue
    variable = stress_xx
    value_type = max
  []
  [num_elements]
    type = NumElements
  []
[]

# -----------------------------------------------------------------------------
# Executioner: steady-state nonlinear solve. Linear problem, so 1 iteration.
# -----------------------------------------------------------------------------
[Executioner]
  type           = Steady
  solve_type     = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol     = 1e-10
  l_max_its      = 50
[]

# -----------------------------------------------------------------------------
# Outputs:
#   Exodus for ParaView visualization
#   CSV for the postprocessor values used in the convergence study
# -----------------------------------------------------------------------------
[Outputs]
  exodus = true
  csv    = true
[]
