# =============================================================================
# Module 3 / Lab 3: MOOSE reference for the plate-with-hole PINN
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  3 (PINNs), Lab 3 Task 2 (PINN vs MOOSE comparison) and Task 5
#          (synthetic DIC measurements for the inverse PINN)
#
# Why this file exists:
#   Lab 0's plate_with_hole.i solves a quarter ANNULUS (r in [0.1, 1]) with a
#   far-field traction cos(theta) on the outer arc.  The Lab 3 PINN
#   (plate_with_hole_fixed.py) solves a different problem: a unit SQUARE with a
#   quarter-disk hole and a prescribed displacement on the right edge.  This
#   input solves exactly the PINN's problem so the two can be compared.
#
# Problem (identical to plate_with_hole_fixed.py):
#   Domain:   [0,1] x [0,1] minus the quarter disk r < 0.1 centred at (0, 0).
#   Material: linear isotropic elasticity, E = 1, nu = 0.3, PLANE_STRAIN,
#             SMALL strain.
#   BCs:      left   (x = 0):  u_x = 0          (u_y free, zero shear traction)
#             bottom (y = 0):  u_y = 0          (u_x free, zero shear traction)
#             right  (x = 1):  u_x = 1          (u_y free, zero shear traction)
#             top    (y = 1):  traction-free    (natural BC, no entry)
#             hole   (r = 0.1): traction-free   (natural BC, no entry)
#
# Mesh (framework-only generators; no Reactor module needed):
#   Two mapped patches, each built as a GeneratedMesh on the unit square in
#   parameter space (xi, eta) and bent into place by ParsedNodeTransform:
#     patch A: theta = eta * 45 deg,         hole arc  ->  right edge  x = 1
#     patch B: theta = 45 deg + eta * 45 deg, hole arc ->  top edge    y = 1
#   Along each ray the radius is graded geometrically (log-polar grading),
#       r(xi) = r_hole * (r_edge(theta) / r_hole)^xi,
#   with r_edge = 1/cos(theta) (patch A) or 1/sin(theta) (patch B), so the
#   elements are smallest at the hole, where the stress concentrates, and
#   keep a near-constant aspect ratio outward.  QUAD9 elements: every node,
#   including the mid-edge nodes, is placed on the exact arc r = 0.1, so the
#   hole is body-fitted (no staircase).  theta increases with eta and r with
#   xi, which keeps the element Jacobians positive.  The two patches share the
#   45-degree diagonal node-for-node and are stitched there.
#
# Resolution:
#   nt = elements along each 45-degree arc; nr = elements along each ray.
#   Refine by overriding the top-level variables (keeps every node on the
#   exact arc):   rom_opt-opt -i plate_square_hole_reference.i nt=96 nr=192
#   (convergence levels nt/nr = 24/48, 48/96, 96/192, 192/384 were checked;
#   the default 192/384 is the shared reference)
#   or by one uniform refinement level (children follow the QUAD9 geometry):
#                 rom_opt-opt -i plate_square_hole_reference.i Mesh/uniform_refine=1
#
# Outputs:
#   Exodus: disp_x, disp_y and NODAL (LAGRANGE) stress_xx, stress_yy,
#   stress_xy, vonmises_stress -- the names extract_moose_reference.py reads.
#   CSV: postprocessors used as verification checks (see [Postprocessors]).
#
# Shared course copy (staged on ARCC):
#   /project/me5475/examples/lab3_reference/plate_square_hole_reference_out.e
# =============================================================================

hole_radius = 0.1
nt = 192           # elements along each 45-degree arc (2*nt around the hole)
nr = 384           # elements along each ray, hole -> outer edge
# The defaults reproduce the shared reference (147,456 QUAD9, ~2.2M dofs,
# ~1.5 min on 8 ranks).  For a quick look use  nt=24 nr=48  (seconds, 1 rank).

[Mesh]
  # ---- patch A: 0 <= theta <= 45 deg, hole -> right edge (x = 1) ----------
  [patch_a_param]
    type = GeneratedMeshGenerator
    dim = 2
    nx = ${nr}               # xi  : radial direction
    ny = ${nt}               # eta : angular direction
    elem_type = QUAD9
    boundary_name_prefix = pa
  []
  [patch_a]
    type = ParsedNodeTransformGenerator
    input = patch_a_param
    constant_names = 'rh q'
    constant_expressions = '${hole_radius} 0.785398163397448310'   # q = pi/4
    x_function = 'rh * (1 / (rh * cos(q*y)))^x * cos(q*y)'
    y_function = 'rh * (1 / (rh * cos(q*y)))^x * sin(q*y)'
    enable_jit = false
  []
  [patch_a_named]
    type = RenameBoundaryGenerator
    input = patch_a
    old_boundary = 'pa_left pa_right pa_bottom pa_top'
    new_boundary = 'hole    right    bottom    diag_a'
  []

  # ---- patch B: 45 <= theta <= 90 deg, hole -> top edge (y = 1) -----------
  [patch_b_param]
    type = GeneratedMeshGenerator
    dim = 2
    nx = ${nr}
    ny = ${nt}
    elem_type = QUAD9
    boundary_name_prefix = pb
  []
  [patch_b]
    type = ParsedNodeTransformGenerator
    input = patch_b_param
    constant_names = 'rh q'
    constant_expressions = '${hole_radius} 0.785398163397448310'
    x_function = 'rh * (1 / (rh * sin(q*(1+y))))^x * cos(q*(1+y))'
    y_function = 'rh * (1 / (rh * sin(q*(1+y))))^x * sin(q*(1+y))'
    enable_jit = false
  []
  [patch_b_named]
    type = RenameBoundaryGenerator
    input = patch_b
    old_boundary = 'pb_left pb_right pb_bottom pb_top'
    new_boundary = 'hole    top      diag_b    left'
  []

  # ---- stitch the two patches on the 45-degree diagonal --------------------
  # (the stitch consumes the two diagonal sidesets, so none are left behind)
  [stitched]
    type = StitchMeshGenerator
    inputs = 'patch_a_named patch_b_named'
    stitch_boundaries_pairs = 'diag_a diag_b'
  []
  # ---- snap round-off (e.g. cos(pi/2) = 6e-17) onto the exact straight edges
  [plate]
    type = ParsedNodeTransformGenerator
    input = stitched
    x_function = 'if(x < 1e-12, 0, if(x > 1 - 1e-12, 1, x))'
    y_function = 'if(y < 1e-12, 0, if(y > 1 - 1e-12, 1, y))'
    enable_jit = false
  []

  uniform_refine = 0
[]

# -----------------------------------------------------------------------------
# Variables: quadratic displacements on the QUAD9 mesh
# -----------------------------------------------------------------------------
[Variables]
  [disp_x]
    family = LAGRANGE
    order  = SECOND
  []
  [disp_y]
    family = LAGRANGE
    order  = SECOND
  []
[]

# -----------------------------------------------------------------------------
# Same solid-mechanics action as Lab 0 (current syntax; Lab 0's
# Modules/TensorMechanics/Master is the deprecated alias of this block).
# Stress outputs are LAGRANGE (nodal) so the extractor can interpolate them.
# Side effect: Newton reports 2 iterations instead of 1 (the nodal material
# evaluation perturbs the first residual only); the converged displacements
# are identical to a MONOMIAL-output run (checked 2026-09-24).
# -----------------------------------------------------------------------------
[Physics/SolidMechanics/QuasiStatic]
  displacements = 'disp_x disp_y'
  [all]
    displacements          = 'disp_x disp_y'
    add_variables          = false
    strain                 = SMALL
    incremental            = false
    planar_formulation     = PLANE_STRAIN
    generate_output        = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    material_output_family = LAGRANGE
    material_output_order  = FIRST
  []
[]

# -----------------------------------------------------------------------------
# Boundary conditions (top and hole are traction-free: natural BCs, no entry)
# -----------------------------------------------------------------------------
[BCs]
  [left_ux]
    type     = DirichletBC
    variable = disp_x
    boundary = left
    value    = 0.0
  []
  [bottom_uy]
    type     = DirichletBC
    variable = disp_y
    boundary = bottom
    value    = 0.0
  []
  [right_ux]
    type     = DirichletBC
    variable = disp_x
    boundary = right
    value    = 1.0
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

# -----------------------------------------------------------------------------
# Postprocessors = verification checks
#   area            exact 1 - pi*0.1^2/4 = 0.99214602  (body-fitted geometry)
#   hole_length     exact pi*0.1/2       = 0.15707963
#   ux_right_min/max            both 1 (Dirichlet)
#   reaction_right_x / _left_x  equal and opposite (global equilibrium)
#   traction_top_* / traction_hole_*  -> 0 with refinement (natural BCs)
#   stress_xx_hole_top          sigma_xx at (0, 0.1), the stress-concentration point
#   avg_stress_xx_right         nominal stress on the loaded edge
# -----------------------------------------------------------------------------
[Postprocessors]
  [max_vonmises_stress]
    type       = ElementExtremeValue
    variable   = vonmises_stress
    value_type = max
  []
  [max_stress_xx]
    type       = ElementExtremeValue
    variable   = stress_xx
    value_type = max
  []
  [stress_xx_hole_top]
    type     = PointValue
    variable = stress_xx
    point    = '0 ${hole_radius} 0'
  []
  [avg_stress_xx_right]
    type     = SideAverageValue
    variable = stress_xx
    boundary = right
  []
  [reaction_right_x]
    type          = SidesetReaction
    boundary      = right
    stress_tensor = stress
    direction     = '1 0 0'
  []
  [reaction_left_x]
    type          = SidesetReaction
    boundary      = left
    stress_tensor = stress
    direction     = '1 0 0'
  []
  [traction_top_x]
    type          = SidesetReaction
    boundary      = top
    stress_tensor = stress
    direction     = '1 0 0'
  []
  [traction_top_y]
    type          = SidesetReaction
    boundary      = top
    stress_tensor = stress
    direction     = '0 1 0'
  []
  [traction_hole_x]
    type          = SidesetReaction
    boundary      = hole
    stress_tensor = stress
    direction     = '1 0 0'
  []
  [traction_hole_y]
    type          = SidesetReaction
    boundary      = hole
    stress_tensor = stress
    direction     = '0 1 0'
  []
  [ux_right_min]
    type       = NodalExtremeValue
    variable   = disp_x
    boundary   = right
    value_type = min
  []
  [ux_right_max]
    type       = NodalExtremeValue
    variable   = disp_x
    boundary   = right
    value_type = max
  []
  [area]
    type = VolumePostprocessor
  []
  [hole_length]
    type     = AreaPostprocessor
    boundary = hole
  []
  [num_elements]
    type = NumElements
  []
  [num_dofs]
    type = NumDOFs
  []
[]

# Full SMP: include the disp_x/disp_y coupling blocks so Newton converges in one
# step on this linear problem (the default preconditioner drops them).
[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'lu       mumps'
  nl_rel_tol          = 1e-10
  nl_abs_tol          = 1e-12
  l_max_its           = 50
[]

[Outputs]
  exodus = true
  csv    = true
[]
