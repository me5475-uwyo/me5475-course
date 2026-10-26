# =============================================================================
# Module 5 / Lab 5 -- Lab 0's plate with a hole, with a trained network as the material
# =============================================================================
#
# Course:  ML for Computational Solid Mechanics, UW Fall 2026
# Module:  5 (NN-FEA integration with MOOSE)
# Status:  validated on ARCC 2026-10-03 with the course executable
#          /project/me5475/software/moose_torch/me5475/me5475-opt
#          (module_5/readings/measured_results.md)
#
# Problem: the SAME plate as module_0/examples/plate_with_hole.i. A quarter annulus
#   (hole r = 0.1, outer edge R = 1), symmetry on x = 0 and y = 0, far-field tension
#   t_x = sigma_inf * cos(theta) on the outer arc, small strain, PLANE_STRAIN.
#   The mesh generator, the boundary conditions and the solver are Lab 0's. What differs:
#
#   1. THE MATERIAL. ComputeTorchScriptStress (a course material, compiled into the
#      course executable) loads a TorchScript file and evaluates the network at every
#      quadrature point:  (eps_xx, eps_yy, gamma_xy) in  ->  (sigma_xx, sigma_yy, sigma_xy)
#      out, with gamma_xy = 2 eps_xy. Newton's tangent d sigma / d eps comes from
#      automatic differentiation through the network. The network is converted to
#      double precision when it is loaded. sigma_zz is set to 0: the network does not
#      predict it (see vonmises_pe below).
#      Lab 0's two elastic materials stay in the file. The `active` line picks which
#      one is used, so the elastic reference is the same file with one override:
#          me5475-opt -i plate_with_hole_nn_material.i \
#              Materials/active='elasticity_tensor elastic_stress'
#
#   2. THE LOAD. sigma_inf = 1e-3 here, not Lab 0's 1. A Lab 1 network has seen strains of a
#      few times 1e-3 only (the course data: |eps_xx| <= 3.4e-3, |eps_yy| <= 2.5e-3,
#      |gamma_xy| <= 5.02e-3). With E = 1, Lab 0's load strains the plate up to about 2.4 at
#      this mesh. At sigma_inf = 1e-3 the largest strain is about 2.4e-3, and with the course
#      reference network the final element-average strains of all but a few elements at the
#      hole lie inside the convex hull of the Lab 1 CSV's strains (measured_results.md section
#      3.1; a coverage proxy: quadrature points and Newton iterates are not checked). The
#      elastic problem is linear, so its answer simply scales by 1e-3. The min_/max_strain_*
#      postprocessors report the extremes of the final element-average strains.
#      sigma_inf=1 on the command line shows what extrapolation does (Lab 5 Task 4).
#
#   3. refine = 3 (12 288 elements), not Lab 0's finest 4: the network run is about four times
#      cheaper, and the L2 network-vs-elastic differences of u and sigma_xx change little from
#      refine 0 to 4; peak values are more mesh-sensitive (measured_results.md section 3.2).
#      refine=4 on the command line gives Lab 0's mesh.
#   4. nl_max_its = 15 (Lab 0: the default 50), see [Executioner].
#
# Run on ARCC with run_nn_simulation.sbatch (it runs the network and the elastic
# reference, one after the other). Compare them with comparison_three_methods.py.
# =============================================================================

sigma_inf = 1e-3                 # far-field stress; 1e-3 keeps the strains inside Lab 1's data
refine    = 3                    # 192 * 4^3 = 12 288 elements (Lab 0's finest, refine = 4, is 49 152)
nu_vm     = 0.3                  # Poisson's ratio used ONLY to rebuild sigma_zz in vonmises_pe

[Mesh]
  [annulus]
    type = AnnularMeshGenerator
    nr   = 8
    nt   = 24
    rmin = 0.1
    rmax = 1.0
    dmin = 0
    dmax = 90
  []
  uniform_refine = ${refine}
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

# Lab 0's action, under its current name (Lab 0's [Modules/TensorMechanics/Master] is the
# deprecated alias of this block and prints deprecation warnings).
[Physics/SolidMechanics/QuasiStatic]
  displacements = 'disp_x disp_y'
  [all]
    displacements      = 'disp_x disp_y'
    add_variables      = false
    strain             = SMALL
    incremental        = false
    planar_formulation = PLANE_STRAIN
    generate_output    = 'stress_xx stress_yy stress_xy stress_zz strain_xx strain_yy strain_xy vonmises_stress'
  []
[]

[BCs]
  [sym_bottom_y]
    type     = DirichletBC
    variable = disp_y
    boundary = dmin               # y = 0
    value    = 0.0
  []
  [sym_left_x]
    type     = DirichletBC
    variable = disp_x
    boundary = dmax               # x = 0
    value    = 0.0
  []
  [tension_outer_x]
    type     = FunctionNeumannBC
    variable = disp_x
    boundary = rmax               # outer arc: t_x = sigma_inf * cos(theta)
    function = '${sigma_inf} * x / sqrt(x*x + y*y)'
  []
[]

# -----------------------------------------------------------------------------
# Materials -- the only physics difference between the two runs.
#   active = 'nn_stress'                          the network (default)
#   active = 'elasticity_tensor elastic_stress'   Lab 0's elastic material (the reference)
# -----------------------------------------------------------------------------
[Materials]
  active = 'nn_stress'
  [nn_stress]
    type     = ComputeTorchScriptStress
    filename = mlp_constitutive_ts.pt   # from export_torchscript.py; NOT the Lab 1 checkpoint
  []
  [elasticity_tensor]
    type           = ComputeIsotropicElasticityTensor
    youngs_modulus = 1.0
    poissons_ratio = 0.3
  []
  [elastic_stress]
    type = ComputeLinearElasticStress
  []
[]

# -----------------------------------------------------------------------------
# von Mises, two ways. MOOSE's vonmises_stress uses the material's sigma_zz: for the
# elastic material sigma_zz = nu (sigma_xx + sigma_yy) (plane strain); for the network it
# is 0, which RAISES the peak von Mises at the hole (Lab 5 Task 3d), for a reason that has
# nothing to do with the network's accuracy.
# vonmises_pe rebuilds sigma_zz = nu_vm (sigma_xx + sigma_yy) from the in-plane element
# stresses for BOTH runs, so the two can be compared like for like. (For the network this
# assumes the nu it should have learned; it is a post-processing choice, not something the
# network predicts. And it is the von Mises of element-average stresses, which differs
# slightly from MOOSE's element average of quadrature-point values.)
# -----------------------------------------------------------------------------
[AuxVariables]
  [vonmises_pe]
    order  = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [vonmises_pe]
    type              = ParsedAux
    variable          = vonmises_pe
    coupled_variables = 'stress_xx stress_yy stress_xy'
    constant_names    = 'nu'
    constant_expressions = '${nu_vm}'
    expression = 'sqrt(0.5*((stress_xx-stress_yy)^2 + (stress_yy-nu*(stress_xx+stress_yy))^2
                  + (nu*(stress_xx+stress_yy)-stress_xx)^2) + 3*stress_xy^2)'
    execute_on        = 'TIMESTEP_END'
  []
[]

[Postprocessors]
  # --- Lab 0's metrics ---
  [max_vonmises_stress]           # MOOSE's von Mises (uses the material's sigma_zz)
    type       = ElementExtremeValue
    variable   = vonmises_stress
    value_type = max
  []
  [max_stress_xx]                 # the hoop stress at the top of the hole
    type       = ElementExtremeValue
    variable   = stress_xx
    value_type = max
  []
  [num_elements]
    type = NumElements
  []
  # --- added for Lab 5 ---
  [max_vonmises_pe]               # von Mises with sigma_zz = nu (sigma_xx + sigma_yy), both runs
    type       = ElementExtremeValue
    variable   = vonmises_pe
    value_type = max
  []
  [stress_xx_hole_top]            # the element at (0, 0.1)
    type     = PointValue
    variable = stress_xx
    point    = '0 0.1 0'
  []
  [ux_hole_side]                  # displacements at four points on the symmetry lines
    type     = PointValue
    variable = disp_x
    point    = '0.1 0 0'
  []
  [uy_hole_top]
    type     = PointValue
    variable = disp_y
    point    = '0 0.1 0'
  []
  [ux_outer]
    type     = PointValue
    variable = disp_x
    point    = '1 0 0'
  []
  [uy_outer]
    type     = PointValue
    variable = disp_y
    point    = '0 1 0'
  []
  [reaction_x_left]               # force on the x = 0 symmetry edge; equilibrium: about -sigma_inf
    type          = SidesetReaction
    boundary      = dmax
    stress_tensor = stress
    direction     = '1 0 0'
  []
  # --- the strain range the material saw (Lab 1 trained on [-0.005, 0.005]; gamma_xy = 2 strain_xy) ---
  [max_strain_xx]
    type       = ElementExtremeValue
    variable   = strain_xx
    value_type = max
  []
  [min_strain_xx]
    type       = ElementExtremeValue
    variable   = strain_xx
    value_type = min
  []
  [max_strain_yy]
    type       = ElementExtremeValue
    variable   = strain_yy
    value_type = max
  []
  [min_strain_yy]
    type       = ElementExtremeValue
    variable   = strain_yy
    value_type = min
  []
  [max_strain_xy]
    type       = ElementExtremeValue
    variable   = strain_xy
    value_type = max
  []
  [min_strain_xy]
    type       = ElementExtremeValue
    variable   = strain_xy
    value_type = min
  []
  # --- solver cost ---
  [nl_its]                        # Newton iterations (elastic: 1; the network: more, it is nonlinear)
    type = NumNonlinearIterations
  []
  [residual_evals]                # every residual evaluation runs the network at every quadrature point
    type = NumResidualEvaluations
  []
[]

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol          = 1e-10
  l_max_its           = 50
  nl_max_its          = 15    # Lab 0 used the default 50. Our converged runs of this plate needed at most 5. Many
                              # more is a sign of trouble worth checking (strain postprocessors, Task 4),
                              # and 50 slow iterations waste the job.
[]

[Outputs]
  exodus    = true
  csv       = true
  file_base = plate_nn_out          # run_nn_simulation.sbatch names the elastic run plate_elastic_out
[]
