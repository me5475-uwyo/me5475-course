# =============================================================================
# Module 3 / L13: the 2-D diffusion problem, solved with MOOSE
# =============================================================================
#
# The same problem as pinn_diffusion_2d.py:
#     -(u_x1x1 + u_x2x2) = 2 pi^2 sin(pi x1) sin(pi x2)   on (0, 1)^2,
#     u = 0 on all four sides,
# exact solution u = sin(pi x1) sin(pi x2).  MOOSE calls the coordinates x, y.
#
# Deliberately parallel to diffusion_1d.i; `diff` the two. Only four things
# change: the mesh is 2-D, the forcing and exact functions gain a sin(pi y),
# the Dirichlet BC covers four sides instead of two, and the checks sample
# the centre and the line y = 0.5.
#
# Walkthrough: module_3/readings/moose_diffusion_walkthrough.md
#
# Run (as a job; it takes seconds):   sbatch run_moose_diffusion.sbatch 2d
# Finer mesh, from the command line:  sbatch run_moose_diffusion.sbatch 2d nx=100
# =============================================================================

nx = 50            # elements per side (ny follows nx); override: nx=100

[Mesh]
  [square]
    type = GeneratedMeshGenerator
    dim  = 2
    nx   = ${nx}
    ny   = ${nx}
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
  []
[]

[Variables]
  [u]
  []
[]

[Functions]
  [forcing]
    type       = ParsedFunction
    expression = '2*pi*pi*sin(pi*x)*sin(pi*y)'
  []
  [exact]
    type       = ParsedFunction
    expression = 'sin(pi*x)*sin(pi*y)'
  []
[]

# Unchanged from 1-D: Diffusion is integral grad(u).grad(psi) in any dimension.
[Kernels]
  [diffusion]
    type     = Diffusion
    variable = u
  []
  [source]
    type     = BodyForce
    variable = u
    function = forcing
  []
[]

# A 2-D GeneratedMeshGenerator names its sides left, right, bottom, top.
[BCs]
  [sides]
    type     = DirichletBC
    variable = u
    boundary = 'left right bottom top'
    value    = 0.0
  []
[]

[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'lu       mumps'
[]

# Checks, written to diffusion_2d_out.csv:
#   l2_error    = || u_h - exact ||_L2  (absolute), falls ~4x when nx doubles
#   u_centre    = u_h at (0.5, 0.5)     (exact value 1)
[Postprocessors]
  [l2_error]
    type     = ElementL2Error
    variable = u
    function = exact
  []
  [u_centre]
    type     = PointValue
    variable = u
    point    = '0.5 0.5 0'
  []
  [num_dofs]
    type = NumDOFs
  []
[]

# u along the middle line y = 0.5, where the exact solution is sin(pi x).
[VectorPostprocessors]
  [u_line]
    type        = LineValueSampler
    variable    = u
    start_point = '0 0.5 0'
    end_point   = '1 0.5 0'
    num_points  = 101
    sort_by     = x
  []
[]

[Outputs]
  exodus = true
  csv    = true
[]
