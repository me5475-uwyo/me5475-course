# =============================================================================
# Module 3 / L13: the 1-D diffusion problem, solved with MOOSE
# =============================================================================
#
# The same problem as pinn_diffusion_1d.py:
#     -u''(x) = pi^2 sin(pi x)    on (0, 1),     u(0) = u(1) = 0,
# exact solution u(x) = sin(pi x).
#
# Weak form (multiply by a test function psi that vanishes where u is
# prescribed, integrate by parts):
#     integral u' psi' dx  -  integral f psi dx  =  0      for every psi
#     \_______________/      \______________/
#      Diffusion kernel       BodyForce kernel (f = the function 'forcing')
#
# Walkthrough, block by block: module_3/readings/moose_diffusion_walkthrough.md
#
# Run (as a job; it takes seconds):   sbatch run_moose_diffusion.sbatch 1d
# Finer mesh, from the command line:  sbatch run_moose_diffusion.sbatch 1d nx=100
# =============================================================================

nx = 50            # number of elements; override on the command line: nx=100

[Mesh]
  [line]
    type = GeneratedMeshGenerator
    dim  = 1
    nx   = ${nx}
    xmin = 0.0
    xmax = 1.0
  []
[]

# One unknown field, u: linear Lagrange (the default), one value per node.
[Variables]
  [u]
  []
[]

# Functions of (x, y, z, t), written as text. 'exact' is used only to check the
# answer (the l2_error postprocessor below); it never enters the solve.
[Functions]
  [forcing]
    type       = ParsedFunction
    expression = 'pi*pi*sin(pi*x)'
  []
  [exact]
    type       = ParsedFunction
    expression = 'sin(pi*x)'
  []
[]

# The two terms of the weak form. Their sum is the residual MOOSE drives to 0.
[Kernels]
  [diffusion]   # + integral u' psi' dx
    type     = Diffusion
    variable = u
  []
  [source]      # - integral f psi dx
    type     = BodyForce
    variable = u
    function = forcing
  []
[]

# GeneratedMeshGenerator names the two ends of a 1-D mesh 'left' and 'right'.
[BCs]
  [ends]
    type     = DirichletBC
    variable = u
    boundary = 'left right'
    value    = 0.0
  []
[]

# The problem is linear, so Newton converges in one step; LU is a direct solve.
[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'lu       mumps'
[]

# Checks, written to diffusion_1d_out.csv:
#   l2_error  = || u_h - sin(pi x) ||_L2  (absolute), falls ~4x when nx doubles
#   u_mid     = u_h at x = 0.5            (exact value 1)
[Postprocessors]
  [l2_error]
    type     = ElementL2Error
    variable = u
    function = exact
  []
  [u_mid]
    type     = PointValue
    variable = u
    point    = '0.5 0 0'
  []
  [num_dofs]
    type = NumDOFs
  []
[]

# u along the line, 101 evenly spaced points -> diffusion_1d_out_u_line_*.csv
# (columns id, u, x, y, z), ready to plot with pandas.
[VectorPostprocessors]
  [u_line]
    type        = LineValueSampler
    variable    = u
    start_point = '0 0 0'
    end_point   = '1 0 0'
    num_points  = 101
    sort_by     = x
  []
[]

[Outputs]
  exodus = true                # diffusion_1d_out.e: the full solution (ParaView)
  csv    = true                # diffusion_1d_out.csv + the u_line CSV
[]
