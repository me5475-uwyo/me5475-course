"""the MOOSE cantilever reading post-processing: contours from the Exodus file + tip-deflection check.
Run AFTER the SLURM job:  python plot_beam_results.py"""
import numpy as np, csv
import netCDF4
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri

ds = netCDF4.Dataset('cantilever_beam_out.e')
X = ds.variables['coordx'][:]; Y = ds.variables['coordy'][:]
conn = ds.variables['connect1'][:] - 1                       # (nelem, 4) quads
def names(k): return [b''.join(c.compressed()).decode() for c in ds.variables[k][:]]
nod = {n: ds.variables['vals_nod_var%d' % (i+1)][-1] for i, n in enumerate(names('name_nod_var'))}
elm = {n: ds.variables['vals_elem_var%deb1' % (i+1)][-1] for i, n in enumerate(names('name_elem_var'))}

tris = np.vstack([conn[:, [0,1,2]], conn[:, [0,2,3]]])       # quads -> tris for nodal contour
tri = mtri.Triangulation(X, Y, tris)
umag = np.sqrt(nod['disp_x']**2 + nod['disp_y']**2)

fig, axes = plt.subplots(2, 2, figsize=(11, 4.6))
panels = [('|u| — displacement magnitude', ('nodal', umag)),
          ('stress_xx — bending stress',   ('elem',  elm['stress_xx'])),
          ('strain_xx',                    ('elem',  elm['strain_xx'])),
          ('von Mises stress',             ('elem',  elm['vonmises_stress']))]
for ax, (title, (kind, v)) in zip(axes.flat, panels):
    if kind == 'nodal':
        m = ax.tricontourf(tri, v, levels=30, cmap='viridis')
    else:
        verts = np.stack([np.column_stack([X[q], Y[q]]) for q in conn])
        pc = PolyCollection(verts, array=np.asarray(v), cmap='viridis', edgecolor='none')
        ax.add_collection(pc); m = pc
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.1)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    plt.colorbar(m, ax=ax, shrink=0.8)
plt.suptitle('Cantilever, clamped at x=0, pressure q=1e-6 on top — 100×10 mesh, plane strain')
plt.tight_layout(); plt.savefig('beam_contours.png', dpi=130)

# ---- analytical check --------------------------------------------------------
E, nu, L, h, q = 1.0, 0.3, 1.0, 0.1, 1.0e-6
Estar = E/(1-nu**2); I = h**3/12; G = E/(2*(1+nu)); kap = 5/6; A = h
d_eb = q*L**4/(8*Estar*I)
d_sh = q*L**2/(2*kap*G*A)
with open('cantilever_beam_out.csv') as f:
    row = list(csv.DictReader(f))[-1]
d_fe = -float(row['tip_disp_y'])
print(f"Euler-Bernoulli (plane-strain E*): {d_eb:.6e}")
print(f"+ Timoshenko shear term:           {d_sh:.6e}")
print(f"Timoshenko total:                  {d_eb+d_sh:.6e}")
print(f"MOOSE tip deflection (100x10):     {d_fe:.6e}")
print(f"diff vs Timoshenko: {100*(d_fe-(d_eb+d_sh))/(d_eb+d_sh):+.2f}%   vs EB: {100*(d_fe-d_eb)/d_eb:+.2f}%")
print(f"max von Mises (at clamp): {float(row['max_vonmises_stress']):.4e}   elements: {int(float(row['num_elements']))}")
