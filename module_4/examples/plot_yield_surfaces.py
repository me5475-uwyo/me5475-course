"""
plot_yield_surfaces.py -- Module 4, companion to readings/plasticity_and_return_mapping_primer.md (after L23).

Draws the primer's seven figures from the yield functions themselves, and checks the formulas they rely on (below).
Geometry and closed-form algebra only: no training, no MOOSE, no measured data. The parameter values below are
illustrative choices in normalized units (stresses in units of the yield stress or of the cohesion), not material
properties.
Run:  python plot_yield_surfaces.py        (writes the PNGs into ../readings/figures/)

Sign convention: tension positive, as in MOOSE and the L23 deck. The meridian-plane figure uses the pressure
p = -I1/3, positive in compression, as soil mechanics and the Cam-Clay formula do; it says so on its axis.

Every plotted curve is found by root-finding f = 0 along rays and then asserted to satisfy |f| < 1e-9; the 1-D
return mapping asserts f <= 0 at every step and f = 0 after every plastic step; the 3-D J2 radial return (isotropic
plus linear kinematic hardening) is asserted against its own backward-Euler equations on random trial stresses.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "readings" / "figures"
TOL = 1e-9

# Course palette (camml.sty): every curve also differs in line style and carries a direct label.
BROWN, GOLD, RED, GRAY = "#2C1C14", "#8A5E0D", "#A4161A", "#6F6558"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9.5,
                     "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5                    # inches: the printed column width the primer assumes

# --- principal stresses and the pi-plane --------------------------------------------------------------------
# In-plane orthonormal coordinates of the deviator, oriented as on L23's return-mapping slide:
# projected sigma_1 axis at 90 deg, sigma_2 at 210 deg, sigma_3 at 330 deg.
UX = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)     # X = (sigma_3 - sigma_2)/sqrt(2)
UY = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)    # Y = (2 sigma_1 - sigma_2 - sigma_3)/sqrt(6)
ONES = np.ones(3)


def from_pi(X, Y, I1=0.0):
    """Principal stresses with deviator X*UX + Y*UY and first invariant I1."""
    return X * UX + Y * UY + I1 / 3.0 * ONES


def J2(sig):
    s = sig - sig.mean()
    return 0.5 * s @ s


def q_vm(sig):
    return np.sqrt(3.0 * J2(sig))


def f_mises(sig, sy):
    return q_vm(sig) - sy


def f_tresca(sig, sy):
    return sig.max() - sig.min() - sy


def f_dp(sig, A, B):                     # sqrt(J2) + A I1 - B, the L23 deck's form
    return np.sqrt(J2(sig)) + A * sig.sum() - B


def f_mc(sig, c, w):                     # (s1 - s3) + (s1 + s3) sin w - 2 c cos w, with s1 >= s2 >= s3
    s1, s3 = sig.max(), sig.min()
    return (s1 - s3) + (s1 + s3) * np.sin(w) - 2.0 * c * np.cos(w)


def dp_matching_mc(c, w, meridian):
    """Drucker-Prager constants (A, B) whose cone passes through Mohr-Coulomb's compression ('outer')
    or extension ('inner') meridian."""
    k = 3.0 - np.sin(w) if meridian == "compression" else 3.0 + np.sin(w)
    return 2.0 * np.sin(w) / (np.sqrt(3.0) * k), 6.0 * c * np.cos(w) / (np.sqrt(3.0) * k)


def root_on_ray(g, lo, hi, n=200):
    """Bisection for g(r) = 0 with g(lo) < 0 < g(hi)."""
    assert g(lo) < 0.0 < g(hi), "ray does not bracket the surface"
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def pi_section(f, I1=0.0, n=721, rmax=10.0):
    """The surface f = 0 on the pi-plane at fixed I1, as (X, Y) points; every point is asserted on f = 0."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    r = np.array([root_on_ray(lambda rr: f(from_pi(rr * np.cos(a), rr * np.sin(a), I1)), 0.0, rmax) for a in t])
    X, Y = r * np.cos(t), r * np.sin(t)
    for x, y in zip(X, Y):
        assert abs(f(from_pi(x, y, I1))) < TOL
    return X, Y


def draw_pi_axes(ax, length):
    for ang, lab in ((90, r"$\sigma_1$"), (210, r"$\sigma_2$"), (330, r"$\sigma_3$")):
        a = np.deg2rad(ang)
        ax.annotate("", xy=(length * np.cos(a), length * np.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        ax.text(1.1 * length * np.cos(a), 1.1 * length * np.sin(a), lab, color=GRAY, ha="center", va="center")
    ax.plot(0, 0, marker="x", color=BROWN, ms=6, mew=1.4)
    lim = 1.22 * length
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(Path(__file__).resolve().parents[2]))


# --- figure 1: the 1-D rheological model -----------------------------------------------------------------------
def spring(ax, x0, x1, y, coils=6, amp=0.12, **kw):
    xs = np.linspace(x0 + 0.15 * (x1 - x0), x1 - 0.15 * (x1 - x0), 2 * coils + 1)
    ys = y + amp * np.array([0] + [(-1) ** i for i in range(1, 2 * coils)] + [0])
    ax.plot([x0, xs[0]], [y, y], **kw)
    ax.plot(xs, ys, **kw)
    ax.plot([xs[-1], x1], [y, y], **kw)


def figure_model():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    kw = dict(color=BROWN, lw=1.5)
    ax.fill_between([-0.25, 0.0], -0.9, 0.9, color="#E4DDCF", hatch="////", edgecolor=GRAY, lw=0)
    ax.plot([0, 0], [-0.9, 0.9], **kw)
    spring(ax, 0.0, 2.2, 0.0, **kw)
    ax.text(1.1, 0.28, r"spring $E$ (elastic)", ha="center")
    ax.plot([2.2, 2.2], [-0.55, 0.55], **kw)
    # top branch: friction slider
    ax.plot([2.2, 3.0], [0.55, 0.55], **kw)
    ax.add_patch(plt.Rectangle((3.0, 0.42), 0.8, 0.26, fill=False, ec=BROWN, lw=1.5))
    ax.plot([2.85, 4.0], [0.40, 0.40], color=BROWN, lw=2.6)
    ax.plot([3.8, 4.6], [0.55, 0.55], **kw)
    ax.text(3.4, 0.80, r"slider: slips when $|\sigma - X| = \sigma_y + H_{\mathrm{iso}}\,\xi$", ha="center")
    # bottom branch: kinematic hardening spring
    spring(ax, 2.2, 4.6, -0.55, **kw)
    ax.text(3.4, -0.90, r"spring $H_{\mathrm{kin}}$: carries the back stress $X$", ha="center")
    ax.plot([4.6, 4.6], [-0.55, 0.55], **kw)
    ax.plot([4.6, 5.2], [0, 0], **kw)
    ax.annotate("", xy=(5.9, 0), xytext=(5.2, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.text(6.0, 0.0, r"$\sigma$", color=RED, va="center", fontsize=11)
    ax.text(1.1, -0.45, r"strain $\varepsilon^{e}$", ha="center", color=GRAY)
    ax.text(3.4, 0.10, r"strain $\varepsilon^{p}$", ha="center", color=GRAY)
    ax.set_xlim(-0.3, 6.3)
    ax.set_ylim(-1.1, 1.05)
    ax.axis("off")
    save(fig, "plasticity_1d_spring_slider_model.png")


# --- the 1-D return mapping (linear isotropic + linear kinematic hardening) --------------------------------
def return_map_1d(eps_path, E, sy, Hiso, Hkin):
    """Backward-Euler stress update, strain-driven. Returns the stress history and the plastic flags."""
    ep = xi = X = 0.0
    sig_hist, plastic = [], []
    for eps in eps_path:
        s_tr = E * (eps - ep)                          # 1 predict
        eta = s_tr - X
        f_tr = abs(eta) - (sy + Hiso * xi)             # 2 check
        if f_tr <= 0.0:
            s, flag = s_tr, False
        else:                                          # 3 return, in closed form
            dg, sg = f_tr / (E + Hiso + Hkin), np.sign(eta)
            s = s_tr - E * dg * sg
            ep, X, xi = ep + dg * sg, X + Hkin * dg * sg, xi + dg      # 4 update
            flag = True
            assert abs(abs(s - X) - (sy + Hiso * xi)) < TOL            # on the updated surface
        assert abs(s - X) <= sy + Hiso * xi + TOL                      # never outside it
        sig_hist.append(s)
        plastic.append(flag)
    return np.array(sig_hist), np.array(plastic)


def figure_loops():
    E, sy, H = 1.0, 1.0, 0.25                        # normalized: eps / eps_y with eps_y = sy / E
    a = 4.0
    path = np.concatenate([np.linspace(0, a, 401), np.linspace(a, -a, 801)[1:], np.linspace(-a, a, 801)[1:]])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    for ax, (title, Hiso, Hkin, ls) in zip(axes, (("Isotropic hardening", H, 0.0, "-"),
                                                  ("Kinematic hardening", 0.0, H, "--"))):
        sig, plastic = return_map_1d(path, E, sy, Hiso, Hkin)
        ax.plot(path, sig, color=BROWN, ls=ls, lw=1.6)
        smax = sig[400]                               # stress at the first reversal
        first_elastic = 401 + np.argmax(~plastic[401:])
        rev = first_elastic + np.argmax(plastic[first_elastic:])   # first plastic step after the reversal
        expected = -smax if Hkin == 0.0 else smax - 2.0 * sy
        # the step that first yields in reverse lands just past the elastic limit; the limit itself is exact
        assert sig[rev - 1] > expected - TOL and sig[rev] <= expected + TOL
        ax.plot(0.5 * (path[rev - 1] + path[rev]), expected, "o", color=RED, ms=6)
        lab = ("reverse yield at\n" + r"$\sigma = -\sigma_{\max}$" if Hkin == 0.0
               else "reverse yield at\n" + r"$\sigma = \sigma_{\max} - 2\sigma_y$")
        ax.annotate(lab, xy=(0.5 * (path[rev - 1] + path[rev]), expected), xytext=(0.9 if Hkin else 1.3, -2.45),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9), color=RED, fontsize=9.5)
        ax.plot(path[400], smax, "s", color=GOLD, ms=5)
        ax.text(path[400] - 0.2, smax + 0.12, r"$\sigma_{\max}$", color=GOLD, ha="right")
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"strain $\varepsilon/\varepsilon_y$")
    axes[0].set_ylabel(r"stress $\sigma/\sigma_y$")
    save(fig, "plasticity_1d_isotropic_vs_kinematic_loops.png")


# --- figure 3: pi-plane sections ----------------------------------------------------------------------------
def figure_pi_plane():
    sy, c, w = 1.0, 1.0, np.deg2rad(30.0)
    Ao, Bo = dp_matching_mc(c, w, "compression")
    Ai, Bi = dp_matching_mc(c, w, "extension")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4))

    ax = axes[0]
    Xm, Ym = pi_section(lambda s: f_mises(s, sy))
    Xt, Yt = pi_section(lambda s: f_tresca(s, sy))
    for x, y in zip(Xt, Yt):                            # Tresca lies inside (inscribed in) von Mises
        assert f_mises(from_pi(x, y), sy) <= TOL
    draw_pi_axes(ax, 1.25)
    ax.plot(Xm, Ym, color=BROWN, lw=1.7)
    ax.plot(Xt, Yt, color=RED, lw=1.5, ls="--")
    ax.text(0.66, 0.80, "von Mises\n(circle)", color=BROWN, ha="left", fontsize=9.5)
    ax.text(0.0, -0.30, "Tresca\n(hexagon)", color=RED, ha="center", va="center", fontsize=9.5,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_title("(a) Metals: equal uniaxial yield")

    ax = axes[1]
    Xc, Yc = pi_section(lambda s: f_mc(s, c, w))
    Xo, Yo = pi_section(lambda s: f_dp(s, Ao, Bo))
    Xi, Yi = pi_section(lambda s: f_dp(s, Ai, Bi))
    ro, ri = np.hypot(Xo, Yo).mean(), np.hypot(Xi, Yi).mean()
    rc, tt = np.hypot(Xc, Yc), np.rad2deg(np.arctan2(Yc, Xc)) % 360
    assert rc.max() <= ro + TOL                    # the outer cone circumscribes MC (touches its outer corners)
    for ang in (30, 150, 270):                     # outer corners: compression meridians
        assert abs(rc[np.isclose(tt, ang)][0] - ro) < TOL
    for ang in (90, 210, 330):                     # inner corners: extension meridians, on the inner cone
        assert abs(rc[np.isclose(tt, ang)][0] - ri) < TOL
    assert rc.min() < ri - 1e-3                    # ...which is NOT inscribed: MC's edges dip inside it
    draw_pi_axes(ax, 2.05)
    ax.plot(Xc, Yc, color=BROWN, lw=1.7)
    ax.plot(Xo, Yo, color=GOLD, lw=1.4, ls="--")
    ax.plot(Xi, Yi, color=RED, lw=1.4, ls="-.")
    ax.annotate("Drucker–Prager:\nouter corners", xy=(ro * np.cos(np.deg2rad(200)), ro * np.sin(np.deg2rad(200))),
                xytext=(-2.55, -1.72), color=GOLD, ha="left", va="top", fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.8))
    ax.text(0.30, -2.05, "Mohr–Coulomb\n(irregular hexagon)", color=BROWN, ha="left", va="center", fontsize=9.5)
    ax.annotate("Drucker–Prager:\ninner corners", xy=(ri * np.cos(np.deg2rad(60)), ri * np.sin(np.deg2rad(60))),
                xytext=(0.95, 2.42), color=RED, ha="left", va="top", fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))
    ax.set_title(r"(b) Frictional, at $I_1 = 0$")
    save(fig, "yield_surfaces_pi_plane.png")


# --- figure 7: why von Mises is a cylinder, and a circle on the pi-plane ------------------------------------------
def figure_vm_cylinder():
    """(a) A flat cut through the hydrostatic axis: a stress point = its hydrostatic part (I1/3)(1,1,1) along the
    axis + its deviator, perpendicular to it; von Mises is every point at the fixed distance |s| = sqrt(2/3) sy from
    the axis, so the cylinder's wall appears as two parallel lines. Adding a pressure p = -I1/3 (compression
    positive, as in the meridian-plane figure) slides a point along the wall.
    (b) The pi-plane: the same circle, with the uniaxial-tension and pure-shear points on it."""
    sy = 1.0
    r = np.sqrt(2.0 / 3.0) * sy
    e = ONES / np.sqrt(3.0)                                  # unit vector along the hydrostatic axis
    P = np.array([sy, 0.0, 0.0])                             # uniaxial tension at yield
    m_P = P.mean()                                           # mean stress I1/3 (tension positive); pressure = -m_P
    H = m_P * ONES                                           # foot of the perpendicular on the axis
    s = P - H                                                # the deviator
    assert abs(s @ e) < TOL                                  # s is perpendicular to the axis
    assert abs(np.linalg.norm(s) - r) < TOL                  # ...and its length is the cylinder's radius
    assert abs(f_mises(P, sy)) < TOL                         # P is on the surface
    for shift in (-1.0, 0.5, 2.0):                           # sliding along the axis keeps it on the surface
        assert abs(f_mises(P + shift * ONES, sy)) < TOL
    tau = sy / np.sqrt(3.0)                                  # pure shear at yield
    Q = np.array([tau, -tau, 0.0])
    assert abs(f_mises(Q, sy)) < TOL and abs(Q.sum()) < TOL

    fig = plt.figure(figsize=(WIDTH, 3.5))
    ax = fig.add_subplot(1, 2, 1)
    # A flat cut through the hydrostatic axis and the point P: axis coordinate a = x.e, distance d = |x - (x.e)e|.
    aP, dP = P @ e, np.linalg.norm(P - (P @ e) * e)
    assert abs(dP - r) < TOL and abs(aP - np.sqrt(3.0) * m_P) < TOL
    dpress = 0.75                                            # add a pressure (compression positive): P - dpress (1,1,1)
    P2 = P - dpress * ONES
    assert abs((-P2.mean()) - (-P.mean()) - dpress) < TOL    # the pressure p = -I1/3 really grows by dpress
    aP2, dP2 = P2 @ e, np.linalg.norm(P2 - (P2 @ e) * e)
    assert abs(dP2 - r) < TOL and abs(f_mises(P2, sy)) < TOL  # same distance, still on the surface
    xs = (-1.05, 2.25)
    ax.plot(xs, (0, 0), color=GOLD, lw=1.3, ls="--")
    ax.text(xs[1], -0.08, "hydrostatic axis\n" + r"$\sigma_1=\sigma_2=\sigma_3$", color=GOLD, fontsize=8.5,
            ha="right", va="top")
    for sgn in (1, -1):
        ax.plot(xs, (sgn * r, sgn * r), color=BROWN, lw=1.7)
    ax.text(xs[0] + 0.02, -r - 0.07, "von Mises: the cylinder's wall (both lines)", color=BROWN, fontsize=8.5,
            ha="left", va="top")
    ax.plot(0, 0, marker="x", color=BROWN, ms=6, mew=1.4)
    ax.text(-0.10, 0.07, "zero stress", color=GRAY, fontsize=8, ha="right", va="bottom")
    ax.plot((0, aP), (0, dP), color=GRAY, lw=0.8, ls=":")                           # the sigma_1 axis
    ax.text(0.30 * aP + 0.05, 0.30 * dP + 0.10, r"$\sigma_1$ axis", color=GRAY, fontsize=8.5, ha="right",
            va="bottom")
    ax.plot((0, aP), (0, 0), color=GOLD, lw=3.0, solid_capstyle="butt")             # hydrostatic part
    ax.plot((aP, aP), (0, dP), color=RED, lw=2.2)                                   # deviator
    mk = 0.07                                                                        # right-angle mark
    ax.plot((aP - mk, aP - mk, aP), (0, mk, mk), color=RED, lw=0.8)
    ax.plot(aP, dP, "o", color=RED, ms=4.5)
    ax.text(aP, -0.09, "hydrostatic\npart", color=GOLD, fontsize=8.5, ha="right", va="top")
    ax.text(aP + 0.06, 0.40 * dP, r"$\mathbf{s}$, length" + "\n" + r"$\sqrt{2/3}\,\sigma_y$", color=RED,
            fontsize=8.5, ha="left", va="center")
    ax.text(aP + 0.10, dP + 0.07, "uniaxial tension", color=RED, fontsize=8.5, ha="left", va="bottom")
    ax.annotate("", xy=(aP2, dP2), xytext=(aP - 0.06, dP), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0))
    ax.plot(aP2, dP2, "o", mfc="white", mec=RED, ms=4.5)
    ax.text(aP2 - 0.08, dP + 0.07, "add pressure (squeeze):\nstill on the wall", color=GRAY, fontsize=8,
            ha="left", va="bottom")
    ax.set_xlim(-1.1, 2.3); ax.set_ylim(-1.15, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) A cut through the axis")

    ax = fig.add_subplot(1, 2, 2)
    Xm, Ym = pi_section(lambda sg: f_mises(sg, sy))
    assert np.allclose(np.hypot(Xm, Ym), r, atol=1e-9)       # a circle of radius sqrt(2/3) sy
    draw_pi_axes(ax, 1.05)
    ax.plot(Xm, Ym, color=BROWN, lw=1.7)
    XP, YP = s @ UX, s @ UY                                  # uniaxial tension: on the sigma_1 direction
    XQ, YQ = Q @ UX, Q @ UY
    ax.annotate("", xy=(XP, YP), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                shrinkA=0, shrinkB=0))
    ax.plot([XP], [YP], "o", color=RED, ms=4.5)
    ax.text(XP + 0.07, YP * 0.45, r"$|\mathbf{s}| =$" + "\n" + r"$\sqrt{2/3}\,\sigma_y$", color=RED, fontsize=9,
            ha="left", va="center")
    ax.annotate("uniaxial\ntension", xy=(XP, YP), xytext=(-0.62, 1.02), color=RED, fontsize=8.5, ha="right",
                va="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.7, shrinkA=2, shrinkB=4))
    ax.plot([XQ], [YQ], "s", color=BROWN, ms=4.5)
    ax.annotate("pure shear\n" + r"$\tau = \sigma_y/\sqrt{3}$", xy=(XQ, YQ), xytext=(0.98, 1.02), color=BROWN,
                fontsize=8.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.7, shrinkA=2, shrinkB=4))
    ax.text(0.0, -1.22, "every pressure p gives this same circle", color=GRAY, fontsize=8.5, ha="center")
    ax.set_title("(b) Down the axis: the π-plane, a circle")
    save(fig, "yield_vm_cylinder_pi_plane.png")


# --- figure 4: plane stress -------------------------------------------------------------------------------
def figure_plane_stress():
    sy = 1.0
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    pts = {}
    for name, f in (("mises", f_mises), ("tresca", f_tresca)):
        r = np.array([root_on_ray(lambda rr: f(np.array([rr * np.cos(a), rr * np.sin(a), 0.0]), sy), 0.0, 5.0)
                      for a in t])
        pts[name] = (r * np.cos(t), r * np.sin(t))
        for x, y in zip(*pts[name]):
            assert abs(f(np.array([x, y, 0.0]), sy)) < TOL
    xm, ym = pts["mises"]
    assert np.allclose(xm**2 - xm * ym + ym**2, sy**2)         # the ellipse the primer writes
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 3.6))
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.plot(*pts["mises"], color=BROWN, lw=1.7)
    ax.plot(*pts["tresca"], color=RED, lw=1.5, ls="--")
    for (x, y), lab, off in (((1, 0), "uniaxial", (0.08, -0.17)), ((1, 1), "equibiaxial", (0.06, 0.06))):
        ax.plot(x, y, "o", color=BROWN, ms=4)
        ax.text(x + off[0], y + off[1], lab, fontsize=9)
    tv, tt = 1.0 / np.sqrt(3.0), 0.5
    ax.plot([tv, tt], [-tv, -tt], "o", color=GOLD, ms=4)
    ax.annotate("pure shear: Tresca yields at\n" + r"$\tau=\sigma_y/2$, von Mises at $\tau=\sigma_y/\sqrt{3}$",
                xy=(tt, -tt), xytext=(0.05, -1.48), fontsize=9, color=GOLD,
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.9))
    ax.text(-1.35, 1.33, "von Mises\n(ellipse)", color=BROWN, fontsize=9.5, va="top")
    ax.text(-0.95, -0.08, "Tresca\n(hexagon)", color=RED, fontsize=9.5, va="top")
    ax.set_xlabel(r"$\sigma_1/\sigma_y$")
    ax.set_ylabel(r"$\sigma_2/\sigma_y$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.5)
    ax.set_ylim(-1.55, 1.4)
    ax.set_title(r"Plane stress ($\sigma_3 = 0$)")
    save(fig, "yield_surfaces_plane_stress.png")


# --- figure 5: the meridian plane, p = -I1/3 (compression positive) -----------------------------------------
def q_on_meridian(f, p, meridian, qmax=20.0):
    """q on the compression (s1 = s2 > s3) or extension (s1 > s2 = s3) meridian at pressure p."""
    def sig(qq):
        if meridian == "compression":
            return np.array([-p + qq / 3.0, -p + qq / 3.0, -p - 2.0 * qq / 3.0])
        return np.array([-p + 2.0 * qq / 3.0, -p - qq / 3.0, -p - qq / 3.0])
    qq = root_on_ray(lambda x: f(sig(x)), 0.0, qmax)
    assert abs(f(sig(qq))) < TOL and abs(q_vm(sig(qq)) - qq) < TOL and abs(-sig(qq).mean() - p) < TOL
    return qq


def figure_meridian():
    sy, c, w = 2.0, 1.0, np.deg2rad(30.0)
    Ao, Bo = dp_matching_mc(c, w, "compression")
    Ai, Bi = dp_matching_mc(c, w, "extension")
    apex = -c / np.tan(w)                                     # MC and both cones: q = 0 at p = -c cot(w)
    p = np.linspace(apex + 1e-6, 3.0, 200)
    qc = np.array([q_on_meridian(lambda s: f_mc(s, c, w), pp, "compression") for pp in p])
    qe = np.array([q_on_meridian(lambda s: f_mc(s, c, w), pp, "extension") for pp in p])
    assert np.allclose(qc, np.sqrt(3.0) * (Bo + 3.0 * Ao * p))  # DP outer cone = MC compression meridian
    assert np.allclose(qe, np.sqrt(3.0) * (Bi + 3.0 * Ai * p))  # DP inner cone = MC extension meridian
    M, pc = 1.2, 3.0
    pcc = np.linspace(0.0, pc, 400)
    qcc = M * np.sqrt(np.clip(pcc * (pc - pcc), 0.0, None))
    assert np.allclose(qcc**2 + M**2 * pcc * (pcc - pc), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    ax.plot(p, qc, color=BROWN, lw=1.7)
    ax.plot(p, qe, color=RED, lw=1.5, ls="--")
    ax.plot([apex, 3.0], [sy, sy], color=GOLD, lw=1.5, ls=":")
    ax.text(-1.95, 4.45, "compression meridian\n(= Drucker–Prager, outer)", color=BROWN, fontsize=9, va="top")
    ax.text(1.55, 2.25, "extension\nmeridian", color=RED, fontsize=9)
    ax.text(1.15, sy - 0.12, "von Mises: flat", color=GOLD, fontsize=9, va="top")
    ax.plot(apex, 0, "o", color=BROWN, ms=4)
    ax.annotate("apex\n(tension side)", xy=(apex, 0), xytext=(-1.6, 0.9), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8))
    ax.set_title("Mohr–Coulomb meridians")
    ax.set_ylim(0, 4.6)

    ax = axes[1]
    ax.plot(pcc, qcc, color=BROWN, lw=1.7)
    ax.plot([0, pc], [0, M * pc], color=GRAY, lw=1.2, ls="-.")
    ax.text(0.1, 3.75, r"critical state line $q = Mp$", color=GRAY, fontsize=9, ha="left", va="top")
    ax.text(pc / 2, M * pc / 2 + 0.12, "top: critical state", color=BROWN, fontsize=9, ha="center")
    ax.plot(pc, 0, "o", color=BROWN, ms=4)
    ax.text(pc - 0.08, 0.14, r"$p_c$", color=BROWN, ha="right")
    ax.text(pc / 2, 0.55, "elastic inside", color=GOLD, ha="center", fontsize=9)
    ax.set_title("Modified Cam-Clay: an ellipse")
    ax.set_ylim(0, 4.0)
    ax.set_xlim(-0.2, 3.3)
    for ax in axes:
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_xlabel(r"pressure $p = -I_1/3$")
    axes[0].set_ylabel(r"$q = \sqrt{3J_2}$")
    fig.text(0.5, -0.06, "p is positive in compression (soil-mechanics convention); stresses in illustrative units",
             ha="center", fontsize=9, color=GRAY)
    save(fig, "yield_surfaces_meridian_plane.png")


# --- figure 6: isotropic vs kinematic hardening on the pi-plane ---------------------------------------------
def figure_hardening():
    sy = 1.0
    r0 = np.sqrt(2.0 / 3.0) * sy                       # von Mises radius on the pi-plane
    t = np.linspace(0, 2 * np.pi, 361)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1))
    ax = axes[0]
    draw_pi_axes(ax, 1.35)
    r1 = 1.35 * r0
    ax.plot(r0 * np.cos(t), r0 * np.sin(t), color=GRAY, lw=1.3, ls="--")
    ax.plot(r1 * np.cos(t), r1 * np.sin(t), color=BROWN, lw=1.7)
    for a in np.deg2rad([30, 150, 270]):
        ax.annotate("", xy=(r1 * np.cos(a), r1 * np.sin(a)), xytext=(r0 * np.cos(a), r0 * np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0))
    ax.text(0.0, -1.42, "same center, larger radius", ha="center", va="top", fontsize=9.5, color=BROWN)
    ax.set_title("Isotropic: the surface grows")

    ax = axes[1]
    draw_pi_axes(ax, 1.35)
    Xc = np.array([0.32, 0.30])                         # the back stress X, drawn in pi-plane coordinates
    ax.plot(r0 * np.cos(t), r0 * np.sin(t), color=GRAY, lw=1.3, ls="--")
    ax.plot(Xc[0] + r0 * np.cos(t), Xc[1] + r0 * np.sin(t), color=BROWN, lw=1.7)
    ax.annotate("", xy=tuple(Xc), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    ax.text(Xc[0] + 0.05, Xc[1] - 0.16, r"$X$", color=RED, fontsize=11)
    ax.text(0.0, -1.42, "same radius, center moved to $X$", ha="center", va="top", fontsize=9.5, color=BROWN)
    ax.set_title("Kinematic: the surface translates")
    for ax in axes:
        ax.text(-1.62, 1.62, "dashed: before\nsolid: after", fontsize=9, color=GRAY, va="top")
    save(fig, "hardening_isotropic_vs_kinematic_pi_plane.png")


# --- the 3-D J2 radial return, isotropic + linear kinematic hardening -----------------------------------------
def check_j2_return(trials=200, seed=0):
    """The primer's closed form against the backward-Euler equations it claims to solve."""
    rng = np.random.default_rng(seed)
    E, nu, sy, h, hk = 100.0, 0.3, 1.0, 5.0, 3.0
    G, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    I = np.eye(3)
    dev = lambda A: A - np.trace(A) / 3.0 * I
    qn = lambda A: np.sqrt(1.5 * np.tensordot(A, A))
    C = lambda A: K * np.trace(A) * I + 2.0 * G * dev(A)                # isotropic elasticity
    for _ in range(trials):
        S = rng.normal(size=(3, 3)); S = 0.5 * (S + S.T) * 3.0           # trial stress
        X0 = dev(0.5 * (lambda B: B + B.T)(rng.normal(size=(3, 3))) * 0.3)  # deviatoric back stress
        xi0 = abs(rng.normal()) * 0.05
        eta_tr = dev(S) - X0
        f_tr = qn(eta_tr) - (sy + h * xi0)
        if f_tr <= 0.0:
            continue
        dg = f_tr / (3.0 * G + h + hk)
        n = 1.5 * eta_tr / qn(eta_tr)                                   # the trial normal
        sig = S - dg * C(n)
        X1, xi1 = X0 + (2.0 / 3.0) * hk * dg * n, xi0 + dg
        eta1 = dev(sig) - X1
        n1 = 1.5 * eta1 / qn(eta1)                                      # the END-of-step normal
        assert abs(qn(eta1) - (sy + h * xi1)) < TOL                      # on the updated surface
        assert np.allclose(n1, n, atol=1e-12)                           # radial: end normal = trial normal
        assert np.allclose(sig, S - dg * C(n1), atol=1e-12)             # backward Euler, n at step end
        assert abs(np.trace(sig) - np.trace(S)) < TOL                   # the pressure is untouched
        assert abs(np.sqrt(2.0 / 3.0 * np.tensordot(dg * n1, dg * n1)) - dg) < TOL   # dgamma = eff. strain step
    print("J2 radial return: all closed-form checks passed")


if __name__ == "__main__":
    check_j2_return()
    figure_model()
    figure_loops()
    figure_pi_plane()
    figure_plane_stress()
    figure_meridian()
    figure_hardening()
    figure_vm_cylinder()
