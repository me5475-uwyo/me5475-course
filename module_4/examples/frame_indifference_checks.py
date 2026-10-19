"""
frame_indifference_checks.py -- Module 4, companion to readings/frame_indifference_primer.md (L20).

Every number in the primer's worked examples, recomputed in 2-D with NumPy only (no training, no MOOSE).
Run:  python frame_indifference_checks.py

Two kinds of rotation:
  passive (change of basis): the SAME tensor, new components:  v' = Q v,  T' = Q T Q^T,  C'_ijkl = Q_ip Q_jq Q_kr Q_ls C_pqrs
  active (superposed rigid rotation of the deformed body):      F* = R F, and each measure transforms as the primer's table says
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def isotropic_plane_strain(E, nu):
    lam, mu, d = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)), np.eye(2)
    return (lam * np.einsum("ij,kl->ijkl", d, d)
            + mu * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)))


def rotate4(C, Q):
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", Q, Q, Q, Q, C)


# --- passive: axes rotated by +30 degrees; Q_ij = e'_i . e_j ---------------------------------
Q = rot(30).T                                   # [[cos, sin], [-sin, cos]]
print("Q =\n", Q, "\nQ Q^T = I:", np.allclose(Q @ Q.T, np.eye(2)))
v = np.array([1.0, 0.0])
print("1. v = (1, 0) -> v' = Q v =", Q @ v, " length", np.linalg.norm(Q @ v))
sig = np.array([[2.0, 1.0], [1.0, 0.0]])
sigp = Q @ sig @ Q.T
print("2. sigma' = Q sigma Q^T =\n", sigp, "\n   trace", np.trace(sig), "->", np.trace(sigp),
      "| det", round(np.linalg.det(sig), 6), "->", round(np.linalg.det(sigp), 6),
      "| principal values", np.linalg.eigvalsh(sig))
eps = np.array([[1e-3, 0.5e-3], [0.5e-3, 0.0]])
epsp = Q @ eps @ Q.T
print("3. eps' (x1e-3) =\n", epsp * 1e3, "\n   engineering shear gamma' = 2 eps'_12 =", round(2 * epsp[0, 1] * 1e3, 4), "e-3")
Ci = isotropic_plane_strain(1.0, 0.3)
print("4. isotropic C (E = 1, nu = 0.3): unchanged by the rotation:", np.allclose(rotate4(Ci, Q), Ci),
      "| C_1111 =", round(Ci[0, 0, 0, 0], 4))
Co = np.zeros((2, 2, 2, 2))
Co[0, 0, 0, 0], Co[1, 1, 1, 1], Co[0, 0, 1, 1], Co[1, 1, 0, 0] = 10.0, 1.0, 0.3, 0.3
for ijkl in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
    Co[ijkl] = 0.5
Cop = rotate4(Co, Q)
print("5. orthotropic (C1111 = 10, C2222 = 1, C1122 = 0.3, C1212 = 0.5) rotated:",
      "C'1111 = %.4f, C'2222 = %.4f, C'1122 = %.4f, C'1212 = %.4f, C'1112 = %.4f"
      % (Cop[0, 0, 0, 0], Cop[1, 1, 1, 1], Cop[0, 0, 1, 1], Cop[0, 1, 0, 1], Cop[0, 0, 0, 1]))
print("   consistent: sigma' = C' : eps' equals Q (C : eps) Q^T:",
      np.allclose(np.einsum("ijkl,kl->ij", Cop, epsp), Q @ np.einsum("ijkl,kl->ij", Co, eps) @ Q.T))

# --- active: superposed rigid rotation R (45 degrees) of a deformed body ------------------------
F = np.array([[1.1, 0.2], [0.0, 0.95]])
R = rot(45)
Fs = R @ F
C, Cs = F.T @ F, Fs.T @ Fs
print("6. F* = R F: C =", C.ravel(), " C* =", Cs.ravel(), " C* = C:", np.allclose(C, Cs),
      "| b* = R b R^T:", np.allclose(Fs @ Fs.T, R @ (F @ F.T) @ R.T))
mu, lam = 1 / (2 * 1.3), 0.3 / (1.3 * 0.4)      # E = 1, nu = 0.3


def neo_hooke(F):
    J, b = np.linalg.det(F), F @ F.T
    psi = 0.5 * mu * (np.trace(F.T @ F) - 2) - mu * np.log(J) + 0.5 * lam * np.log(J) ** 2
    sig = (mu * (b - np.eye(2)) + lam * np.log(J) * np.eye(2)) / J
    P = J * sig @ np.linalg.inv(F).T
    return psi, sig, P, np.linalg.inv(F) @ P


p1, s1, P1, S1 = neo_hooke(F)
p2, s2, P2, S2 = neo_hooke(Fs)
print("7. neo-Hookean: psi(F) = %.6f, psi(RF) = %.6f | sigma* = R sigma R^T: %s | P* = R P: %s | S* = S: %s"
      % (p1, p2, np.allclose(s2, R @ s1 @ R.T), np.allclose(P2, R @ P1), np.allclose(S2, S1)))
bad = lambda F: 0.5 * mu * np.sum((F - np.eye(2)) ** 2)
print("8. not frame-indifferent, psi = mu/2 |F - I|^2: psi(F) = %.6f, psi(RF) = %.6f" % (bad(F), bad(Fs)))
eps45 = 0.5 * (R + R.T) - np.eye(2)
E45 = 0.5 * (R.T @ R - np.eye(2))
R1 = rot(1)
print("9. a pure 45-degree rotation: small strain eps =", eps45.ravel(), "| Green-Lagrange E =", E45.ravel(),
      "| a 1-degree rotation: eps_11 = %.3e" % (0.5 * (R1 + R1.T) - np.eye(2))[0, 0])

# --- hyperelasticity: the stress is the derivative of the stored energy -----------------------
def P_formula(F):                               # P = mu (F - F^-T) + lam ln J F^-T
    FiT = np.linalg.inv(F).T
    return mu * (F - FiT) + lam * np.log(np.linalg.det(F)) * FiT


h, P_fd = 1e-6, np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        dF = np.zeros((2, 2))
        dF[i, j] = h
        P_fd[i, j] = (neo_hooke(F + dF)[0] - neo_hooke(F - dF)[0]) / (2 * h)
print("10. P = d psi / dF: formula", P_formula(F).ravel().round(6), "| central difference", P_fd.ravel().round(6),
      "| max |difference| = %.1e" % np.abs(P_formula(F) - P_fd).max(), "| equals J sigma F^-T:", np.allclose(P_formula(F), P1))
H = np.array([[1.0, 0.5], [0.2, -0.3]])
for d in (1e-2, 1e-4):
    eps = 0.5 * d * (H + H.T)
    hooke = np.einsum("ijkl,kl->ij", Ci, eps)
    print("11. small-strain limit, F = I + %.0e H: |P - C : eps| / |C : eps| = %.1e" % (d, np.linalg.norm(P_formula(np.eye(2) + d * H) - hooke) / np.linalg.norm(hooke)))

# --- material symmetry: Q on the left vs Q on the right (primer section 8 table) --------------
F12, a = np.diag([1.1, 1.0]), np.array([1.0, 0.0])          # stretch along e1; unit fiber a = e1
def row(G):
    C12 = G.T @ G
    return ([round(float(c), 4) for c in np.diag(C12)], round(float(a @ C12 @ a), 4), round(float(np.trace(C12)), 4),
            bool(np.allclose(C12, np.diag(np.diag(C12)))))
print("12. F = diag(1.1, 1), fiber a = e1: (diag C, a.Ca, I1)")
print("    original F           ", row(F12)[:3])
print("    Q F, Q = 30/45/90 deg", [row(rot(t) @ F12)[:3] for t in (30, 45, 90)])
print("    F Q, Q = 90 deg      ", row(F12 @ rot(90))[:3], "| C diagonal:", row(F12 @ rot(90))[3])
print("    F Q, Q = 180 deg     ", row(F12 @ rot(180))[:3])
