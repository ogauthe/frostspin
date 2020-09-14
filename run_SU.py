#! /usr/bin/env python3

import numpy as np
from time import time
from simple_update import SimpleUpdate2x2
from test_tools import SdS_22, SdS_22b

# import scipy.linalg as lg
from ctmrg import CTMRG

# from toolsU1 import checkU1, combine_colors


tiling = "AB\nCD"

d = 2
a = 2
chi = 40
ctm_iter = 200
cut = 6
Ds = (1,) * 8
print(f"run simple update with d = {d}, a = {a}, Ds = {Ds}")

J2 = 0.0
beta = 1.0
tau = 1e-4
su_iter = int(beta / 2 / tau)  # rho is quadratic in psi

pcol = np.array([1, -1], dtype=np.int8)
vcol = np.zeros(1, dtype=np.int8)
colors = [pcol, -pcol, vcol, vcol, vcol, vcol, vcol, vcol, vcol, vcol]
A = np.eye(d).reshape(d, a, 1, 1, 1, 1)
B = A.copy()
C = B.copy()
D = A.copy()

h1 = SdS_22b
h2 = J2 * SdS_22
# colors A and D = [[1,-1], [1,-1], [2, 0, -2], [2, 0, -2], [2, 0, -2], [2, 0, -2]]
# B and C have opposed colors due to pi-rotation.

print("#" * 79)
print(f"Run simple update for spin 1/2 Heisenberg with J1=1, J2={J2}")
print(f"run with tau = {tau} up to beta = {beta}")
su = SimpleUpdate2x2(
    d, a, Ds, cut, h1, h2, tau, tensors=(A, B, C, D), colors=colors, verbosity=0
)

t = time()
for i in range(su_iter // 2):  # 2nd order Trotter
    # for i in range(2):
    su.update()
print(f"\ndone with SU, t={time()-t:.1f}")

lambdas = su.lambdas
print("lambdas =")
print(lambdas[1])
print(lambdas[2])
print(lambdas[3])
print(lambdas[4])
print(lambdas[5])
print(lambdas[6])
print(lambdas[7])
print(lambdas[8])
print()

A, B, C, D = su.get_ABCD()
(pcol, acol), col1, col2, col3, col4, col5, col6, col7, col8 = su.colors
colorsA = [pcol, acol, col1, col2, col3, col4]
colorsB = [-pcol, -acol, -col5, -col4, -col6, -col2]
colorsC = [-pcol, -acol, -col3, -col7, -col1, -col8]
colorsD = [pcol, acol, col6, col8, col5, col7]


ctm = CTMRG(
    chi,
    tensors=(A, B, C, D),
    tiling=tiling,
    colors=(colorsA, colorsB, colorsC, colorsD),
    verbosity=0,
)


def compute_energy(ctm, h1, h2):
    # Tr(AH) = (A*H.T).sum() and H is exactly symmetric
    rho1 = ctm.compute_rdm2x1(-1, 0)
    rho2 = ctm.compute_rdm1x2(-1, -1)
    rho3 = ctm.compute_rdm2x1(-1, -1)
    rho4 = ctm.compute_rdm1x2(0, -1)
    rho5 = ctm.compute_rdm2x1(0, 0)
    rho6 = ctm.compute_rdm2x1(0, -1)
    rho7 = ctm.compute_rdm1x2(-1, 0)
    rho8 = ctm.compute_rdm1x2(0, 0)
    eps1 = ((rho1 + rho2 + rho3 + rho4 + rho5 + rho6 + rho7 + rho8) * h1).sum()

    rho9 = ctm.compute_rdm_diag_dr(0, 0)
    rho10 = ctm.compute_rdm_diag_ur(-1, 0)
    rho11 = ctm.compute_rdm_diag_dr(-1, -1)
    rho12 = ctm.compute_rdm_diag_ur(0, -1)
    rho13 = ctm.compute_rdm_diag_dr(-1, 0)
    rho14 = ctm.compute_rdm_diag_ur(0, 0)
    rho15 = ctm.compute_rdm_diag_dr(0, -1)
    rho16 = ctm.compute_rdm_diag_ur(-1, -1)
    eps2 = ((rho9 + rho10 + rho11 + rho12 + rho13 + rho14 + rho15 + rho16) * h2).sum()

    energy = (eps1 + eps2) / 4
    return energy


print(f"Converge CTMRG with chi = {chi} and niter = {ctm_iter}")
t = time()
for i in range(ctm_iter):
    print(i, compute_energy(ctm, h1, h2))
    ctm.iterate()

print(f"\ndone with CTM iteration, t={time()-t:.1f}")
energy = compute_energy(ctm, h1, h2)
print("energy =", energy)

save = f"data_ctm_SU_ABCD_J2_{J2}_beta{beta}_tau{tau}_chi{chi}.npz"
ctm.save_to_file(save)
print("CTMRG data saved in file", save)
