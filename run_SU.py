#! /usr/bin/env python3

import numpy as np
import json
from sys import argv
from time import time
from simple_update import SimpleUpdate2x2
from ctmrg import CTMRG


def compute_energy(ctm, h1, h2):
    """
    Compute mean per site site energy on a 2x2 plaquette with first neighbor Hamiltonian
    h1 and second neighbor Hamiltonian h2 from CTMRG.
    """
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


########################################################################################
# Initialization
########################################################################################
print("#" * 79)
print("Finite temperature simple update for J1-J2 Heisenberg model")
if len(argv) < 2:
    config_file = "input_sample/input_sample_run_SU.json"
    print("No input file given, use", config_file)
else:
    config_file = argv[1]
    print("Take input parameters from file", config_file)

with open(config_file) as f:
    config = json.load(f)

# physical model parameters
tiling = "AB\nCD"
d = 2
a = 2
J2 = config["J2"]
SdS_22b = np.array(
    [
        [-0.25, 0.0, 0.0, -0.5],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [-0.5, 0.0, 0.0, -0.25],
    ]
)
SdS_22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)
h1 = SdS_22b
h2 = J2 * SdS_22
print(f"Physical model: d = {d}, a = {a}, J1 = 1, J2 = {J2}")

# simple update parameters
beta = config["beta"]
tau = config["tau"]
Dmax = config["Dmax"]
print(f"Simple update parameters: beta = {beta}, tau = {tau}, Dmax = {Dmax}")

# CTMRG parameters
chi = config["chi"]
ctm_iter = config["ctm_iter"]
print(f"CTMRG parameters: chi = {chi}, ctm_iter = {ctm_iter}")

# misc
print("Save data in file", config["save_data"])

# Tensor nitialization at beta = 0
A = np.eye(d).reshape(d, a, 1, 1, 1, 1)
B = A.copy()
C = B.copy()
D = A.copy()

# colors initialization
pcol = np.array([1, -1], dtype=np.int8)
vcol = np.array([0], dtype=np.int8)
colors = [pcol, -pcol, vcol, vcol, vcol, vcol, vcol, vcol, vcol, vcol]


########################################################################################
# Simple update
########################################################################################
print("\n" + "#" * 79)
print(f"Start simple update with tau = {tau} up to beta = {beta}")
su = SimpleUpdate2x2(
    d, a, Dmax, h1, h2, tau, tensors=(A, B, C, D), colors=colors, verbosity=10
)

t = time()
su.evolve(beta)
print(f"\nDone with SU, t = {time()-t:.0f}")

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

data_su = su.save_to_file()
np.savez_compressed(config["save_data"], **data_su)
print("Simple update data saved in file", config["save_data"])

A, B, C, D = su.get_ABCD()
(pcol, acol), col1, col2, col3, col4, col5, col6, col7, col8 = su.colors
colorsA = [pcol, acol, col1, col2, col3, col4]
colorsB = [-pcol, -acol, -col5, -col4, -col6, -col2]
colorsC = [-pcol, -acol, -col3, -col7, -col1, -col8]
colorsD = [pcol, acol, col6, col8, col5, col7]


########################################################################################
# CTMRG
########################################################################################
print("\n" + "#" * 79)
print("Compute observables using CTMRG")
ctm = CTMRG(
    chi,
    tensors=(A, B, C, D),
    tiling=tiling,
    colors=(colorsA, colorsB, colorsC, colorsD),
    verbosity=0,
)

print(f"energy before iteration = {compute_energy(ctm, h1, h2)}")
print(f"Converge CTMRG with chi = {chi} and niter = {ctm_iter}")
t = time()
for i in range(ctm_iter):
    ctm.iterate()
    energy = compute_energy(ctm, h1, h2)
    print(f"i = {i+1}, t = {time()-t:.0f}, energy = {energy}")

print(f"\ndone with CTM iteration, t={time()-t:.0f}")
print("energy =", energy)

data_ctm = ctm.save_to_file()
np.savez_compressed(config["save_data"], **data_su, **data_ctm)
print("Simple update and CTMRG data saved in file", config["save_data"])

print("\n" + "#" * 79)
print("done")
