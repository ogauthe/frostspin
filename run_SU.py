#! /usr/bin/env python3

import numpy as np
import json
from sys import argv
from time import time
from simple_update import SimpleUpdate2x2
from ctmrg import CTMRG


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
J2 = float(config["J2"])
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
Dmax = int(config["Dmax"])
tau = float(config["tau"])
beta_list = np.array(config["beta_list"], dtype=float)
print(f"\nSimple update parameters: tau = {tau}, Dmax = {Dmax}")
print("Compute environment and observables for beta in", list(beta_list))
pcol = np.array([1, -1], dtype=np.int8)  # U(1) colors for physical spin 1/2
su = SimpleUpdate2x2(d, a, Dmax, h1, h2, tau, colors=pcol, verbosity=0)

# CTMRG parameters
chi_list = np.array(config["chi_list"], dtype=int)
ctm_tol = float(config["ctm_tol"])
ctm_maxiter = int(config["ctm_maxiter"])
print(f"\nCompute CTMRG environment for chi in {list(chi_list)}")
print(f"Converge CTMRG with tol = {ctm_tol} for a max of {ctm_maxiter} iterations")

# misc
save_su_root = str(config["save_su_root"])
save_ctm_root = str(config["save_ctm_root"])
save_rdm_root = str(config["save_rdm_root"])
print("\nSave simple update data in files " + save_su_root + "{beta}.npz")
print("Save CTMRG data in files " + save_ctm_root + "{beta}_chi{chi}.npz")
print("Save reduced density matrices in files" + save_rdm_root + "{beta}_chi{chi}.npz")


########################################################################################
# Computation starts
########################################################################################
t = time()
beta = 0.0
for new_beta in beta_list:
    ####################################################################################
    # Simple update
    ####################################################################################
    print("\n" + "#" * 79)
    print(f"Evolve in imaginary time for beta from {beta}/2 to {new_beta}/2...")
    su.evolve((new_beta - beta) / 2)  # rho is quadratic in mono-layer tensor
    print(f"done with imaginary time evolution, t = {time()-t:.0f}")
    lambdas = su.lambdas
    _, col1, col2, col3, col4, col5, col6, col7, col8 = su.colors
    print(
        "lambdas =",
        lambdas[1],
        lambdas[2],
        lambdas[3],
        lambdas[4],
        lambdas[5],
        lambdas[6],
        lambdas[7],
        lambdas[8],
        sep="\n",
    )
    print("colors =", col1, col2, col3, col4, col5, col6, col7, col8, sep="\n")
    save_su = save_su_root + f"{beta}.npz"
    su.save_to_file(save_su)
    print("Simple update data saved in file", save_su)
    tensors = su.get_ABCD()
    colors = su.get_colors_ABCD()

    if not beta:  # initialize CTMRG from 1st beta value
        ctm_list = []
        for chi in chi_list:
            ctm_list.append(CTMRG(chi, tiling=tiling, tensors=tensors, colors=colors))
    else:  # set tensors to new values
        for ctm in ctm_list:
            ctm.set_tensors(tensors, colors)
    beta = new_beta

    ####################################################################################
    # CTMRG
    ####################################################################################
    for ctm in ctm_list:
        print("\n" + " " * 4 + "#" * 75)
        print(f"    Converge CTMRG for D = {Dmax} and chi = {ctm.chi}...")
        i, rdm_1st_nei = ctm.converge(ctm_tol, maxiter=ctm_maxiter)
        print(f"    done, converged after {i} iterations, t = {time()-t:.0f}")
        save_ctm = save_ctm_root + f"{beta}_chi{ctm.chi}.npz"
        ctm.save_to_file(save_ctm)
        print("    CTMRG data saved in file", save_ctm)

        print("    Compute reduced density matrix cell average for second neighbor...")
        rdm_2nd_nei = ctm.compute_rdm_cell_average_2nd_nei()
        print(f"    done with rdm computation, t = {time()-t:.0f}")
        save_rdm = save_rdm_root + f"{beta}_chi{ctm.chi}.npz"
        np.savez_compressed(save_rdm, rdm_1st_nei=rdm_1st_nei, rdm_2nd_nei=rdm_2nd_nei)
        print("    rdm saved in file", save_rdm)
        energy = (rdm_1st_nei * h1).sum() + (rdm_2nd_nei * h2).sum()
        print(f"    energy = {energy}")


print("\n" + "#" * 79)
print("Program terminated.")
