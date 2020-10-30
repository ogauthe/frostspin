#! /usr/bin/env python3

import sys
import json
import time

import numpy as np

from simple_update import SimpleUpdate2x2

# from ctmrg import CTMRG
from ctmrg import CTMRG_U1


########################################################################################
# Initialization
########################################################################################
print("#" * 79)
print("Finite temperature simple update and CTMRG for J1-J2 Heisenberg model")
if len(sys.argv) < 2:
    config_file = "input_sample/input_sample_run_full_simulation.json"
    print("No input file given, use", config_file)
else:
    config_file = sys.argv[1]
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
compute_capacity = bool(config["compute_capacity"])
beta_list = np.array(config["beta_list"], dtype=float)
print(f"\nSimple update parameters: tau = {tau}, Dmax = {Dmax}")
print("Compute environment and observables for beta in", list(beta_list))
if compute_capacity:
    print(
        "Compute thermal capacity: for each beta, also compute energy for beta + 4*tau"
    )
    beta_list = np.sort(list(beta_list + 4 * tau) + list(beta_list))
print("Actual beta list is now", list(beta_list))

# initilialize SU
if config["su_restart_file"] is not None:
    su_restart = str(config["su_restart_file"])
    print("Restart simple update from file", su_restart)
    su = SimpleUpdate2x2(d, a, Dmax, tau, file=su_restart)
    if ((su.h1 - h1) ** 2).sum() ** 0.5 > 1e-15 or (
        (su.h2 - h2) ** 2
    ).sum() ** 0.5 > 1e-15:
        raise ValueError("Saved hamiltonians differ from input")
else:
    pcol = np.array([1, -1], dtype=np.int8)  # U(1) colors for physical spin 1/2
    su = SimpleUpdate2x2(d, a, Dmax, tau, h1, h2, colors=pcol, verbosity=0)
print(f"Start from beta = {su.beta}")

# CTMRG parameters
ctm_list = []
chi_list = np.array(config["chi_list"], dtype=int)
ctm_tol = float(config["ctm_tol"])
ctm_warmup = int(config["ctm_warmup"])
ctm_maxiter = int(config["ctm_maxiter"])
print(f"\nCompute CTMRG environment for chi in {list(chi_list)}")
print(f"Converge CTMRG for at least {ctm_warmup} and at most {ctm_maxiter} iterations")
print(f"Set converge tolerance to {ctm_tol}")

# save parameters (do not mix internal CTM/SU stuff and simulation parameters)
su_params = {"tau": tau, "J2": J2, "Dmax": Dmax}
ctm_params = {
    "ctm_tol": ctm_tol,
    "ctm_warmup": ctm_warmup,
    "ctm_maxiter": ctm_maxiter,
    **su_params,
}
save_su_root = str(config["save_su_root"])
save_ctm_root = str(config["save_ctm_root"])
save_rdm_root = str(config["save_rdm_root"])
print("\nSave simple update data in files " + save_su_root + "{beta}.npz")
print("Save CTMRG data in files " + save_ctm_root + "{beta}_chi{chi}.npz")
print("Save reduced density matrices in files " + save_rdm_root + "{beta}_chi{chi}.npz")

# misc
energies = []


########################################################################################
# Computation starts
########################################################################################

for beta in beta_list:
    ####################################################################################
    # Simple update
    ####################################################################################
    print("\n" + "#" * 79)
    print(f"Evolve in imaginary time for beta from {su.beta} to {beta}...")
    beta_evolve = beta - su.beta
    t = time.time()
    su.evolve(beta_evolve)
    print(f"done with imaginary time evolution, t = {time.time()-t:.0f}")
    print("lambdas =", *su.lambdas[1:], sep="\n")
    print("colors =", *su.colors[1:], sep="\n")
    if beta_evolve > 5 * tau:  # do not save again SU after just 1 update
        save_su = save_su_root + f"{su.beta}.npz"
        data_su = su.save_to_file()
        np.savez_compressed(save_su, beta=su.beta, **su_params, **data_su)
        print("Simple update data saved in file", save_su)
    tensors = su.get_ABCD()
    colors = su.get_colors_ABCD()

    if not ctm_list:  # initialize CTMRG from 1st beta value
        for chi in chi_list:
            ctm_list.append(
                # CTMRG(chi, tiling=tiling, tensors=tensors)
                CTMRG_U1(chi, tiling, tensors=tensors, colors=colors)
            )
        rdm_params = {"cell_coords": ctm_list[-1].neq_coords.copy(), **ctm_params}
    else:  # set tensors to new values
        for ctm in ctm_list:
            # ctm.set_tensors(tensors)
            ctm.set_tensors(tensors, colors)

    ####################################################################################
    # CTMRG
    ####################################################################################
    eps_beta = []
    for ctm in ctm_list:
        print("\n" + " " * 4 + "#" * 75)
        print(
            f"    Converge CTMRG at beta = {su.beta} for D = {Dmax} and",
            f"chi = {ctm.chi}...",
        )
        try:
            j, rdm2x1_cell, rdm1x2_cell = ctm.converge(
                ctm_tol, warmup=ctm_warmup, maxiter=ctm_maxiter
            )
            print(f"done, converged after {j} iterations, t = {time.time()-t:.0f}")
        except RuntimeError as err:
            msg, (j, rdm2x1_cell, rdm1x2_cell) = err.args
            print(
                f"\n*** RuntimeError after {j} iterations, t = {time.time()-t:.0f} ***"
            )
            print(f"Error: '{msg}'. Save CTM and move on.\n")
        save_ctm = save_ctm_root + f"{su.beta}_chi{ctm.chi}.npz"
        data_ctm = ctm.save_to_file()
        np.savez_compressed(
            save_ctm, beta=su.beta, chi=ctm.chi, **ctm_params, **data_ctm
        )
        print("    CTMRG data saved in file", save_ctm)

        print("    Compute reduced density matrix for all second neighbor bonds...")
        rdm2x1_cell = np.array(rdm2x1_cell)
        rdm1x2_cell = np.array(rdm1x2_cell)
        save_rdm = save_rdm_root + f"{su.beta}_chi{ctm.chi}.npz"
        t = time.time()
        rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
        rdm_dr_cell = np.array(rdm_dr_cell)
        rdm_ur_cell = np.array(rdm_ur_cell)
        print(f"    done with rdm computation, t = {time.time()-t:.0f}")
        np.savez_compressed(
            save_rdm,
            beta=su.beta,
            chi=ctm.chi,
            rdm1x2_cell=rdm1x2_cell,
            rdm2x1_cell=rdm2x1_cell,
            rdm_dr_cell=rdm_dr_cell,
            rdm_ur_cell=rdm_ur_cell,
            **rdm_params,
        )
        print("    rdm saved to file", save_rdm)
        eps_beta.append(
            ((rdm1x2_cell + rdm2x1_cell) * h1).sum() / 4
            + ((rdm_dr_cell + rdm_ur_cell) * h2).sum() / 4
        )
        print(f"    energy = {eps_beta[-1]}")
    energies += [eps_beta]


print("\n" + "#" * 79)
print("Computations finished.\n\nEnergies:")
print(" beta           ", *(f"chi = {chi}" + " " * 15 for chi in chi_list))
for beta, eps_beta in zip(beta_list, energies):
    print(f"{beta:.4f}  ", *(f"  {e:.17f}" for e in eps_beta))
