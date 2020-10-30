#! /usr/bin/env python3

import sys
import json
import time

import numpy as np

from simple_update import SimpleUpdate2x2
from ctmrg import CTMRG_U1


########################################################################################
# Initialization
########################################################################################
print("#" * 79)
print("CTMRG algorithm for J1-J2 Heisenberg model")
if len(sys.argv) < 2:
    config_file = "input_sample/input_sample_run_CTMRG.json"
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

# initilialize SU
su_restart = str(config["su_restart_file"])
print("Restart simple update from file", su_restart)
su = SimpleUpdate2x2(d, a, file=su_restart)
if ((su.h1 - h1) ** 2).sum() ** 0.5 > 1e-15 or ((su.h2 - h2) ** 2).sum() ** 0.5 > 1e-15:
    raise ValueError("Saved hamiltonians differ from input")

Dmax = su.Dmax
tau = su.tau
beta0 = su.beta
if config["capacity_dbeta"] is None:
    dbeta = 4 * su.tau  # one SU iteration
else:
    dbeta = float(config["dbeta"])
print(f"\nSimple update parameters: tau = {tau}, Dmax = {Dmax}, beta = {beta0}")
print(f"Compute environment and observables for beta = {su.beta}")
print(f"Once done evolve simple update for imaginary time dbeta = {dbeta}")
print("then recompute environment to evaluate thermal capacity.")

# CTMRG parameters
chi_list = np.array(config["chi_list"], dtype=int)
ctm_tol = float(config["ctm_tol"])
ctm_warmup = int(config["ctm_warmup"])
ctm_maxiter = int(config["ctm_maxiter"])
chi = chi_list[0]
print(f"\nCompute CTMRG environment for chi in {list(chi_list)}")
print(f"Converge CTMRG for at least {ctm_warmup} and at most {ctm_maxiter} iterations")
print(f"Set converge tolerance to {ctm_tol}")

# save parameters (do not mix internal CTM/SU stuff and simulation parameters)
ctm_params = {
    "tau": tau,
    "J2": J2,
    "Dmax": Dmax,
    "ctm_tol": ctm_tol,
    "ctm_warmup": ctm_warmup,
    "ctm_maxiter": ctm_maxiter,
}
save_ctm_root = str(config["save_ctm_root"])
print("Save CTMRG data in files " + save_ctm_root + "{beta}_chi{chi}.npz")

# observables
compute_rdm_2nd_nei = bool(config["compute_rdm_2nd_nei"])
if compute_rdm_2nd_nei:
    print(
        "Compute next nearest neighbor density matrix and observables once",
        "environment is converged",
    )
    print("Compute capacity as -(beta+dbeta/2)**2 * (E(beta) - E(beta+dbeta)) / dbeta")
    save_rdm_root = str(config["save_rdm_root"])
    print(
        "Save reduced density matrices in files "
        + save_rdm_root
        + "{beta}_chi{chi}.npz"
    )
    energies0 = []
    order_parameter0 = []
    energies1 = []
    order_parameter1 = []
    capacities = []
else:
    print("Do not compute next nearest neighbor density matrix")


# CTMRG initialization
print()
if config["ctm_restart_file"] is not None:
    ctm_restart = str(config["ctm_restart_file"])
    print("Restart environment tensors from file", ctm_restart)
    ctm = CTMRG_U1(chi, tiling, file=ctm_restart, verbosity=1)
    ctm.set_tensors(su.get_ABCD(), su.get_colors_ABCD())
else:
    ctm = CTMRG_U1(chi, tiling, su.get_ABCD(), su.get_colors_ABCD(), verbosity=1)


if compute_rdm_2nd_nei:
    rdm_params = {"cell_coords": ctm.neq_coords.copy(), **ctm_params}  # need ctm

########################################################################################
# Computation for beta = beta0
########################################################################################
print("", "#" * 75, f"beta = {beta0}", sep="\n")
print("lambdas =", *su.lambdas[1:], sep="\n")
print("colors =", *su.colors[1:], sep="\n")
for chi in chi_list:
    print("", "#" * 75, f"Set chi to {chi}", sep="\n")
    ctm.chi = chi  # restart from last envrionment, just change chi
    print(f"Converge CTMRG at beta = {beta0} for D = {Dmax} and chi = {ctm.chi}...")
    t = time.time()
    try:
        j, rdm2x1_cell, rdm1x2_cell = ctm.converge(
            ctm_tol, warmup=ctm_warmup, maxiter=ctm_maxiter
        )
        print(f"done, converged after {j} iterations, t = {time.time()-t:.0f}")
    except RuntimeError as err:
        msg, (j, rdm2x1_cell, rdm1x2_cell) = err.args
        print(f"\n*** RuntimeError after {j} iterations, t = {time.time()-t:.0f} ***")
        print(f"Error: '{msg}'. Save CTM and move on.\n")
    save_ctm = save_ctm_root + f"{beta0}_chi{ctm.chi}.npz"
    data_ctm = ctm.save_to_file()
    np.savez_compressed(save_ctm, beta=beta0, chi=ctm.chi, **ctm_params, **data_ctm)
    del data_ctm
    print("CTMRG data saved in file", save_ctm)

    if compute_rdm_2nd_nei:
        rdm2x1_cell = np.array(rdm2x1_cell)
        rdm1x2_cell = np.array(rdm1x2_cell)
        print("\nCompute reduced density matrix for all second neighbor bonds...")
        t = time.time()
        rdm_dr_cell, rdm_ur_cell = np.array(ctm.compute_rdm_2nd_neighbor_cell())
        print(f"done with rdm computation, t = {time.time()-t:.0f}")
        save_rdm = save_rdm_root + f"{beta0}_chi{ctm.chi}.npz"
        np.savez_compressed(
            save_rdm,
            beta=beta0,
            chi=ctm.chi,
            rdm1x2_cell=rdm1x2_cell,
            rdm2x1_cell=rdm2x1_cell,
            rdm_dr_cell=rdm_dr_cell,
            rdm_ur_cell=rdm_ur_cell,
            **rdm_params,
        )
        print("rdm saved to file", save_rdm)
        energy = ((rdm1x2_cell + rdm2x1_cell) * h1).sum() / 4 + (
            (rdm_dr_cell + rdm_ur_cell) * h2
        ).sum() / 4
        op = ((rdm1x2_cell - rdm2x1_cell) * h1).sum()
        print(f"energy = {energy}")
        print(f"Ising order parameter = {op}")
        energies0.append(energy)
        order_parameter0.append(op)

print("", "#" * 79, f"Done for beta = {beta0}", sep="\n")
if compute_rdm_2nd_nei:
    print(f"Energy and order parameter for D = {Dmax}, tau = {tau}, beta = {beta0}:")
    for chi, eps, op in zip(chi_list, energies0, order_parameter0):
        print(f"chi = {chi}: energy = {eps:.17f}, order_parameter = {op:.6f}")

########################################################################################
# Imaginary time evolution for minimal step
########################################################################################
print("\n" + "#" * 79)
print(f"Compute thermal capacity: evolve simple update for imaginary time {dbeta}")
t = time.time()
su.evolve(dbeta)
print(f"done with imaginary time evolution, t = {time.time()-t:.0f}")
beta1 = su.beta
print(f"beta is now {beta1}")
print("lambdas =", *su.lambdas[1:], sep="\n")
print("colors =", *su.colors[1:], sep="\n")

########################################################################################
# Computation for beta = beta1
########################################################################################
# restart CTMRG from environment with same chi at previous temperature value beta0
for i, chi in enumerate(chi_list):
    print("\n" + "#" * 75)
    save_ctm = save_ctm_root + f"{beta0}_chi{chi}.npz"
    print("restart environment from file", save_ctm)
    ctm = CTMRG_U1(chi, tiling, file=save_ctm, verbosity=1)
    ctm.set_tensors(su.get_ABCD(), su.get_colors_ABCD())
    print(f"Converge CTMRG at beta = {beta1} for D = {Dmax} and chi = {ctm.chi}...")
    t = time.time()
    try:
        j, rdm2x1_cell, rdm1x2_cell = ctm.converge(
            ctm_tol, warmup=ctm_warmup, maxiter=ctm_maxiter
        )
        print(f"done, converged after {j} iterations, t = {time.time()-t:.0f}")
    except RuntimeError as err:
        msg, (j, rdm2x1_cell, rdm1x2_cell) = err.args
        print(f"\n*** RuntimeError after {j} iterations, t = {time.time()-t:.0f} ***")
        print(f"Error: '{msg}'. Save CTM and move on.\n")
    save_ctm = save_ctm_root + f"{beta1}_chi{ctm.chi}.npz"
    data_ctm = ctm.save_to_file()
    np.savez_compressed(save_ctm, beta=beta1, chi=ctm.chi, **ctm_params, **data_ctm)
    del data_ctm
    print("CTMRG data saved in file", save_ctm)

    if compute_rdm_2nd_nei:
        rdm2x1_cell = np.array(rdm2x1_cell)
        rdm1x2_cell = np.array(rdm1x2_cell)
        print("Compute reduced density matrix for all second neighbor bonds...")
        t = time.time()
        rdm_dr_cell, rdm_ur_cell = np.array(ctm.compute_rdm_2nd_neighbor_cell())
        print(f"done with rdm computation, t = {time.time()-t:.0f}")
        save_rdm = save_rdm_root + f"{beta1}_chi{ctm.chi}.npz"
        np.savez_compressed(
            save_rdm,
            beta=beta1,
            chi=ctm.chi,
            rdm1x2_cell=rdm1x2_cell,
            rdm2x1_cell=rdm2x1_cell,
            rdm_dr_cell=rdm_dr_cell,
            rdm_ur_cell=rdm_ur_cell,
            **rdm_params,
        )
        print("rdm saved to file", save_rdm)
        energy = ((rdm1x2_cell + rdm2x1_cell) * h1).sum() / 4 + (
            (rdm_dr_cell + rdm_ur_cell) * h2
        ).sum() / 4
        op = ((rdm1x2_cell - rdm2x1_cell) * h1).sum()
        energies1.append(energy)
        order_parameter1.append(op)
        cap = -(((beta0 + beta1) / 2) ** 2) * (energy - energies0[i]) / (beta1 - beta0)
        capacities.append(cap)
        print(f"energy = {energy}")
        print(f"Ising order parameter = {op}")
        print(f"thermal capacity = {cap}")


print("\n" + "#" * 79)
print("Computations finished.")
if compute_rdm_2nd_nei:
    print(
        f"Energy and order parameter and capacity for D = {Dmax}, tau = {tau},",
        f"beta = {beta1}:",
    )
    print(
        "chi"
        + " " * 11
        + "energy"
        + " " * 12
        + "order parameter"
        + " " * 8
        + "capacity"
    )
    for chi, eps, op, cap in zip(chi_list, energies1, order_parameter1, capacities):
        print(f"{chi}    {eps:.17f}       {op: .6f}           {cap: .6f}")
