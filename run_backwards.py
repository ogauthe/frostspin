#! /usr/bin/env python3
"""
This executable is very similar to run_SU.py, except that it first execute the whole
simple update algorithm and then converge CTMRG starting from the *lowest* temperature.
At each new temperature, it updates the tensors but keeps the environment, so that the
temperature always raises.

Thermal capacity is always computed, even for beta steps larger than 4*tau. Note that
the capacity value between beta and beta + 4*tau will be stored in beta obs file, not in
beta + 4*tau file.

The input file format is the same as run_SU.py, even if some fields are not read.
"""

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
print("Finite temperature PEPS for J1-J2 Heisenberg model")
print("Compute CTMRG by *raising* temperature gradually")
if len(sys.argv) < 2:
    config_file = "input_sample/input_sample_run_SU.json"
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
beta_setpoint = np.array(config["beta_list"], dtype=float)
dbeta = 4 * tau  # one SU iteration
beta_list = np.round(beta_setpoint / dbeta) * dbeta
print(f"\nSimple update parameters: tau = {tau}, Dmax = {Dmax}")
print("Beta is set to reach", list(beta_setpoint))
print("With given tau, reached beta will be:", beta_list)
print(
    "Evolve in imaginary time and save simple update tensors for beta in",
    list(beta_list),
)

chi_list = np.array(config["chi_list"], dtype=int)
measure_capacity = bool(config["measure_capacity"])
if measure_capacity:
    print("Compute thermal capacity: for each beta, add beta + dbeta to beta_list,")
    print(f"with dbeta = {dbeta}")
    beta_list = np.round(np.sort(list(beta_list + dbeta) + list(beta_list)), 10)
    print("Actual beta list is now", list(beta_list))

# initilialize SU
if config["su_restart_file"] is not None:
    su_restart = str(config["su_restart_file"])
    print("Restart simple update from file", su_restart)
    su = SimpleUpdate2x2(d, a, file=su_restart)
    if ((su.h1 - h1) ** 2).sum() ** 0.5 > 1e-15 or (
        (su.h2 - h2) ** 2
    ).sum() ** 0.5 > 1e-15:
        raise ValueError("Saved hamiltonians differ from input")
    # overwrite SU parameters with those from input
    su.Dmax = Dmax
    su.tau = tau
    print(f"Start from beta = {su.beta}")
else:
    print("Start simple update from scratch at beta = 0")
    pcol = np.array([1, -1], dtype=np.int8)  # U(1) colors for physical spin 1/2
    su = SimpleUpdate2x2(d, a, Dmax, tau, h1, h2, colors=pcol, verbosity=0)

# save parameters (do not mix internal CTM/SU stuff and simulation parameters)
su_params = {"tau": tau, "J2": J2, "Dmax": Dmax}
save_su_root = str(config["save_su_root"])
print("Save simple update data in files " + save_su_root + "{beta}.npz")

ctm_tol = float(config["ctm_tol"])
ctm_warmup = int(config["ctm_warmup"])
ctm_maxiter = int(config["ctm_maxiter"])
print(f"\nCompute CTMRG environment for chi in {list(chi_list)}")
if config["ctm_restart_file"] is not None:
    ctm_restart = str(config["ctm_restart_file"])
    print("Restart environment from file", ctm_restart)
else:
    ctm_restart = None
    print("ctm_restart_file not provided, start from scratch")
print(
    f"Converge CTMRG for at least {ctm_warmup} and at most {ctm_maxiter}", "iterations"
)
print(f"Set converge tolerance to {ctm_tol}")

ctm_params = {
    "ctm_tol": ctm_tol,
    "ctm_warmup": ctm_warmup,
    "ctm_maxiter": ctm_maxiter,
    **su_params,
}
save_ctm_root = str(config["save_ctm_root"])
print("Save CTMRG data in files " + save_ctm_root + "{beta}_chi{chi}.npz\n")

# observables
obs_str = "chi   ising"
save_obs_root = str(config["save_obs_root"])
print("For all beta, save observables in files " + save_obs_root + "{beta}.npz\n")

measure_xi = bool(config["measure_xi"])
if measure_xi:
    print("Measure correlation length in both directions after convergence")
    obs_str += "   xi_h0   xi_h1   xi_v0   xi_v1"
else:
    print("Do not measure correlation length.")

compute_rdm_2nd_nei = bool(config["compute_rdm_2nd_nei"])
if compute_rdm_2nd_nei:
    print(
        "Compute next nearest neighbor density matrix and observables once",
        "environment is converged",
    )
    obs_str += "    energy    capacity"
    print(
        "Compute capacity as -(beta+dbeta/2)**2 * (E(beta) - E(beta+dbeta)) /", "dbeta"
    )
    save_rdm_root = str(config["save_rdm_root"])
    print(
        "Save reduced density matrices in files "
        + save_rdm_root
        + "{beta}_chi{chi}.npz"
    )
else:
    print("Do not compute next nearest neighbor density matrix.")


########################################################################################
# Computation starts
########################################################################################

su_files = []
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
    save_su = save_su_root + f"{su.beta:.4f}.npz"  # beta != su.beta due to finite tau
    data_su = su.save_to_file()
    np.savez_compressed(save_su, beta=su.beta, **su_params, **data_su)
    su_files.append(save_su)
    print("Simple update data saved in file", save_su)


####################################################################################
# CTMRG
####################################################################################
# CTMRG initialization
print("", "#" * 75, sep="\n")
print(
    "Simple update is over. Now compute CTMRG for all simple update saved files",
    "going backwards in beta",
)
print("", "#" * 75, sep="\n")
ctm = CTMRG_U1.from_elementary_tensors(
    su.get_ABCD(), su.get_colors_ABCD(), tiling, chi_list[0], verbosity=1
)
last_beta = np.nan
last_energy = np.full(chi_list.size, np.nan)

for suf in reversed(su_files):
    print("", "#" * 75, sep="\n")
    print("Get back simple update from file", suf)
    su = su = SimpleUpdate2x2(d, a, file=suf)
    ctm.set_tensors(su.get_ABCD(), su.get_colors_ABCD())
    ctm_params["beta"] = su.beta
    ctm_restart = save_ctm_root + f"{su.beta:.4f}_chi{chi_list[0]}.npz"  # for next beta

    # prepare observable for several chis
    energy_chi, ising_chi, xi_h_chi, xi_v_chi, capacity_chi = [], [], [], [], []
    obs_chi = []

    # run CTMRG at fixed chi
    for chi_index, chi in enumerate(chi_list):
        print("\n" + "#" * 75)
        ctm.chi = chi
        print(
            f"Compute environment at beta = {su.beta} for D = {Dmax} and chi =",
            f"{ctm.chi}...",
        )
        t = time.time()
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
        save_ctm = save_ctm_root + f"{su.beta:.4f}_chi{ctm.chi}.npz"
        ctm_params["chi"] = ctm.chi
        ctm.save_to_file(save_ctm, ctm_params)

        ################################################################################
        # Observables
        ################################################################################
        rdm2x1_cell = np.array(rdm2x1_cell)
        rdm1x2_cell = np.array(rdm1x2_cell)
        ising_op = ((rdm1x2_cell - rdm2x1_cell) * h1).sum()
        ising_chi.append(ising_op)
        print(f"Ising order parameter = {ising_op:.3e}")
        obs = f"{ising_op:.3f}"

        if measure_xi:
            print("Compute every non-equivalent correlation lengths...")
            t = time.time()
            xi_h0 = ctm.compute_corr_length_h(0)
            xi_h1 = ctm.compute_corr_length_h(1)
            xi_h_chi.append((xi_h0, xi_h1))
            print(f"done for horizontal direction, t = {time.time()-t:.0f}")
            print(f"xi_h = {xi_h0:.3f}, {xi_h1:.3f}")
            t = time.time()
            xi_v0 = ctm.compute_corr_length_v(0)
            xi_v1 = ctm.compute_corr_length_v(1)
            xi_v_chi.append((xi_v0, xi_v1))
            print(f"done for vertical direction, t = {time.time()-t:.0f}")
            print(f"xi_v = {xi_v0:.3f}, {xi_v1:.3f}")
            obs += f"   {xi_h0:.3f}   {xi_h1:.3f}   {xi_v0:.3f}   {xi_v1:.3f}"

        if compute_rdm_2nd_nei:
            print("Compute reduced density matrix for all second neighbor bonds...")
            save_rdm = save_rdm_root + f"{su.beta:.4f}_chi{ctm.chi}.npz"
            t = time.time()
            rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
            rdm_dr_cell = np.array(rdm_dr_cell)
            rdm_ur_cell = np.array(rdm_ur_cell)
            print(f"done with rdm computation, t = {time.time()-t:.0f}")
            np.savez_compressed(
                save_rdm,
                rdm1x2_cell=rdm1x2_cell,
                rdm2x1_cell=rdm2x1_cell,
                rdm_dr_cell=rdm_dr_cell,
                rdm_ur_cell=rdm_ur_cell,
                cell_coords=ctm.neq_coords,
                **ctm_params,
            )
            print("rdm saved to file", save_rdm)
            energy = ((rdm1x2_cell + rdm2x1_cell) * h1).sum() / 4 + (
                (rdm_dr_cell + rdm_ur_cell) * h2
            ).sum() / 4
            energy_chi.append(energy)
            print(f"energy = {energy}")
            obs += f"   {energy:.5f}"

            # always compute capacity, even if some values will be wrong.
            x = ((su.beta + last_beta) / 2) ** 2 / (last_beta - su.beta)
            c = x * (energy - last_energy[chi_index])
            capacity_chi.append(c)
            obs += f"   {c:.5f}"
        obs_chi.append(obs)
    last_beta = su.beta

    ################################################################################
    # Save and display observables for all chis
    ################################################################################
    obs_dic = {"chi_list": chi_list, "ising": np.array(ising_chi)}
    if measure_xi:
        obs_dic["xi_h"] = np.array(xi_h_chi)
        obs_dic["xi_v"] = np.array(xi_v_chi)
    if compute_rdm_2nd_nei:
        obs_dic["energy"] = np.array(energy_chi)
        obs_dic["capacity"] = np.array(capacity_chi)
        last_energy = energy_chi
    print("", "#" * 75, sep="\n")
    save_obs = save_obs_root + f"{su.beta:.4f}.npz"
    del ctm_params["chi"]
    np.savez_compressed(save_obs, **obs_dic, **ctm_params)
    print("observables saved in file", save_obs)
    print(f"observables for D = {Dmax}, tau = {tau}, beta = {su.beta}:")
    print(obs_str)
    for chi, obs in zip(chi_list, obs_chi):
        print(f"{chi:>3d}   {obs}")


print("\n" + "#" * 79)
print("Computations finished.")
