#! /usr/bin/env python3

import sys
import json
import time

import numpy as np

from simple_update import SimpleUpdate2x2
from ctmrg import CTMRG_U1
from converger import Converger


########################################################################################
# Initialization
########################################################################################
print("#" * 79)
print("Finite temperature PEPS for J1-J2 Heisenberg model")
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
dbeta = 4 * tau  # one SU iteration
beta_goal = np.array(config["beta_goal"], dtype=float)
su_cutoff = float(config["su_cutoff"])
print(f"\nSimple update parameters: tau = {tau}, Dmax = {Dmax}, cutoff = {su_cutoff}")
print("Goal for imaginary time evolution steps are:", repr(beta_goal)[6:-1])

chi_list = np.array(config["chi_list"], dtype=int)
run_CTMRG = chi_list.size != 0
measure_capacity = bool(config["measure_capacity"])
last_energy = np.full(chi_list.size, np.nan)
if run_CTMRG and measure_capacity:
    print("Compute thermal capacity: for each beta, add beta + dbeta to beta goal")
    print(f"with dbeta = {dbeta}")
    beta_goal = np.hstack((beta_goal + dbeta, beta_goal))

beta_reach = np.array(sorted(set(np.rint(beta_goal / dbeta).astype(int)))) * dbeta
print("Take into accout finite tau and remove doubles: reached beta will be:")
print(repr(beta_reach)[6:-1])

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
    su = SimpleUpdate2x2(
        d, a, Dmax, tau, h1, h2, colors=pcol, cutoff=su_cutoff, verbosity=0
    )

# save parameters (do not mix internal CTM/SU stuff and simulation parameters)
su_params = {"tau": tau, "J2": J2, "Dmax": Dmax}
save_su_root = str(config["save_su_root"])
print("Save simple update data in files " + save_su_root + "{beta}.npz")

# CTMRG parameters
if not run_CTMRG:
    print("\nDo not compute CTMRG environment.")
else:
    ctm_tol = float(config["ctm_tol"])
    ctm_warmup = int(config["ctm_warmup"])
    ctm_maxiter = int(config["ctm_maxiter"])

    def ctm_value(ctm):  # estimate convergence from 1 horizontal bond + 1 vertical bond
        (x, y) = ctm.neq_coords[0]
        value = np.empty((2, 4, 4))
        value[0] = ctm.compute_rdm1x2(x, y)
        value[1] = ctm.compute_rdm2x1(x, y)
        return value

    print(f"\nCompute CTMRG environment for chi in {list(chi_list)}")
    if config["ctm_restart_file"] is not None:
        ctm_restart = str(config["ctm_restart_file"])
        print("Restart environment from file", ctm_restart)
    else:
        ctm_restart = None
        print("ctm_restart_file not provided, start from scratch")
    print(
        f"Converge CTMRG for at least {ctm_warmup} and at most {ctm_maxiter}",
        "iterations",
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
        obs_str += "    energy"
        print(
            "Compute capacity as -(beta+dbeta/2)**2 * (E(beta) - E(beta+dbeta)) /",
            "dbeta",
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

for beta in beta_reach:
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
    dbeta_step = bool(beta_evolve < 1.1 * dbeta)
    save_su = save_su_root + f"{su.beta:.4f}.npz"
    data_su = su.save_to_file()
    np.savez_compressed(save_su, beta=su.beta, **su_params, **data_su)
    print("Simple update data saved in file", save_su)

    ####################################################################################
    # CTMRG
    ####################################################################################
    # CTMRG initialization
    if run_CTMRG:
        print("", "#" * 75, sep="\n")
        if ctm_restart is None:  # init from scratch
            ctm = CTMRG_U1.from_elementary_tensors(
                su.get_ABCD(), su.get_colors_ABCD(), tiling, chi_list[0], verbosity=1
            )
        else:
            ctm = CTMRG_U1.from_file(ctm_restart, verbosity=1)
            ctm.set_tensors(su.get_ABCD(), su.get_colors_ABCD())

        conv = Converger(ctm.iterate, lambda: ctm_value(ctm), verbosity=1)
        ctm_params["beta"] = su.beta
        ctm_restart = save_ctm_root + f"{su.beta:.4f}_chi{chi_list[0]}.npz"  # next iter

        # prepare observable for several chis
        energy_chi, ising_chi, xi_h_chi, xi_v_chi, capacity_chi = [], [], [], [], []
        obs_chi = []

    # run CTMRG at fixed chi
    for chi_index, chi in enumerate(chi_list):
        print("\n" + "#" * 75)
        ctm.chi = chi
        conv.reset()
        print(
            f"Compute environment at beta = {su.beta} for D = {Dmax} and chi =",
            f"{ctm.chi}...",
        )
        t = time.time()
        is_converged, msg = conv.converge(ctm_tol, ctm_warmup, maxiter=ctm_maxiter)
        if is_converged:
            print(f"converged after {conv.niter} iterations, t = {time.time()-t:.0f}")
        else:
            print(f"Convergence failed after {conv.niter} iterations:", msg)
        save_ctm = save_ctm_root + f"{su.beta:.4f}_chi{ctm.chi}.npz"
        ctm_params["chi"] = ctm.chi
        ctm.save_to_file(save_ctm, ctm_params)

        ################################################################################
        # Observables
        ################################################################################
        rdm1x2_cell = np.empty((4, 4, 4))
        rdm2x1_cell = np.empty((4, 4, 4))
        rdm1x2_cell[0] = conv.value[0]
        rdm2x1_cell[0] = conv.value[1]
        print("Compute reduced density matrix for all first neighbor bonds...")
        t = time.time()
        for i, (x, y) in enumerate(ctm.neq_coords[1:]):
            rdm1x2_cell[i + 1] = ctm.compute_rdm1x2(x, y)
            rdm2x1_cell[i + 1] = ctm.compute_rdm2x1(x, y)
        print(f"done with rdm computation, t = {time.time()-t:.0f}")

        ising_op = ((rdm1x2_cell - rdm2x1_cell) * h1).sum()
        ising_chi.append(ising_op)
        print(f"\nIsing order parameter = {ising_op:.3e}")
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

            if dbeta_step:
                x = -((su.beta - dbeta / 2) ** 2) / dbeta
                c = x * (energy - last_energy[chi_index])
                capacity_chi.append(c)
                obs += f"   {c:.5f}"
        obs_chi.append(obs)

    ################################################################################
    # Save and display observables for all chis
    ################################################################################
    if run_CTMRG:
        obs_dic = {"chi_list": chi_list, "ising": np.array(ising_chi)}
        if measure_xi:
            obs_dic["xi_h"] = np.array(xi_h_chi)
            obs_dic["xi_v"] = np.array(xi_v_chi)
        if compute_rdm_2nd_nei:
            obs_dic["energy"] = np.array(energy_chi)
            last_energy = energy_chi
            if dbeta_step:
                obs_dic["capacity"] = np.array(capacity_chi)
        print("", "#" * 75, sep="\n")
        save_obs = save_obs_root + f"{su.beta:.4f}.npz"
        del ctm_params["chi"]
        np.savez_compressed(save_obs, **obs_dic, **ctm_params)
        print("observables saved in file", save_obs)
        print(f"observables for D = {Dmax}, tau = {tau}, beta = {su.beta}:")
        print(obs_str + dbeta_step * compute_rdm_2nd_nei * "    capacity")
        for chi, obs in zip(chi_list, obs_chi):
            print(f"{chi:>3d}   {obs}")


print("\n" + "#" * 79)
print("Computations finished.")
