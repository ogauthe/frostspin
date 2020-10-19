#! /usr/bin/env python3

import sys
import json
import time

import numpy as np

from simple_update import SimpleUpdate2x2


########################################################################################
# Initialization
########################################################################################
print("#" * 79)
print("Finite temperature simple update for J1-J2 Heisenberg model")
if len(sys.argv) < 2:
    config_file = "input_sample/input_sample_run_SU.json"
    print("No input file given, use", config_file)
else:
    config_file = sys.argv[1]
    print("Take input parameters from file", config_file)

with open(config_file) as f:
    config = json.load(f)

# physical model parameters
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
print("Save SU for beta in", list(beta_list))

# save parameters (do not mix internal SU stuff and simulation parameters)
su_params = {"tau": tau, "J2": J2, "Dmax": Dmax}
save_su_root = str(config["save_su_root"])
print("\nSave simple update data in files " + save_su_root + "{beta}.npz")

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


########################################################################################
# Computation starts
########################################################################################

for beta in beta_list:
    print("\n" + "#" * 79)
    print(f"Evolve in imaginary time for beta from {su.beta} to {beta}...")
    beta_evolve = beta - su.beta
    t = time.time()
    su.evolve(beta_evolve)
    print(f"done with imaginary time evolution, t = {time.time()-t:.0f}")
    print("lambdas =", *su.lambdas[1:], sep="\n")
    print("colors =", *su.colors[1:], sep="\n")
    save_su = save_su_root + f"{su.beta}.npz"
    data_su = su.save_to_file()
    np.savez_compressed(save_su, beta=su.beta, **su_params, **data_su)
    print("Simple update data saved in file", save_su)


print("\n" + "#" * 79)
print("Computations finished.")
