import os
import json
import argparse

import numpy as np


local_dir = os.path.join(os.path.dirname(__file__))
default_su2_cg_file = os.path.join(
    local_dir, "clebsch-gordan_data/data_su2_clebsch-gordan.json"
)


def compute_CG(max_spin_dim):
    # load sympy only if recomputing Clebsch-Gordon is required.
    import sympy as sp
    from sympy.physics.quantum.cg import CG

    elementary_projectors = {}
    max_irr = max_spin_dim + 1
    for irr1 in range(1, max_irr):
        s1 = sp.Rational(irr1 - 1, 2)

        # irr1 x irr2 = sum irr3
        for irr2 in range(1, max_irr):
            s2 = sp.Rational(irr2 - 1, 2)
            for irr3 in range(abs(irr2 - irr1) + 1, irr1 + irr2, 2):
                s3 = sp.Rational(irr3 - 1, 2)
                p = np.zeros((irr1, irr2, irr3), dtype=float)
                for i1 in range(irr1):
                    m1 = s1 - i1
                    for i2 in range(irr2):
                        m2 = s2 - i2
                        for i3 in range(irr3):
                            m3 = s3 - i3
                            p[i1, i2, i3] = CG(s1, m1, s2, m2, s3, m3).doit()  # exact
                elementary_projectors[irr1, irr2, irr3] = p  # cast to float
    return elementary_projectors


# json format is text format, easier to read and parse
# it can also be added to git
# however it generates heavier files and is slower to load
def _save_CG_json(savefile, elementary_projectors):
    max_irr = np.array(list(elementary_projectors.keys()))[:, 0].max()
    info = [
        "This file was generated by froSTspin/groups/su2_clebsch_gordan.py.",
        "It contains Clebsch-Gordan coefficients for SU(2).",
        "Each irreducible represensation is labelled by its dimension 2j+1.",
        "Clebsch-Gordan tensors with shape (2j1+1, 2j2+1, 2j3+1) are stored as",
        "a pair of indices and coeff for the flat tensor in C order.",
        "Do not edit this file manually.",
    ]
    cg_data = {"!!info": info, "!maximal_spin_dimension": int(max_irr)}
    for k in elementary_projectors:
        nk = f"{k[0]},{k[1]},{k[2]}"

        # save as sparse 1D
        proj = elementary_projectors[k]
        inds = np.flatnonzero(proj)
        coeff = proj.flat[inds]
        cg_data[nk] = [inds.tolist(), coeff.tolist()]

    with open(savefile, "w") as out:
        json.dump(cg_data, out, indent=2, sort_keys=True)
    return


def _load_CG_json(savefile):
    with open(savefile) as fin:
        cg_data = json.load(fin)

    del cg_data["!!info"]
    del cg_data["!maximal_spin_dimension"]
    elementary_projectors = {}
    for key in cg_data:
        sh = tuple(map(int, key.split(",")))
        inds, coeff = cg_data[key]
        proj = np.zeros((sh[0] * sh[1] * sh[2],))
        proj[inds] = coeff
        proj = proj.reshape(sh)
        elementary_projectors[sh] = proj

    return elementary_projectors


# for intensive computation and large CG files, a binary format is more convenient
def _save_CG_npz(savefile, elementary_projectors):
    cg_data = {}
    for k in elementary_projectors:
        sfx = f"_{k[0]}_{k[1]}_{k[2]}"

        # save as sparse 1D
        proj = elementary_projectors[k]
        inds = np.flatnonzero(proj)
        coeff = proj.flat[inds]
        cg_data["i" + sfx] = inds
        cg_data["c" + sfx] = coeff

    np.savez_compressed(savefile, **cg_data)
    return


def _load_CG_npz(savefile):
    elementary_projectors = {}
    with np.load(savefile) as data:
        for key in data.files:
            if key[0] == "i":
                sh = tuple(int(d) for d in key[2:].split("_"))
                proj = np.zeros((sh[0] * sh[1] * sh[2],))
                inds = data[key]
                coeff = data["c" + key[1:]]
                proj[inds] = coeff
                proj = proj.reshape(sh)
                elementary_projectors[sh] = proj
    return elementary_projectors


def load_su2_cg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--froSTspin-SU2-CG-file",
        help="savefile for SU(2) CG",
        type=str,
        default=default_su2_cg_file,
    )
    parser.add_argument(
        "--froSTspin-quiet", help="silence froSTspin", action="store_true"
    )
    args = parser.parse_args()

    cg_file = args.froSTspin_SU2_CG_file
    extension = cg_file.split(".")[-1]

    if not args.froSTspin_quiet:
        print("Load SU(2) Clebsch-Gordan coefficient from file", cg_file)

    try:
        if extension == "npz":
            elementary_projectors = _load_CG_npz(cg_file)
        elif extension == "json":
            elementary_projectors = _load_CG_json(cg_file)
        else:
            raise ValueError(
                f"Invalid file extension: must be json or npz, got {extension}"
            )
    except FileNotFoundError:
        print("Error: SU(2) Clebsch-Gordan coefficient savefile not found")

        # Computing CG coeff is expensive.
        # better to crash now than to silently trigger recomputation.
        raise FileNotFoundError(cg_file)

    max_spin_dim = max(k[0] for k in elementary_projectors.keys())
    elementary_projectors["maximal_spin_dimension"] = max_spin_dim
    print("Clebsch-Gordan maximal spin dimension =", max_spin_dim)

    return elementary_projectors


def save_su2_cg(savefile, max_spin_dim):
    # input validation
    max_spin_dim = int(max_spin_dim)
    cg_file = str(savefile)
    extension = cg_file.split(".")[-1]
    if extension not in ["json", "npz"]:
        print(
            f"Error: Invalid savefile extension: must be json or npz, got {extension}"
        )
        print(f"savefile: {cg_file}")
        raise ValueError(extension)

    # check output file is writable
    try:
        with open(cg_file, "w") as out:
            out.write("-")
    except FileNotFoundError:
        print("Error: cannot write SU(2) Clebsch-Gordan coefficient savefile")
        print(f"savefile: {cg_file}")
        raise FileNotFoundError(cg_file)

    # compute CG coeff
    print(f"Compute SU(2) Clebsch-Gordon coefficients up to 2s+1 = {max_spin_dim}")
    elementary_projectors = compute_CG(max_spin_dim)
    print("Coefficient computation done.")

    # save coeff
    if extension == "npz":
        _save_CG_npz(cg_file, elementary_projectors)
    else:
        _save_CG_json(cg_file, elementary_projectors)
    print(f"SU2(2) Clebsch-Gordon coefficients saved in file {cg_file}")

    return
