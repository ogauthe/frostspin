#!/usr/bin/env python

"""
This script allows to compute SU(2) Clebsch-Gordan coefficients and to store them.
By default it recomputes the sample, which provides enough coefficients to run the tests
and the examples. For intensive computation, consider running this script with a large
max_spin_dimension value and save the coefficients in a .npz file.
"""

import argparse

from frostspin.groups.su2_clebsch_gordan import default_su2_cg_file, save_su2_cg

parser = argparse.ArgumentParser()

parser.add_argument(
    "--SU2-CG-file", help="savefile for SU(2) CG", type=str, default=default_su2_cg_file
)
parser.add_argument(
    "--max-spin-dimension", help="Maximal spin dimension", type=int, default=10
)
args = parser.parse_args()

cg_file = args.SU2_CG_file
max_spin_dim = args.max_spin_dimension

save_su2_cg(cg_file, max_spin_dim)
