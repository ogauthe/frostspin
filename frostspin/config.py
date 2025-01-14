"""
This file sets global configuration for frostspin
"""

import argparse
import os

__version__ = "0.5.0"

# =================================  Parse arguments  ==================================
local_dir = os.path.join(os.path.dirname(__file__))
default_su2_cg_file = os.path.join(
    local_dir, "groups/clebsch-gordan_data/data_su2_clebsch-gordan.json"
)

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("--frostspin-quiet", help="silence frostspin", action="store_true")
parser.add_argument(
    "--frostspin-SU2-CG-file",
    help="savefile for SU(2) Clebsch-Gordan",
    type=str,
    default=default_su2_cg_file,
)

args, _ = parser.parse_known_args()
config = {"quiet": args.frostspin_quiet, "SU2_CG_file": args.frostspin_SU2_CG_file}


# ==============================  Display debug warning  ===============================
if __debug__:  # noqa: SIM102
    if not config["quiet"]:
        print("\nWarning: assert statement are activated")
        print("They may significantly impact performances")
        print("Consider running the code in optimized mode with python -O")
        print("You may disable this warning with the flag --frostspin-quiet\n")

print("\n *** FROSTSPIN IS ON BRANCH DEV ***\n")

ASSERT_TOL = 4e-13
