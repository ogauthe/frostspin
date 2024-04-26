"""
This file sets global configuration for froSTspin
"""

import argparse
import os

local_dir = os.path.join(os.path.dirname(__file__))
default_su2_cg_file = os.path.join(
    local_dir, "groups/clebsch-gordan_data/data_su2_clebsch-gordan.json"
)

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("--froSTspin-quiet", help="silence froSTspin", action="store_true")
parser.add_argument(
    "--froSTspin-SU2-CG-file",
    help="savefile for SU(2) Clebsch-Gordan",
    type=str,
    default=default_su2_cg_file,
)

args, _ = parser.parse_known_args()
config = {"quiet": args.froSTspin_quiet, "SU2_CG_file": args.froSTspin_SU2_CG_file}

# display warning if running in debug mode
if __debug__:  # noqa: SIM102
    if not config["quiet"]:
        print("\nWarning: assert statement are activated")
        print("They may significantly impact performances")
        print("Consider running the code in optimized mode with python -O")
        print("You may disable this warning with the flag --froSTspin-quiet\n")
