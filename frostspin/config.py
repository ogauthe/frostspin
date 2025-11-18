"""
This file sets global configuration for frostspin
"""

import argparse
import os

__version__ = "1.0.0"

# =================================  Parse arguments  ==================================
local_dir = os.path.join(os.path.dirname(__file__))
default_su2_cg_file = os.path.join(
    local_dir, "groups/clebsch-gordan_data/data_su2_clebsch-gordan.json"
)

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    "--frostspin-SU2-CG-file",
    help="savefile for SU(2) Clebsch-Gordan",
    type=str,
    default=default_su2_cg_file,
)

args, _ = parser.parse_known_args()
config = {"SU2_CG_file": args.frostspin_SU2_CG_file}


# ==============================  Display debug warning  ===============================
if not __debug__:
    print("\nInfo: assert statements are disabled")

ASSERT_TOL = 4e-13
