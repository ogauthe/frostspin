#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from test_tools import H_AKLT

d = 5
a = 1
D = 3
sh = ((d,a,D,D,D,D))

gates = [H_AKLT]*4
su = SimpleUpdateAB(sh,gates)

for i in range(100):
  su.update()
