#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from test_tools import H_AKLT
from CTMRG import CTMRG


tiling = """
AB
BA
"""

def evalH(su,chi,niter=100):
  A,B = su.get_AB()
  A = A.reshape(d,D,D,D,D)   # remove ancila
  B = B.reshape(d,D,D,D,D)
  ctm = CTMRG((A,B),tiling,chi,verbosity=0)
  for i in range(niter):
    ctm.iterate()

  rdm1x2 = ctm.compute_rdm1x2()
  return np.trace(H_AKLT @ rdm1x2)

d = 5
a = 1
D = 3
chi = 40
sh = ((d,a,D,D,D,D))
su_iter = 1000
print(f'run simple update with d = {d}, a = {a}, D = {D}, converge CTM with chi = {chi}')

tau = 0.01
gates = [lg.expm(-tau*H_AKLT)]*4
su = SimpleUpdateAB(sh,gates)

eps0 = evalH(su,chi)
print(f'Random tensors: <H_AKLT> = {eps0:.1e}')

print(f'Iter simple update for {su_iter} times', end='...')
for i in range(su_iter):
  su.update()
print(' Done. Evaluate energy')

eps1 = evalH(su,chi)
print(f'Done, <H_AKLT> = {eps1:.1e}')
