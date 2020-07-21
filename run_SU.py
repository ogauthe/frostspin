#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from simple_updateABCD import SimpleUpdateABCD
from test_tools import SdS_22b
from CTMRG import CTMRG


tiling = """
AB
CD
"""

def evalH(su,chi,niter=10):
  A,B,C,D = su.get_ABCD()
  ctm = CTMRG((A,B,C,D),tiling,chi,verbosity=2)
  for i in range(niter):
    ctm.iterate()

  rdm1x2 = ctm.compute_rdm1x2()
  return np.trace(su.h1 @ rdm1x2)

d = 2
a = 1
chi = 11
Ds = 2,3,4,5,6,7,9,10
su_iter = 100
print(f'run simple update with d = {d}, a = {a}, Ds = {Ds}, converge CTM with chi = {chi}')

tau = 0.1
su = SimpleUpdateABCD(d,a,Ds,SdS_22b, SdS_22b, tau)

eps0 = evalH(su,chi,3)
print(f'Random tensors: <h1> = {eps0:.4e}')

print("test 1st neighbor updates")
su.update_bond1()
su.update_bond2()
su.update_bond3()
su.update_bond4()
su.update_bond5()
su.update_bond6()
su.update_bond7()
su.update_bond8()
print("done")

"""
print(f'Iter simple update for {su_iter} times', end='...')
for i in range(su_iter):
  su.update()
print(' Done. Evaluate energy')

eps1 = evalH(su,chi)
print(f'Done, <h1> = {eps1:.4e}')
"""
