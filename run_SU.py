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


d = 2
a = 2
chi = 25
niter = 10
Ds = [5]*8
print(f'run simple update with d = {d}, a = {a}, Ds = {Ds}')

tau = 0.001
su = SimpleUpdateABCD(d,a,Ds,SdS_22b, np.zeros((4,4)), tau)

print("Run simple update for spin 1/2 Heisenberg with J1=1, J2=0")
print(f"run with tau = {tau} up to beta = 1")
"""
for i in range(np.rint(1/2/tau).astype(int)):
  print(i, end=" ")
  su.update_bond1()
  su.update_bond2()
  su.update_bond3()
  su.update_bond4()
  su.update_bond5()
  su.update_bond6()
  su.update_bond7()
  su.update_bond8()
  su.update_bond8()
  su.update_bond7()
  su.update_bond6()
  su.update_bond5()
  su.update_bond4()
  su.update_bond3()
  su.update_bond2()
  su.update_bond1()
"""
print(f"done with SU. Converge CTMRG with chi = {chi} and niter = {niter}")
ctm = CTMRG(su.get_ABCD(),tiling,chi,verbosity=0)
for i in range(niter):
  print(i)
  ctm.iterate()

print("done with CTM. Compute rdm")
rdmAB = ctm.compute_rdm1x2(0,0)
rdmAC = ctm.compute_rdm2x1(0,0)
rdmBA = ctm.compute_rdm1x2(1,0)
rdmCA = ctm.compute_rdm2x1(0,1)
eps = 0.5*np.trace(su.h1 @ (rdmAB + rdmAC + rdmBA + rdmCA))
print(f"epsilon = {eps}")

"""
su.update_bonds25() # through B
su.update_bonds17() # through C
su.update_bonds26() # through B
su.update_bonds37() # through C
su.update_bonds46() # through B
su.update_bonds38() # through C
su.update_bonds45() # through B
su.update_bonds18() # through C

su.update_bonds41() # through A
su.update_bonds58() # through D
su.update_bonds43() # through A
su.update_bonds68() # through D
su.update_bonds23() # through A
su.update_bonds67() # through D
su.update_bonds21() # through A
su.update_bonds57() # through D

su.update_bonds57() # through D
su.update_bonds21() # through A
su.update_bonds67() # through D
su.update_bonds23() # through A
su.update_bonds68() # through D
su.update_bonds43() # through A
su.update_bonds58() # through D
su.update_bonds41() # through A

su.update_bonds18() # through C
su.update_bonds45() # through B
su.update_bonds38() # through C
su.update_bonds46() # through B
su.update_bonds37() # through C
su.update_bonds26() # through B
su.update_bonds17() # through C
su.update_bonds25() # through B
print("done")

"""
