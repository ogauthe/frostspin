#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from simple_updateABCD import SimpleUpdateABCD
from test_tools import SdS_22b, cconj2
from CTMRG import CTMRG


tiling = 'AB\nCD'

d = 2
a = 2
chi = 25
ctm_iter = 10
D = 5   # 2+2+1
Ds = (D,)*8
print(f'run simple update with d = {d}, a = {a}, Ds = {Ds}')

beta = 1
tau = 0.001
su_iter = int(beta/2/tau)  # rho is quadratic in psi

A = np.zeros((d,a,D,D,D,D))  # maximally entangled between physical and ancila
A[0,1,2,2,2,2] = 1
A[1,0,2,2,2,2] = -1
B = np.tensordot(cconj2,A,((1,),(0,)))   # rotation on B sites
C = B.copy()
D = A.copy()

su = SimpleUpdateABCD(d,a,Ds,SdS_22b, np.zeros((4,4)), tau, tensors=(A,B,C,D))


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
print(f"done with SU. Converge CTMRG with chi = {chi} and niter = {ctm_iter}")
ctm = CTMRG((A,B,C,D),tiling,chi,verbosity=0)
for i in range(ctm_iter):
  print(i)
  ctm.iterate()


rdmD = ctm.compute_rdm1x1(0,0)
rdmC = ctm.compute_rdm1x1(1,0)
rdmB = ctm.compute_rdm1x1(0,1)
rdmA = ctm.compute_rdm1x1(1,1)

rdmAB = ctm.compute_rdm1x2(0,0)
rdmAC = ctm.compute_rdm2x1(0,0)
rdmBA = ctm.compute_rdm1x2(1,0)
rdmCA = ctm.compute_rdm2x1(0,1)

print(lg.norm(rdmA - 0.5*np.eye(2)), lg.norm(rdmB - 0.5*np.eye(2)), lg.norm(rdmC - 0.5*np.eye(2)) ,lg.norm(rdmD - 0.5*np.eye(2)))
print(lg.norm(rdmAB - 0.25*np.eye(4)), lg.norm(rdmAB - 0.25*np.eye(4)), lg.norm(rdmBA - 0.25*np.eye(4)) ,lg.norm(rdmCA - 0.25*np.eye(4)))

print("done with CTM. Compute rdm")
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
