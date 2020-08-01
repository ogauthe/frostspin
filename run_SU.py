#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from simple_updateABCD import SimpleUpdateABCD
from test_tools import SdS_22b, cconj2
from CTMRG import CTMRG
from toolsU1 import checkU1, combine_colors


tiling = 'AB\nCD'

d = 2
a = 2
chi = 35
ctm_iter = 100
D = 5 # 3+1+1 at start
Ds = (D,)*8
print(f'run simple update with d = {d}, a = {a}, Ds = {Ds}')

beta = 1
tau = 0.001
su_iter = int(beta/2/tau)  # rho is quadratic in psi

# zero singular values lead to divide by zero when removing weights
# to avoid this, start with tensors made of decoupled product of (2x2)->1 on
# physical x ancila and 3^4 -> 1 (B1) on virtual legs.
p22_3333 = np.zeros((2,2,3,3,3,3),dtype=np.int8)
p22_3333.flat[[89,95,103,105,115,119,123,127,137,139,147,153,170,176,184,186,
               196,200,204,208,218,220,228,234]] = [1,-1,1,-1,-1,1,1,-1,-1,1,
                                 -1,1,-1,1,-1,1,1,-1,-1, 1,1,-1, 1,-1]

A = np.zeros((d,a,D,D,D,D))
A[:,:,:3,:3,:3,:3] = p22_3333
#B = np.tensordot(cconj2,A,((1,),(0,)))   # rotation on B sites
B = A.copy()
C = B.copy()
D = A.copy()

# colors A and D = [[1,-1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]
# opposed colors on B and C physical leg
# colors B and C = [[-1,1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]

#su = SimpleUpdateABCD(d,a,Ds,SdS_22b, np.zeros((4,4)), tau, tensors=(A,B,C,D))
#su = SimpleUpdateABCD(d,1,Ds,SdS_22b, np.zeros((4,4)), tau, verbosity=10)
pcol = np.array([1,-1],dtype=np.int8)
vcol = np.array([2,0,-2,0,0],dtype=np.int8)
colors = [pcol,pcol,vcol,vcol,vcol,vcol,vcol,vcol,vcol,vcol]
#su = SimpleUpdateABCD(d,a,Ds,SdS_22b, np.zeros((4,4)), tau, tensors=(A,B,C,D), verbosity=10)
print(checkU1(A,[pcol,pcol,vcol,vcol,vcol,vcol]))
print(checkU1(B,[-pcol,-pcol,-vcol,-vcol,-vcol,-vcol]))
print(checkU1(SdS_22b,[combine_colors(pcol,-pcol),-combine_colors(pcol,-pcol)]))
#print(checkU1(B,[pcol,-pcol,-vcol,-vcol,-vcol,-vcol]))
su = SimpleUpdateABCD(d,a,Ds,SdS_22b, np.zeros((4,4)), tau, tensors=(A,B,C,D), colors=colors, verbosity=10)

print("#"*79)
su.update_bond1()
su.update_bond2()
su.update_bond3()
su.update_bond4()
su.update_bond5()
su.update_bond6()
su.update_bond7()
su.update_bond8()
print("#"*79)
su.update_bond8()
su.update_bond7()
su.update_bond6()
su.update_bond5()
su.update_bond4()
su.update_bond3()
su.update_bond2()
su.update_bond1()
print("#"*79)
su.update_bond1()
su.update_bond2()
su.update_bond3()
su.update_bond4()
su.update_bond5()
su.update_bond6()
su.update_bond7()
su.update_bond8()
print("#"*79)
su.update_bond8()
su.update_bond7()
su.update_bond6()
su.update_bond5()
su.update_bond4()
su.update_bond3()
su.update_bond2()
su.update_bond1()

"""
print("Run simple update for spin 1/2 Heisenberg with J1=1, J2=0")
print(f"run with tau = {tau} up to beta = {beta}")

#for i in range(su_iter//2): # 2nd order Trotter
for i in range(5): # 1st order
  print("iter", i)
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

print(f"\ndone with SU. Converge CTMRG with chi = {chi} and niter = {ctm_iter}")
ctm = CTMRG(su.get_ABCD(),tiling,chi,verbosity=0)
rdmD = ctm.compute_rdm1x1(0,0)
rdmC = ctm.compute_rdm1x1(1,0)
rdmB = ctm.compute_rdm1x1(0,1)
rdmA = ctm.compute_rdm1x1(1,1)

rdmAB = ctm.compute_rdm1x2(0,0)
rdmAC = ctm.compute_rdm2x1(0,0)
rdmBA = ctm.compute_rdm1x2(1,0)
rdmCA = ctm.compute_rdm2x1(0,1)

print("before CTM, compute rdm")
eps = 0.5*np.trace(su.h1 @ (rdmAB + rdmAC + rdmBA + rdmCA))
print(f"epsilon = {eps}")
for i in range(ctm_iter):
  print(i, end=" ")
  ctm.iterate()


rdmD = ctm.compute_rdm1x1(0,0)
rdmC = ctm.compute_rdm1x1(1,0)
rdmB = ctm.compute_rdm1x1(0,1)
rdmA = ctm.compute_rdm1x1(1,1)

rdmAB = ctm.compute_rdm1x2(0,0)
rdmAC = ctm.compute_rdm2x1(0,0)
rdmBA = ctm.compute_rdm1x2(1,0)
rdmCA = ctm.compute_rdm2x1(0,1)

#print(lg.norm(rdmA - 0.5*np.eye(2)), lg.norm(rdmB - 0.5*np.eye(2)), lg.norm(rdmC - 0.5*np.eye(2)) ,lg.norm(rdmD - 0.5*np.eye(2)))
#print(lg.norm(rdmAB - 0.25*np.eye(4)), lg.norm(rdmAB - 0.25*np.eye(4)), lg.norm(rdmBA - 0.25*np.eye(4)) ,lg.norm(rdmCA - 0.25*np.eye(4)))

print("\ndone with CTM. Compute rdm")
eps = 0.5*np.trace(su.h1 @ (rdmAB + rdmAC + rdmBA + rdmCA))
print(f"epsilon = {eps}")


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
