#! /usr/bin/env python3
import numpy as np
import scipy.linalg as lg
from CTMRG import CTMRG

####   SU(2) misc   ###########################################################
def construct_genSU2_s(s):
  """
  Construct generator for spin-s irrep of SU(2)
  """
  if s<0 or int(2*s) != 2*s:
    raise ValueError("s must be a positive half integer")

  d = int(2*s) + 1
  basis = np.arange(s,-s-1,-1)
  Sm = np.zeros((d,d))
  Sm[np.arange(1,d),np.arange(d-1)] = np.sqrt(s*(s+1)-basis[:-1]*(basis[:-1]-1))
  Sp = Sm.T
  gen = np.empty((3,d,d),dtype=complex)
  gen[0] = (Sp + Sm)/2     # Sx
  gen[1] = (Sp - Sm)/2j    # Sy
  gen[2] = np.diag(basis)  # Sz
  return gen

tAKLT2 = np.zeros((5, 2, 2, 2, 2),dtype=np.float64)
tAKLT2[0,0,0,0,0] = 1.0
tAKLT2[1,0,0,0,1] = 0.5
tAKLT2[1,0,0,1,0] = 0.5
tAKLT2[1,0,1,0,0] = 0.5
tAKLT2[1,1,0,0,0] = 0.5
tAKLT2[2,0,0,1,1] = 1/np.sqrt(6)
tAKLT2[2,0,1,0,1] = 1/np.sqrt(6)
tAKLT2[2,0,1,1,0] = 1/np.sqrt(6)
tAKLT2[2,1,0,0,1] = 1/np.sqrt(6)
tAKLT2[2,1,0,1,0] = 1/np.sqrt(6)
tAKLT2[2,1,1,0,0] = 1/np.sqrt(6)
tAKLT2[3,0,1,1,1] = 0.5
tAKLT2[3,1,0,1,1] = 0.5
tAKLT2[3,1,1,0,1] = 0.5
tAKLT2[3,1,1,1,0] = 0.5
tAKLT2[4,1,1,1,1] = 1.0

gen5 = construct_genSU2_s(2)
SdS55 = np.tensordot(gen5, -gen5.conj(), (0,0)).real.swapaxes(1,2).reshape(25,25)
SdS55_2 = SdS55 @ SdS55

H_AKLT = 1/28*SdS55 + 1/40*SdS55_2 + 1/180*SdS55@SdS55_2 + 1/2520*SdS55_2@SdS55_2
###############################################################################

d = 5
D = 2
chi = 3

tensors = [tAKLT2.copy(),tAKLT2.copy(),tAKLT2.copy()]
tiling = """
ABC
BCA
CAB
"""

tensors = [tAKLT2.copy()]
tiling = """
A
"""


ctm = CTMRG(tensors,tiling,chi,verbosity=100)
print('CTM constructed, tensors shapes are:')
print('T1 shape =', ctm.env.get_T1(0,0).shape)
print('C1 shape =', ctm.env.get_C1(0,0).shape)
print('T2 shape =', ctm.env.get_T2(0,0).shape)
print('C2 shape =', ctm.env.get_C2(0,0).shape)
print('T3 shape =', ctm.env.get_T3(0,0).shape)
print('C3 shape =', ctm.env.get_C3(0,0).shape)
print('T4 shape =', ctm.env.get_T4(0,0).shape)
print('C4 shape =', ctm.env.get_C4(0,0).shape)

print('\ntry 2 iterations:')
ctm.iterate()
ctm.iterate()
ctm.chi = 5
ctm.iterate()
print('done')

ctm.chi = 6
for i in range(20):
  ctm.iterate()

rdm1x1 = ctm.compute_rdm1x1()
rdm1x2 = ctm.compute_rdm1x2()
rdm2x1 = ctm.compute_rdm2x1()
rdm2x2 = ctm.compute_rdm2x2()

print('rdm1x1 is hermitian:', lg.norm(rdm1x1-rdm1x1.T.conj()))
print('rdm1x2 is hermitian:', lg.norm(rdm1x2-rdm1x2.T.conj()))
print('rdm2x1 is hermitian:', lg.norm(rdm2x1-rdm2x1.T.conj()))
print('rdm2x2 is hermitian:', lg.norm(rdm2x2-rdm2x2.T.conj()))

print('trace(H_AKLT @ rdm1x2) =', np.trace(H_AKLT @ rdm1x2))
print('trace(H_AKLT @ rdm2x1) =', np.trace(H_AKLT @ rdm2x1))
