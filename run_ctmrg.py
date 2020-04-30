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
cconj5 = np.rint(lg.expm(-1j*np.pi*gen5[1]).real)  # SU(2) charge conjugation = pi-rotation over y
tAKLT2_conj = np.tensordot(cconj5, tAKLT2,((1,),(0,)))
SdS55 = np.tensordot(gen5, gen5, (0,0)).real.swapaxes(1,2).reshape(25,25)
SdS55_2 = SdS55 @ SdS55
SdS55b = np.tensordot(gen5, -gen5.conj(), (0,0)).real.swapaxes(1,2).reshape(25,25)
SdS55b_2 = SdS55b @ SdS55b

H_AKLT = 1/28*SdS55 + 1/40*SdS55_2 + 1/180*SdS55@SdS55_2 + 1/2520*SdS55_2@SdS55_2
H_AKLT_55b = 1/28*SdS55b + 1/40*SdS55b_2 + 1/180*SdS55b@SdS55b_2 + 1/2520*SdS55b_2@SdS55b_2
###############################################################################


print('#'*79,'\nTest asymetric CTMRG')

d = 5
D = 2
chi = 3

tensors = [tAKLT2.copy()]
tiling = """
A
"""

tensors = [tAKLT2.copy(),tAKLT2.copy(),tAKLT2.copy()]
tiling = """
ABC
BCA
CAB
"""

tensors = [tAKLT2.copy(),tAKLT2_conj.copy()]
tiling = """
AB
BA
"""


ctm = CTMRG(tensors,tiling,chi,verbosity=0)
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
ctm.chi = 10
ctm.iterate()
print('done')

ctm.chi = 6
for i in range(20):
  ctm.iterate()

rdm1x1 = ctm.compute_rdm1x1()
rdm1x2 = ctm.compute_rdm1x2()
rdm2x1 = ctm.compute_rdm2x1()
rdm2x2 = ctm.compute_rdm2x2()

print(f'rdm1x1 is hermitian: {lg.norm(rdm1x1-rdm1x1.T.conj()):.1e}')
print(f'rdm1x2 is hermitian: {lg.norm(rdm1x2-rdm1x2.T.conj()):.1e}')
print(f'rdm2x1 is hermitian: {lg.norm(rdm2x1-rdm2x1.T.conj()):.1e}')
print(f'rdm2x2 is hermitian: {lg.norm(rdm2x2-rdm2x2.T.conj()):.1e}')

print(f'trace(H_AKLT_55b @ rdm1x2) = {np.trace(H_AKLT_55b @ rdm1x2):.1e}')
print(f'trace(H_AKLT_55b @ rdm2x1) = {np.trace(H_AKLT_55b @ rdm2x1):.1e}')
print(f'trace(H_AKLT @ rdm1x2) = {np.trace(H_AKLT @ rdm1x2):.1e}')
print(f'trace(H_AKLT @ rdm2x1) = {np.trace(H_AKLT @ rdm2x1):.1e}')

T1s = ctm._env._neq_T1s
print('T1s are all the same:', end=' ')
for t in T1s[1:]:
  print(f'{lg.norm(t-T1s[0]):.1e}', end=', ')

T2s = ctm._env._neq_T2s
print('\nT2s are all the same:', end=' ')
for t in T2s[1:]:
  print(f'{lg.norm(t-T2s[0]):.1e}', end=', ')

T3s = ctm._env._neq_T3s
print('\nT3s are all the same:', end=' ')
for t in T3s[1:]:
  print(f'{lg.norm(t-T3s[0]):.1e}', end=', ')

T4s = ctm._env._neq_T4s
print('\nT4s are all the same:', end=' ')
for t in T4s[1:]:
  print(f'{lg.norm(t-T4s[0]):.1e}', end=', ')

C1s = ctm._env._neq_C1s
print('\nC1s are all the same:', end=' ')
for t in C1s[1:]:
  print(f'{lg.norm(t-C1s[0]):.1e}', end=', ')

C2s = ctm._env._neq_C2s
print('\nC2s are all the same:', end=' ')
for t in C2s[1:]:
  print(f'{lg.norm(t-C2s[0]):.1e}', end=', ')

C3s = ctm._env._neq_C3s
print('\nC3s are all the same:', end=' ')
for t in C3s[1:]:
  print(f'{lg.norm(t-C3s[0]):.1e}', end=', ')

C4s = ctm._env._neq_C4s
print('\nC4s are all the same:', end=' ')
for t in C4s[1:]:
  print(f'{lg.norm(t-C4s[0]):.1e}', end=', ')

print('\n'+'#'*79)
