#! /usr/bin/env python3
import numpy as np
import scipy.linalg as lg
from CTMRG import CTMRG
from test_tools import *


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



###############################################################################
print('\n'+'#'*30 + 'RVB tensor' + '#'*30)

chi = 17
niter = 10
tensors = [tRVB2]
tiling = """
A
"""

ctm = CTMRG(tensors,tiling,chi,verbosity=0)
print(f'chi = {chi}, launch {niter} iterations')
for i in range(niter):
  ctm.iterate()

print('measure AF Heisenberg energy')
for i in range(10):
  ctm.iterate()
  rdm1x2 = ctm.compute_rdm1x2()
  rdm2x1 = ctm.compute_rdm2x1()
  print(f'trace(SdS_22b @ rdm1x2) = {np.trace(SdS_22b @ rdm1x2):.5e}')
  print(f'trace(SdS_22b @ rdm2x1) = {np.trace(SdS_22b @ rdm2x1):.5e}')

