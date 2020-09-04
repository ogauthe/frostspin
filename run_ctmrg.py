#! /usr/bin/env python3
import numpy as np
import scipy.linalg as lg
from CTMRG import CTMRG
from test_tools import *
from time import time


print('#'*79,'\nTest asymetric CTMRG')

chi = 8

tilingA = """
A
"""
tilingAB = """
AB
BA
"""
tilingABC = """
ABC
BCA
CAB
"""


til9 = """
ABC
DEF
GHI
"""

"""
tiling = tilingAB
tensors = [tAKLT2.copy(),tAKLT2.copy()]
#tensors = [tRVB2.copy()]
A = np.random.random((6,9,4,5,2,3))
B = np.random.random((7,8,2,3,4,5))
tensors = [A,B]
tiling = tilingAB

A = np.random.random((10,11,2,3,4,5))
B = np.random.random((12,13,4,6,7,3))
C = np.random.random((8,9,7,5,2,6))
tensors = [A,B,C]
tiling = tilingABC



for i in range(1,19):
  Ds = 2*np.ones(19,dtype=np.int8)
  Ds[i] = 3
  print('#'*79,f"\ni = {i}, Ds = {Ds}")

  A = np.random.random((3,1,Ds[1],Ds[2],Ds[3],Ds[4]))
  B = np.random.random((5,1,Ds[5],Ds[6],Ds[7],Ds[2]))
  C = np.random.random((2,7,Ds[8],Ds[4],Ds[9],Ds[6]))
  D = np.random.random((7,1,Ds[3],Ds[10],Ds[11],Ds[12]))
  E = np.random.random((3,2,Ds[7],Ds[14],Ds[15],Ds[10]))
  F = np.random.random((3,5,Ds[9],Ds[12],Ds[13],Ds[14]))
  G = np.random.random((3,7,Ds[11],Ds[16],Ds[1],Ds[17]))
  H = np.random.random((7,3,Ds[15],Ds[18],Ds[5],Ds[16]))
  I = np.random.random((5,3,Ds[13],Ds[17],Ds[8],Ds[18]))


  ctm = CTMRG(chi, tensors=(A,B,C,D,E,F,G,H,I), tiling=til9)
  ctm.print_tensor_shapes()
  for i in range(5):
    ctm.iterate()
  rdm1x1 = ctm.compute_rdm1x1()
  rdm1x2 = ctm.compute_rdm1x2()
  rdm2x1 = ctm.compute_rdm2x1()
  rdm2x2 = ctm.compute_rdm2x2()

  print(f'rdm1x1 is hermitian: {lg.norm(rdm1x1-rdm1x1.T.conj()):.1e}')
  print(f'rdm1x2 is hermitian: {lg.norm(rdm1x2-rdm1x2.T.conj()):.1e}')
  print(f'rdm2x1 is hermitian: {lg.norm(rdm2x1-rdm2x1.T.conj()):.1e}')
  print(f'rdm2x2 is hermitian: {lg.norm(rdm2x2-rdm2x2.T.conj()):.1e}')


ctm = CTMRG(chi, tensors=tensors, tiling=tiling, verbosity=2)
ctm.print_tensor_shapes()
print('try 2 iterations:')
ctm.iterate()
print("#"*40,'  1st done  ',"#"*40,sep="")
ctm.iterate()
print("#"*40,'  2nd done, change chi  ',"#"*35,sep="")
T1s = ctm._env._neq_T1s
ctm.chi = 10
ctm.iterate()
print("#"*40,'  3rd done  ',"#"*40,sep="")
ctm.iterate()
print("#"*40,'  4th done, change chi  ',"#"*40,sep="")

ctm.chi = 6
for i in range(5):
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
"""


###############################################################################
print('\n'+'#'*30 + 'RVB tensor' + '#'*30)

chi = 70
niter = 0

tiling = tilingA
ctm = CTMRG(chi, tensors=(tRVB2,), tiling='A')
print(f'chi = {chi}, measure symmetry and Heisenberg term for rdm1x2/2x1')
rdm1x2 = ctm.compute_rdm1x2()
rdm2x1 = ctm.compute_rdm2x1()
print(f'{lg.norm(rdm1x2 - rdm1x2.T):.1e}, {lg.norm(rdm2x1 - rdm2x1.T):.1e}, {np.trace(SdS_22b @ rdm1x2):.4e}, {np.trace(SdS_22b @ rdm2x1):.4e}')

niter = 200
t = time()
for i in range(niter):
  ctm.iterate()
  rdm1x2 = ctm.compute_rdm1x2()
  rdm2x1 = ctm.compute_rdm2x1()
  print(f'{lg.norm(rdm1x2 - rdm1x2.T):.1e}, {lg.norm(rdm2x1 - rdm2x1.T):.1e}, {np.trace(SdS_22b @ rdm1x2):.4e}, {np.trace(SdS_22b @ rdm2x1):.4e}')

t = time() - t
print(f'{niter} iter done, time = {t}')

