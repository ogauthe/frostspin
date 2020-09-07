#! /usr/bin/env python3

import numpy as np
import scipy.linalg as lg
from time import time
from simple_updateABCD import SimpleUpdateABCD
from test_tools import SdS_22, SdS_22b
from CTMRG import CTMRG
from toolsU1 import checkU1, combine_colors


tiling = 'AB\nCD'

d = 2
a = 2
chi = 35
ctm_iter = 5
D = 9 # 3+1+1 at start
Ds = (D,)*8
print(f'run simple update with d = {d}, a = {a}, Ds = {Ds}')

J2 = 0.9
beta = 1.
tau = 0.01
su_iter = int(beta/2/tau)  # rho is quadratic in psi

# zero singular values lead to divide by zero when removing weights
# to avoid this, start with tensors made of decoupled product of (2x2)->1 on
# physical x ancila and 3^4 -> 1 (B1) on virtual legs.
p22_3333 = np.zeros((2,2,3,3,3,3),dtype=np.int8)
p22_3333.flat[[89,95,103,105,115,119,123,127,137,139,147,153,170,176,184,186,
               196,200,204,208,218,220,228,234]] = [1,-1,1,-1,-1,1,1,-1,-1,1,
                                 -1,1,-1,1,-1,1,1,-1,-1, 1,1,-1, 1,-1]

pcol = np.array([1,-1],dtype=np.int8)
vcol = np.array([2,0,-2] +[0]*(D-3),dtype=np.int8)
colors = [pcol,pcol,vcol,vcol,vcol,vcol,vcol,vcol,vcol,vcol]
A = np.zeros((d,a,D,D,D,D))
A[:,:,:3,:3,:3,:3] = p22_3333
B = A.copy()
C = B.copy()
D = A.copy()

h1 = SdS_22b
h2 = J2*SdS_22
# colors A and D = [[1,-1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]
# opposed colors on B and C physical leg
# colors B and C = [[-1,1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]

print("#"*79)
print(f"Run simple update for spin 1/2 Heisenberg with J1=1, J2={J2}")
print(f"run with tau = {tau} up to beta = {beta}")
su = SimpleUpdateABCD(d, a, Ds, h1, h2, tau, tensors=(A,B,C,D), colors=colors)

t = time()
for i in range(su_iter//2): # 2nd order Trotter
#for i in range(2):
  su.update()
print(f"\ndone with SU, t={time()-t:.1f}")

A,B,C,D = su.get_ABCD()
colors = su.colors
colorsA = [colors[0],colors[1],colors[2],colors[3],colors[4],colors[5]]
colorsB = [-colors[0],-colors[1],-colors[6],-colors[5],-colors[7],-colors[3]]
colorsC = [-colors[0],-colors[1],-colors[4],-colors[8],-colors[2],-colors[9]]
colorsD = [colors[0],colors[1],colors[7],colors[9],colors[6],colors[8]]

ctm = CTMRG(chi, tensors=(A,B,C,D), tiling=tiling, colors=(colorsA,colorsB,colorsC,colorsD), verbosity=0)

print(f'Converge CTMRG with chi = {chi} and niter = {ctm_iter}')
t = time()
for i in range(ctm_iter):
  print(i, end=" ")
  ctm.iterate()

print(f"\ndone with CTM iteration, t={time()-t:.1f}")


h1_4sites = np.kron(h1,np.eye(d**2)).reshape( (d,)*8)
h2_4sites = np.kron(h2,np.eye(d**2)).reshape( (d,)*8)
sh = (d**4,d**4)
# rho ABCD has NN bonds 2,3,6,7 and NNN bonds 26/37
h_ABCD = (  h1_4sites.reshape(sh)   # 0-1
         + h1_4sites.transpose(0,2,1,3,4,6,5,7).reshape(sh)        # 0//2
         + h1_4sites.transpose(3,1,2,0,7,5,6,4).reshape(sh)        # 1//3
         + h1_4sites.transpose(2,3,0,1,6,7,4,5).reshape(sh) )/2 \
       + h2_4sites.transpose(0,3,2,1,4,7,6,5).reshape(sh)       \
       + h2_4sites.transpose(2,1,0,3,6,5,4,7).reshape(sh)          # 1/-2


# coordinates stand for C1.
print("compute 4 rdm 2x2")
t = time()
rho_ABCD = ctm.compute_rdm2x2(-1,-1)
rho_BADC = ctm.compute_rdm2x2(0,-1)
rho_CDAB = ctm.compute_rdm2x2(-1,0)
rho_DCBA = ctm.compute_rdm2x2(0,0)
print(f"done, t = {time()-t:.1f}")

# Tr(AB) = (A*B.T).sum() and H is exactly symmetric (not just up to precision)
energy = ((rho_ABCD + rho_BADC + rho_CDAB + rho_DCBA)*h_ABCD).sum()/4
print(f"energy = {energy}")

save = f"data_ctm_SU_ABCD_J2_{J2}_beta{beta}_tau{tau}_chi{chi}.npz"
ctm.save_to_file(save)
print("CTMRG data saved in file", save)
