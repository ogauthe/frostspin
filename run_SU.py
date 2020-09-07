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
beta = 1
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

colors = [pcol,pcol,vcol[:2],vcol[:3],vcol[:4],vcol[:5],vcol[:6],vcol[:7],vcol[:8],vcol[:9]]
Ds = (2,3,4,5,6,7,8,9)
A = A[:,:,:2,:3,:4,:5]
B = B[:,:,:6,:5,:7,:3]
C = C[:,:,:4,:8,:2,:9]
D = D[:,:,:7,:9,:6,:8]

h1 = SdS_22b
h2 = J2*SdS_22
# colors A and D = [[1,-1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]
# opposed colors on B and C physical leg
# colors B and C = [[-1,1], [1,-1],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0],[2, 0, -2, 0, 0]]

print("#"*79)
print("Run simple update for spin 1/2 Heisenberg with J1=1, J2={J2}")
print(f"run with tau = {tau} up to beta = {beta}")
su = SimpleUpdateABCD(d, a, Ds, h1, h2, tau, tensors=(A,B,C,D), colors=colors)

t = time()
#for i in range(su_iter//2): # 2nd order Trotter
for i in range(2):
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


print("done")

