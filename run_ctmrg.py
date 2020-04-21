#! /usr/bin/env python3
import numpy as np
import scipy.linalg as lg
from CTMRG import CTMRG

d = 5
D = 2
chi = 3

tiling = """
ABC
BCA
CAB
"""

tensors = [np.ones((5,2,2,2,2)),np.ones((5,2,2,2,2)),np.ones((5,2,2,2,2))]
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

