import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB

sh = ((2,2,6,4,5,3))
gates = [np.random.random((4,4))]*4
su = SimpleUpdateAB(sh,gates)

su.update()
su.update()
