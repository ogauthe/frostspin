import numpy as np
import scipy.linalg as lg

def svd_SU(M_A, M_B, lambda_dir, gate, d):
  """
  utilitary function
  Construct matrix theta from tensors M_A and M_B, where lambda have been added
  to gammaA and gammaB. Cut it using SVD.
  """

  # 1) SVD cut between constant tensors and effective tensor to update
  # hence reduce main SVD to dimension D*d < a*D**3
  #     \|        \|
  #     -A-    -> -W==M-
  #      |\        |   \
  D_dir = lambda_dir.shape[0]
  W_A, sA, M_A = lg.svd(M_A, full_matrices=False)
  D_effA = sA.shape[0]
  M_A *= sA[:,None]
  M_B, sB, W_B = lg.svd(M_B, full_matrices=False)
  D_effB = sB.shape[0]
  M_B *= sB

  # 2) construct matrix theta with gate g
  #
  #             =MA-lr-MB=
  #                \  /
  #   theta =       gg
  #                /  \
  theta = M_A.reshape(D_effA*d, D_dir)
  theta *= lambda_dir
  theta = np.dot(theta, M_B.reshape(D_dir, d*D_effB) )
  theta = theta.reshape(D_effA, d, d, D_effB).transpose(0,3,1,2).reshape(D_effA*D_effB, d**2)
  theta = np.dot(theta, gate)

  # 3) cut theta with SVD
  theta = theta.reshape(D_effA, D_effB, d, d).swapaxes(1,2).reshape(D_effA*d, D_effB*d)
  M_A,s,M_B = lg.svd(theta)

  # 4) renormalize link dimension
  s = s[:D_dir]
  s /= s.sum()  # singular values are positive

  # 5) start reconstruction of new gammaA and gammaB by unifying cst and eff
  M_A = M_A[:,:D_dir].reshape(D_effA, d*D_dir)
  M_A = np.dot(W_A, M_A)
  M_B = M_B[:D_dir].reshape(D_dir,D_effB,d).swapaxes(1,2).reshape(D_dir*d, D_effB)
  M_B = np.dot(M_B, W_B)
  return M_A, s, M_B


class SimpleUpdateABCD(object):

  def __init__(self, d, a, Ds, h1, h2, tau, tensors=None):
    """
    Simple update algorithm on plaquette AB//CD.

    Parameters
    ----------
    d: integer
      Dimension of physical leg.
    a: integer
      Dimension of ancila leg. a=1 for a pure wavefunction and a=d for a
      thermal ensemble.
    Ds: enumerable of 8 int
      Dimension of 8 non-equivalent bonds
    h1 : (d**2,d**2) float or complex ndarray
      First neigbor Hamltionian.
    h2 : (d**2,d**2) float or complex ndarray
      Second neigbor Hamltionian.
    tau : float
      Imaginary time step.
    tensors : otional, enumerable of 4 ndarrays with shapes (d,a,D,D,D,D)
      Initial tensors. If not provided, random tensors are taken.
    """

    # allowing for different D on each bond is uncommon but allows easy debug.
    # No need for different d and a.
    if d > a*min(Ds)**2:   # not sufficient: need D_eff < D_ini for every bond.
      raise ValueError('D_eff > D_cst, cannot reshape in first SVD')

    self._d = d
    self._a = a
    self._D1, self._D2, self._D3, self._D4, self._D5, self._D6, self._D7, self._D8 = Ds

    if tensors is not None:
      A0,B0,C0,D0 = tensors
      if A0.shape != (d,a,self._D1,self._D2,self._D3,self._D4):
        raise ValueError("invalid shape for A0")
      if B0.shape != (d,a,self._D5,self._D4,self._D6,self._D2):
        raise ValueError("invalid shape for B0")
      if C0.shape != (d,a,self._D3,self._D8,self._D1,self._D7):
        raise ValueError("invalid shape for C0")
      if D0.shape != (d,a,self._D6,self._D7,self._D5,self._D8):
        raise ValueError("invalid shape for D0")
    else:
      A0 = np.random.random((d,a,self._D1,self._D2,self._D3,self._D4)) - 0.5
      B0 = np.random.random((d,a,self._D5,self._D4,self._D6,self._D2)) - 0.5
      C0 = np.random.random((d,a,self._D3,self._D8,self._D1,self._D7)) - 0.5
      D0 = np.random.random((d,a,self._D6,self._D7,self._D5,self._D8)) - 0.5
    self._gammaA /= lg.norm(A0)
    self._gammaB /= lg.norm(B0)
    self._gammaC /= lg.norm(C0)
    self._gammaD /= lg.norm(D0)

    # hamilt can be either 1 unique numpy array for all bonds or a list/tuple
    # of 4 bond-dependant Hamiltonians
    if h1.shape != (self._d**2, self._d**2):
        raise ValueError('invalid shape for Hamiltonian h1')
    if h2.shape != (self._d**2, self._d**2):
        raise ValueError('invalid shape for Hamiltonian h2')
    self._h1 = h1
    self._h2 = h2

    self.tau = tau
    self._lambda1 = np.ones(self._D1)
    self._lambda2 = np.ones(self._D2)
    self._lambda2 = np.ones(self._D3)
    self._lambda4 = np.ones(self._D4)
    self._lambda5 = np.ones(self._D5)
    self._lambda6 = np.ones(self._D6)
    self._lambda7 = np.ones(self._D7)
    self._lambda8 = np.ones(self._D8)


  @property
  def tau(self):
    return self._tau

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    self._g1 = lg.expm(-tau*self._h1)
    self._g2 = lg.expm(-tau/2*self._h2)  # apply twice sqrt(gate)

  @property
  def d(self):
    return self._d

  @property
  def a(self):
    return self._a

  @property
  def Ds(self):
    return (self._D1, self._D2, self._D3, self._D4, self._D5, self._D6, self._D7, self._D8)


  def get_ABCD(self):
    """
    return optimized tensors A, B, C and D
    Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
    """
    sl1 = np.sqrt(self._lambda1)
    sl2 = np.sqrt(self._lambda2)
    sl3 = np.sqrt(self._lambda3)
    sl4 = np.sqrt(self._lambda4)
    sl5 = np.sqrt(self._lambda5)
    sl6 = np.sqrt(self._lambda6)
    sl7 = np.sqrt(self._lambda7)
    sl8 = np.sqrt(self._lambda8)
    A = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaA,sl1,sl2,sl3,sl4)
    B = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaB,sl5,sl4,sl6,sl2)
    C = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaC,sl3,sl8,sl1,sl7)
    D = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaD,sl6,sl7,sl5,sl8)
    A /= lg.norm(A)
    B /= lg.norm(B)
    C /= lg.norm(C)
    D /= lg.norm(D)
    return A,B,C,D


  def update(self):
    """
    update all links
    """
    # TODO: do not add and remove lambdas every time, keep some
    # TODO: merge all directional functions, fine dealing with lambda
    # goes to 2nd order Trotter by reversing order, tau twice bigger for gl
    # and call once. Wait for working ctm to test.
    self.update_bond1()   # AC up
    self.update_bond2()   # AB left
    self.update_bond3()   # AC down
    self.update_bond4()   # AB right
    self.update_bond5()   # BD up
    self.update_bond6()   # BD down
    self.update_bond7()   # CD left
    self.update_bond8()   # CD right

    # link AD right up
    self.update_bonds25() # through B
    self.update_bonds18() # through C
    # link AD right down
    self.update_bonds26() # through B
    self.update_bonds38() # through C
    # link AD left down
    self.update_bonds46() # through B
    self.update_bonds37() # through C
    # link AD left up
    self.update_bonds45() # through B
    self.update_bonds17() # through C

    # link BC right up
    self.update_bonds41() # through A
    self.update_bonds57() # through D
    # link BC right down
    self.update_bonds43() # through A
    self.update_bond67() # through D
    # link BC left down
    self.update_bonds23() # through A
    self.update_bonds68() # through D
    # link BC left up
    self.update_bonds21() # through A
    self.update_bonds58() # through D


  def update_bond1(self):
    """
    update lambda1 between A and C by applying gate g1 to A upper bond
    """
    # add diagonal weights to gammaA and gammaC
    M_A = np.einsum('paurdl,r,d,l->ardlpu', self._gammaA, self._lambda2, self._lambda3,
                    self._lambda4).reshape(self._a*self._D2*self._D3*self._D4, self._d*self._D1)
    M_C = np.einsum('paurdl,u,r,l->dpaurl', self._gammaC, self._lambda3, self._lambda8,
                    self._lambda7).reshape(self._D1*self._d, self._a*self._D3*self._D8*self._D7)

    # construct matrix theta, renormalize bond dimension and get back tensors
    M_A, self._lambda1, M_C = svd_SU(M_A, M_C, self._lambda1, self._g1, self._d)

    # define new gammaA and gammaC from renormalized M_A and M_C
    M_A = M_A.reshape(self._a, self._D2, self._D3, self._D4, self._d, self._D1)
    self._gammaA = np.einsum('ardlpu,r,d,l->paurdl', M_A, self._lambda2**-1, self._lambda3**-1, self._lambda4**-1)
    M_C = M_C.reshape(self._D1, self._d, self._a, self._D3, self._D8, self._D7)
    self._gammaC = np.einsum('dpaurl,u,r,l->paurdl', M_C, self._lambda3**-1, self._lambda8**-1, self._lambda7**-1)


  def update_bond2(self):
    """
    update lambda2 between A and B by applying gate to A right bond
    """
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda1, self._lambda3,
                    self._lambda4).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = np.einsum('paurdl,u,r,d->lpaurd', self._gammaB, self._lambda5, self._lambda4,
                    self._lambda6).reshape(self._D2*self._d, self._a*self._D5*self._D4*self._D6)

    M_A, self._lambda2, M_B = svd_SU(M_A, M_B, self._lambda2, self._g1, self._d)

    M_A = M_A.reshape(self._a, self._D1, self._D3, self._D4, self._d, self._D2)
    self._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, self._lambda1**-1, self._lambda3**-1, self._lambda3**-1)
    M_B = M_B.reshape(self._Dr, self._d, self._a, self._Dd, self._Dl, self._Du)
    self._gammaB = np.einsum('lpaurd,u,r,d->paurdl', M_B, self._lambda5**-1, self._lambda4**-1, self._lambda6**-1)


  def update_bond3(self):
    """
    update lambda3 between A and C by applying gate to A down bond
    """
    M_A = np.einsum('paurdl,u,r,l->aurlpd', self._gammaA, self._lambda1, self._lambda2,
                    self._lambda4).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D3)
    M_C = np.einsum('paurdl,r,d,l->upardl', self._gammaC, self._lambda8, self._lambda1,
                    self._lambda7).reshape(self._D3*self._d, self._a*self._D8*self._D1*self._D7)

    M_A, self._lambda3, M_B = svd_SU(M_A, M_C, self._lambda3, self._g1, self._d)

    M_A = M_A.reshape(self._a, self._D1, self._D2, self._D4, self._d, self._D3)
    self._gammaA = np.einsum('aurlpd,u,r,l->paurdl', M_A, self._lambda1**-1, self._lambda2**-1, self._lambda4**-1)
    M_C = M_C.reshape(self._D3, self._d, self._a, self._D8, self._D1, self._D7)
    self._gammaC = np.einsum('upardl,r,d,l->paurdl', M_C, self._lambda8**-1, self._lambda1**-1, self._lambda7**-1)


  def update_bond4(self):
    """
    update lambda4 between A and B by applying gate to A right bond
    """
    M_A = np.einsum('paurdl,u,r,d->aurdpl', self._gammaA, self._lambda1, self._lambda2,
                    self._lambda3).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D4)
    M_B = np.einsum('paurdl,u,d,l->rpaudl', self._gammaB, self._lambda5, self._lambda6,
                    self._lambda2).reshape(self._D4*self._d, self._a*self._D5*self._D6*self._D2)

    M_A, self._lambda4, M_B = svd_SU(M_A, M_B, self._lambda4, self._g1, self._d)
    M_A = M_A.reshape(self._a, self._D1, self._D2, self._D3, self._d, self._D4)
    self._gammaA = np.einsum('aurdpl,u,r,d->paurdl', M_A, self._lambda1**-1, self._lambda2**-1, self._lambda3**-1)
    M_B = M_B.reshape(self._D4, self._d, self._a, self._D5, self._D6, self._D2)
    self._gammaB = np.einsum('rpaudl,u,d,l->paurdl', M_B, self._lambda5**-1, self._lambda6**-1, self._lambda2**-1)


