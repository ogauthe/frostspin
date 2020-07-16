import numpy as np
import scipy.linalg as lg

def svd_SU(M_A, M_B, lambda_dir, gate, D_cst, d, D_dir):
  """
  utilitary function
  Construct matrix theta from tensors M_A and M_B, where lambda have been added
  to gammaA and gammaB. Cut it using SVD.
  """

  # 1) SVD cut between constant tensors and effective tensor to update
  # hence reduce main SVD to dimension D_eff*d < D_cst*d
  #     \|        \|
  #     -A-    -> -W==M-
  #      |\        |   \
  M_A = M_A.reshape(D_cst, d*D_dir)
  W_A, sA, M_A = lg.svd(M_A, full_matrices=False)
  D_effA = len(sA)
  M_A *= sA[:,None]
  M_B = M_B.reshape(D_dir*d, D_cst)
  M_B, sB, W_B = lg.svd(M_B, full_matrices=False)
  D_effB = len(sB)
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

  def __init__(self, shA, shB, shC, shD, h1, h2, tau, tensors=None):
    """
    Simple update algorithm on plaquette AB//CD.

    Parameters
    ----------
    shA, shB, shC, shD : tuple of 6 ints.
      Shape of tensors A,B,C and D, with convention (d,a,Du,Dr,Dd,Dl), where
      a=1 for a pure wavefunction and a=d for a thermal ensemble
    h1 : (d**2,d**2) float or complex ndarray
      first neigbor Hamltionian
    h2 : (d**2,d**2) float or complex ndarray
      second neigbor Hamltionian
    tau : imaginary time step
    tensors : otional, enumerable of 4 tensors with shapes shA,shB,shC,shD.
      Starting tensors. If not provided, random tensors are taken.
    """

    self._d, self._a, self._Du, self._Dr, self._Dd, self._Dl = sh

    if self._Du*self._d > self._a*self._Dr*self._Dd*self._Dl:
      raise ValueError('up bond: D_eff > D_cst, cannot reshape')
    if self._Dr*self._d > self._a*self._Du*self._Dd*self._Dl:
      raise ValueError('right bond: D_eff > D_cst, cannot reshape')
    if self._Dd*self._d > self._a*self._Du*self._Dr*self._Dl:
      raise ValueError('down bond: D_eff > D_cst, cannot reshape')
    if self._Dl*self._d > self._a*self._Du*self._Dr*self._Dd:
      raise ValueError('left bond: D_eff > D_cst, cannot reshape')

    if tensors is not None:
      A0,B0,C0,D0 = tensors
      if A0.shape != shA:
        raise ValueError("invalid shape for A0")
      if B0.shape != shB:
        raise ValueError("invalid shape for A0")
      if C0.shape != shC:
        raise ValueError("invalid shape for C0")
      if D0.shape != shD:
        raise ValueError("invalid shape for D0")
    else:
      A0 = np.random.random(shA)
      B0 = np.random.random(shB)
      C0 = np.random.random(shC)
      D0 = np.random.random(shD)
    self._gammaA /= A0/lg.norm(A0)
    self._gammaB = B0/lg.norm(B0)
    self._gammaC /= C0/lg.norm(C0)
    self._gammaD = D0/lg.norm(D0)

    # hamilt can be either 1 unique numpy array for all bonds or a list/tuple
    # of 4 bond-dependant Hamiltonians
    if h1.shape != (self._d**2, self._d**2)
        raise ValueError('invalid shape for Hamiltonian h1')
    if h2.shape != (self._d**2, self._d**2)
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
    self._g2 = lg.expm(-tau*self._h2)


  def get_ABCD(self):
    """
    return optimized tensors A, B, C and D
    Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
    """
    sl1 = np.sqrt(self._lambda_1)
    sl2 = np.sqrt(self._lambda_2)
    sl3 = np.sqrt(self._lambda_3)
    sl4 = np.sqrt(self._lambda_4)
    sl5 = np.sqrt(self._lambda_5)
    sl6 = np.sqrt(self._lambda_6)
    sl7 = np.sqrt(self._lambda_7)
    sl8 = np.sqrt(self._lambda_8)
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
    self.update_up()
    self.update_right()
    self.update_down()
    self.update_left()


  def update_right(self):
    """
    update lambda_r by applying gate gr to right link
    """
    # add diagonal lambdas to gammaA and gammaB
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda_u, self._lambda_d, self._lambda_l)
    M_B = np.einsum('paurdl,u,r,d->lpaurd', self._gammaB, self._lambda_d, self._lambda_l, self._lambda_u)

    # construct matrix theta, renormalize bond dimension and get back tensors
    M_A, self._lambda_r, M_B = svd_SU(M_A, M_B, self._lambda_r, self._gr, self._a*self._Du*self._Dd*self._Dl, self._d, self._Dr)

    # define new gammaA and gammaB from renormalized M_A and M_B
    M_A = M_A.reshape(self._a, self._Du, self._Dd, self._Dl, self._d, self._Dr)
    self._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, self._lambda_u**-1, self._lambda_d**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Dr, self._d, self._a, self._Dd, self._Dl, self._Du)
    self._gammaB = np.einsum('lpaurd,u,r,d->paurdl', M_B, self._lambda_d**-1, self._lambda_l**-1, self._lambda_u**-1)


  def update_down(self):
    """
    update lambda_d by applying gate gd to down bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,u,r,l->aurlpd', self._gammaA, self._lambda_u, self._lambda_r, self._lambda_l)
    M_B = np.einsum('paurdl,r,d,l->upardl', self._gammaB, self._lambda_l, self._lambda_u, self._lambda_r)
    M_A, self._lambda_d, M_B = svd_SU(M_A, M_B, self._lambda_d, self._gd, self._a*self._Du*self._Dr*self._Dl, self._d, self._Dd)
    M_A = M_A.reshape(self._a, self._Du, self._Dr, self._Dl, self._d, self._Dd)
    self._gammaA = np.einsum('aurlpd,u,r,l->paurdl', M_A, self._lambda_u**-1, self._lambda_r**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Dd, self._d, self._a, self._Dl, self._Du, self._Dr)
    self._gammaB = np.einsum('upardl,r,d,l->paurdl', M_B, self._lambda_l**-1, self._lambda_u**-1, self._lambda_r**-1)


  def update_left(self):
    """
    update lambda_l by applying gate gl to left bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,u,r,d->aurdpl', self._gammaA, self._lambda_u, self._lambda_r, self._lambda_d)
    M_B = np.einsum('paurdl,u,d,l->rpaudl', self._gammaB, self._lambda_d, self._lambda_u, self._lambda_r)
    M_A, self._lambda_l, M_B = svd_SU(M_A, M_B, self._lambda_l, self._gl, self._a*self._Du*self._Dr*self._Dd, self._d, self._Dl)
    M_A = M_A.reshape(self._a, self._Du, self._Dr, self._Dd, self._d, self._Dl)
    self._gammaA = np.einsum('aurdpl,u,r,d->paurdl', M_A, self._lambda_u**-1, self._lambda_r**-1, self._lambda_d**-1)
    M_B = M_B.reshape(self._Dl, self._d, self._a, self._Dd, self._Du, self._Dr)
    self._gammaB = np.einsum('rpaudl,u,d,l->paurdl', M_B, self._lambda_d**-1, self._lambda_u**-1, self._lambda_r**-1)


  def update_up(self):
    """
    update lambda_u by applying gate gu to up bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,r,d,l->ardlpu', self._gammaA, self._lambda_r, self._lambda_d, self._lambda_l)
    M_B = np.einsum('paurdl,u,r,l->dpaurl', self._gammaB, self._lambda_d, self._lambda_l, self._lambda_r)
    M_A, self._lambda_u, M_B = svd_SU(M_A, M_B, self._lambda_u, self._gu, self._a*self._Dr*self._Dd*self._Dl, self._d, self._Du)
    M_A = M_A.reshape(self._a, self._Dr, self._Dd, self._Dl, self._d, self._Du)
    self._gammaA = np.einsum('ardlpu,r,d,l->paurdl', M_A, self._lambda_r**-1, self._lambda_d**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Du, self._d, self._a, self._Dd, self._Dl, self._Dr)
    self._gammaB = np.einsum('dpaurl,u,r,l->paurdl', M_B, self._lambda_d**-1, self._lambda_l**-1, self._lambda_r**-1)
