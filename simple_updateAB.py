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


class SimpleUpdateAB(object):

  def __init__(self, sh, hamilt, tau, A0=None, B0=None):
    """
    sh: tuple of int, shape of tensor A.
    form (d,a,Du,Dr,Dd,Dl), where a=1 for pure wavefunction and a=d for thermal TN
    shape of B is then (d,a,Dd,Dl,Du,Dr)
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

    if A0 is not None:
      if A0.shape != sh:
        raise ValueError("invalid shape for A0")
      self._gammaA = A0
    else:
      self._gammaA = np.random.random(sh)
    if B0 is not None:
      if B0.shape != (self._d,self._a,self._Dd,self._Dl,self._Du,self._Dr):
        raise ValueError("invalid shape for B0")
      self._gammaB = B0
    else:
      self._gammaB = np.random.random((self._d,self._a,self._Dd,self._Dl,self._Du,self._Dr))
    self._gammaA = self._gammaA/lg.norm(self._gammaA)
    self._gammaB = self._gammaB/lg.norm(self._gammaB)

    # hamilt can be either 1 unique numpy array for all bonds or a list/tuple
    # of 4 bond-dependant Hamiltonians
    shH = (self._d**2, self._d**2)
    if type(hamilt) == list or type(hamilt) == tuple:
      self._hamilt = None
      self._hu, self._hr, self._hd, self._hl = hamilt
      if self._hu.shape != shH:
        raise ValueError('invalid shape for up Hamilt')
      if self._hr.shape != shH:
        raise ValueError('invalid shape for right Hamilt')
      if self._hd.shape != shH:
        raise ValueError('invalid shape for down Hamilt')
      if self._hl.shape != shH:
        raise ValueError('invalid shape for left Hamilt')
    else:
      if hamilt.shape != shH:
        raise ValueError('invalid shape for Hamiltonian')
      self._hamilt = hamilt

    self.tau = tau
    self._lambda_u = np.ones(self._Du)
    self._lambda_r = np.ones(self._Dr)
    self._lambda_d = np.ones(self._Dd)
    self._lambda_l = np.ones(self._Dl)


  @tau.setter
  def tau(self, tau):
    self._tau = tau
    if self._hamilt is not None:
      g = lg.expm(-tau*self._hamilt)
      self._gu, self._gr, self._gd, self._gl = [g]*4
    else:
      self._gu = lg.expm(-tau*self._hu)
      self._gr = lg.expm(-tau*self._hr)
      self._gd = lg.expm(-tau*self._hd)
      self._gl = lg.expm(-tau*self._hl)


  def get_AB(self):
    """
    return optimized tensors A and B.
    A and B are obtained by adding relevant sqrt(lambda) to every leg of gammaA
    and gammaB
    """
    u = np.sqrt(self._lambda_u)
    r = np.sqrt(self._lambda_r)
    d = np.sqrt(self._lambda_d)
    l = np.sqrt(self._lambda_l)
    A = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaA,u,r,d,l)
    B = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaB,d,l,u,r)
    A /= lg.norm(A)
    B /= lg.norm(B)
    return A,B


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
