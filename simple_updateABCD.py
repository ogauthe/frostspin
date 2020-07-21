import numpy as np
import scipy.linalg as lg

def update_first_neighbor(M0_L, M0_R, lambda0, gate, d):
  """
  First neighbor simple update algorithm.
  Construct matrix theta from matrices M0_L and M0_R, obtained by adding
  diagonal weights to non-updated bonds and reshaping to matrices initial
  tensors gammaX and gammaY

  For clarity, a 1D geometry
  =M0_L -- lambda0 -- M_R0=
  is considered in the notations, but the function is direction-agnostic.
  """

  # 1) SVD cut between constant tensors and effective tensor to update
  # hence reduce main SVD to dimension D*d < a*D**3
  #     \|        \|
  #     -L-    -> -W==ML-
  #      |\        |   \
  D = lambda0.shape[0]
  W_L, sL, M_L = lg.svd(M0_L, full_matrices=False)
  D_effL = sL.shape[0]
  M_L *= sL[:,None]
  #     \|            \|
  #     -R-    -> -MR==W-
  #      |\        /   |
  M_R, sR, W_R = lg.svd(M0_R, full_matrices=False)
  D_effR = sR.shape[0]
  M_R *= sR

  # 2) construct matrix theta with gate g
  #
  #             =ML-l-MR=
  #                \  /
  #   theta =       gg
  #                /  \
  theta = M_L.reshape(D_effL*d, D)*lambda0
  theta = np.dot(theta, M_R.reshape(D, d*D_effR) )
  theta = theta.reshape(D_effL, d, d, D_effR).transpose(0,3,1,2).reshape(D_effL*D_effR, d**2)
  theta = np.dot(theta, gate)

  # 3) cut theta with SVD
  theta = theta.reshape(D_effL, D_effR, d, d).swapaxes(1,2).reshape(D_effL*d, D_effR*d)
  M_L, new_lambda, M_R = lg.svd(theta, full_matrices=False)

  # 4) renormalize link dimension
  new_lambda = new_lambda[:D]
  new_lambda /= new_lambda.sum()  # singular values are positive

  # 5) start reconstruction of new gammaX and gammaY by unifying cst and eff parts
  M_L = M_L[:,:D].reshape(D_effL, d*D)
  M_L = np.dot(W_L, M_L)
  M_R = M_R[:D].reshape(D, D_effR, d).swapaxes(1,2).reshape(D*d, D_effR)
  M_R = np.dot(M_R, W_R)
  return M_L, new_lambda, M_R


def update_second_neighbor(M0_L, M0_mid, M0_R, lambda_L, lambda_R, gate, d):
  """
  Second and third neighbor simple update algorithm.
  Construct matrix theta from matrices M0_L, M0_mid and M0_R obtained by adding
  diagonal weights to non-updated bonds and reshaping to matrices initial
  tensors gammaX, gammaY and gammaZ.

  For clarity, a 1D geometry
  M_left -- lambda_left -- M_mid -- lambda_right -- M_right
  is considered in the notations, but the function is direction-agnostic and
  works for any geometry with two extremity tensors linked by a middle one.
  """

  D_L, D_R = lambda_L.shape[0], lambda_R.shape[0]

  # 1) SVD cut between constant tensors and effective tensors to update
  #     \|        \|
  #     -L-    -> -cstL==effL-lambda_L- (D_L)
  #      |\        |       \
  cst_L, s_L, eff_L = lg.svd(M0_L, full_matrices=False)
  D_effL = s_L.shape[0]
  eff_L = (s_L[:,None]*eff_L).reshape(D_effL*d, D_L)*lambda_L  # add lambda
  #                       \|/|
  #                       cstM
  #     \|                 ||
  #     -M-   ->  (D_L) - effM - (D_R)
  #      |\
  eff_m, s_m, cst_m = lg.svd(M0_mid, full_matrices=False)
  D_effm = s_m.shape[0]
  eff_m = (eff_m*s_m).reshape(D_L, D_R*D_effm)
  #     \|                              \|
  #     -R-   ->  (D_R)  lambda_R-effR==cstR
  #      |\                              |\
  eff_R, s_R, cst_R = lg.svd(M0_R, full_matrices=False)
  D_effR = s_R.shape[0]
  eff_R = (eff_R*s_R).reshape(D_R, d, D_effR)*lambda_R[:,None,None]

  # contract tensor network
  #                         ||
  #    =effL-lambdaL -- eff_mid -- lambdaR-effR=
  #         \                             /
  #          \----------- gate ----------/
  theta = np.dot(eff_L, eff_m).reshape(D_effL, d, D_R, D_effm)
  theta = np.tensordot(theta, eff_R, ((2,),(0,)))
  theta = theta.transpose(0,2,4,1,3).reshape(D_effL*D_effm*D_effR, d**2)
  theta = np.dot(theta, g2).reshape(D_effL, D_effm, D_effR, d, d)
  theta = theta.transpose(0,3,1,2,4).reshape(D_effL*d, D_effm*D_effR*d)

  # first SVD: cut left part
  new_L, new_lambda_L, theta = lg.svd(theta, full_matrices=False)
  new_L = newL[:,:D_L].reshape(D_effL, d*D_L)
  new_lambda_L = new_lambda_L[:D_L]
  new_lambda_L /= new_lambda_L.sum()

  # second SVD: split middle and right parts
  theta = (new_lambda_L[:,None]*theta[:D_L]).reshape(D_L*D_effm, D_effR*d)
  new_mid, new_lambda_R, new_R = lg.svd(theta, full_matrices=False)
  new_mid = new_mid[:,:D_R].reshape(D_L, D_effm, D_R)
  new_R = new_R[:D_R].reshape(D_R, D_eff_R, d)
  new_lambda_R = new_lambda_R[:D_R]
  new_lambda_R /= new_lambda_R.sum()

  # bring back constant parts
  new_L = np.dot(cst_L, new_L)
  new_mid = new_mid.swapaxes(1,2).reshape(D_L*D_R, D_effm)
  new_mid = np.dot(new_mid, cst_m)
  new_R = new_R.swapaxes(1,2).reshape(D_R*d, D_effR)
  new_R = np.dot(new_R, cst_R)

  return new_L, new_mid, new_R, new_lambda_L, new_lambda_R


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


    Conventions for leg ordering
    ----------------------------
          1     5
          |     |
       4--A--2--B--4
          |     |
          3     6
          |     |
       8--C--7--D--8
          |     |
          1     5
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
      if C0.shape != (d,a,self._D3,self._D7,self._D1,self._D8):
        raise ValueError("invalid shape for C0")
      if D0.shape != (d,a,self._D6,self._D8,self._D5,self._D7):
        raise ValueError("invalid shape for D0")
    else:
      A0 = np.random.random((d,a,self._D1,self._D2,self._D3,self._D4)) - 0.5
      B0 = np.random.random((d,a,self._D5,self._D4,self._D6,self._D2)) - 0.5
      C0 = np.random.random((d,a,self._D3,self._D7,self._D1,self._D8)) - 0.5
      D0 = np.random.random((d,a,self._D6,self._D8,self._D5,self._D7)) - 0.5
    self._gammaA = A0/lg.norm(A0)
    self._gammaB = B0/lg.norm(B0)
    self._gammaC = C0/lg.norm(C0)
    self._gammaD = D0/lg.norm(D0)

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

  @property
  def h1(self):
    return self._h1

  @property
  def h2(self):
    return self._h2


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
    C = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaC,sl3,sl7,sl1,sl8)
    D = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaD,sl6,sl8,sl5,sl7)
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
    self.update_bonds17() # through C
    # link AD right down
    self.update_bonds26() # through B
    self.update_bonds37() # through C
    # link AD left down
    self.update_bonds46() # through B
    self.update_bonds38() # through C
    # link AD left up
    self.update_bonds45() # through B
    self.update_bonds18() # through C

    # link BC right up
    self.update_bonds41() # through A
    self.update_bonds58() # through D
    # link BC right down
    self.update_bonds43() # through A
    self.update_bonds68() # through D
    # link BC left down
    self.update_bonds23() # through A
    self.update_bonds67() # through D
    # link BC left up
    self.update_bonds21() # through A
    self.update_bonds57() # through D


###############################################################################
# first neighbor updates
###############################################################################
  def update_bond1(self):
    """
    update lambda1 between A and C by applying gate g1 to A upper bond
    """
    # add diagonal weights to gammaA and gammaC
    M_A = np.einsum('paurdl,r,d,l->ardlpu', self._gammaA, self._lambda2, self._lambda3,
                    self._lambda4).reshape(self._a*self._D2*self._D3*self._D4, self._d*self._D1)
    M_C = np.einsum('paurdl,u,r,l->dpaurl', self._gammaC, self._lambda3, self._lambda7,
                    self._lambda8).reshape(self._D1*self._d, self._a*self._D3*self._D7*self._D8)

    # construct matrix theta, renormalize bond dimension and get back tensors
    M_A, self._lambda1, M_C = update_first_neighbor(M_A, M_C, self._lambda1, self._g1, self._d)

    # define new gammaA and gammaC from renormalized M_A and M_C
    M_A = M_A.reshape(self._a, self._D2, self._D3, self._D4, self._d, self._D1)
    self._gammaA = np.einsum('ardlpu,r,d,l->paurdl', M_A, self._lambda2**-1, self._lambda3**-1, self._lambda4**-1)
    M_C = M_C.reshape(self._D1, self._d, self._a, self._D3, self._D7, self._D8)
    self._gammaC = np.einsum('dpaurl,u,r,l->paurdl', M_C, self._lambda3**-1, self._lambda7**-1, self._lambda8**-1)


  def update_bond2(self):
    """
    update lambda2 between A and B by applying gate to A right bond
    """
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda1, self._lambda3,
                    self._lambda4).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = np.einsum('paurdl,u,r,d->lpaurd', self._gammaB, self._lambda5, self._lambda4,
                    self._lambda6).reshape(self._D2*self._d, self._a*self._D5*self._D4*self._D6)

    M_A, self._lambda2, M_B = update_first_neighbor(M_A, M_B, self._lambda2, self._g1, self._d)

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
    M_C = np.einsum('paurdl,r,d,l->upardl', self._gammaC, self._lambda7, self._lambda1,
                    self._lambda8).reshape(self._D3*self._d, self._a*self._D7*self._D1*self._D8)

    M_A, self._lambda3, M_B = update_first_neighbor(M_A, M_C, self._lambda3, self._g1, self._d)

    M_A = M_A.reshape(self._a, self._D1, self._D2, self._D4, self._d, self._D3)
    self._gammaA = np.einsum('aurlpd,u,r,l->paurdl', M_A, self._lambda1**-1, self._lambda2**-1, self._lambda4**-1)
    M_C = M_C.reshape(self._D3, self._d, self._a, self._D7, self._D1, self._D8)
    self._gammaC = np.einsum('upardl,r,d,l->paurdl', M_C, self._lambda7**-1, self._lambda1**-1, self._lambda8**-1)


  def update_bond4(self):
    """
    update lambda4 between A and B by applying gate to A right bond
    """
    M_A = np.einsum('paurdl,u,r,d->aurdpl', self._gammaA, self._lambda1, self._lambda2,
                    self._lambda3).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D4)
    M_B = np.einsum('paurdl,u,d,l->rpaudl', self._gammaB, self._lambda5, self._lambda6,
                    self._lambda2).reshape(self._D4*self._d, self._a*self._D5*self._D6*self._D2)

    M_A, self._lambda4, M_B = update_first_neighbor(M_A, M_B, self._lambda4, self._g1, self._d)
    M_A = M_A.reshape(self._a, self._D1, self._D2, self._D3, self._d, self._D4)
    self._gammaA = np.einsum('aurdpl,u,r,d->paurdl', M_A, self._lambda1**-1, self._lambda2**-1, self._lambda3**-1)
    M_B = M_B.reshape(self._D4, self._d, self._a, self._D5, self._D6, self._D2)
    self._gammaB = np.einsum('rpaudl,u,d,l->paurdl', M_B, self._lambda5**-1, self._lambda6**-1, self._lambda2**-1)


  def update_bond5(self):
    """
    update lambda5 between B and D by applying gate to B upper bond
    """
    M_B = np.einsum('paurdl,r,d,l->ardlpu', self._gammaB, self._lambda4, self._lambda6,
                    self._lambda2).reshape(self._a*self._D4*self._D6*self._D2, self._d*self._D5)
    M_D = np.einsum('paurdl,u,r,l->dpaurl', self._gammaD, self._lambda6, self._lambda8,
                    self._lambda7).reshape(self._D5*self._d, self._a*self._D6*self._D8*self._D7)

    M_B, self._lambda5, M_D = update_first_neighbor(M_B, M_D, self._lambda5, self._g1, self._d)
    M_B = M_B.reshape(self._a, self._D4, self._D6, self._D2, self._d, self._D5)
    self._gammaB = np.einsum('ardlpu,r,d,l->paurdl', M_B, self._lambda4**-1, self._lambda6**-1, self._lambda2**-1)
    M_D = M_D.reshape(self._D5, self._d, self._a, self._D6, self._D8, self._D7)
    self._gammaD = np.einsum('dpaurl,u,r,l->paurdl', M_D, self._lambda6**-1, self._lambda8**-1, self._lambda7**-1)


  def update_bond6(self):
    """
    update lambda6 between B and D by applying gate to B down bond
    """
    M_B = np.einsum('paurdl,u,r,l->aurlpd', self._gammaB, self._lambda5, self._lambda4,
                    self._lambda2).reshape(self._a*self._D5*self._D4*self._D2, self._d*self._D6)
    M_D = np.einsum('paurdl,r,d,l->upardl', self._gammaD, self._lambda8, self._lambda5,
                    self._lambda7).reshape(self._D6*self._d, self._a*self._D8*self._D5*self._D7)

    M_B, self._lambda6, M_D = update_first_neighbor(M_B, M_D, self._lambda6, self._g1, self._d)
    M_B = M_B.reshape(self._a, self._D5, self._D5, self._D2, self._d, self._D6)
    self._gammaB = np.einsum('aurlpd,u,r,l->paurdl', M_B, self._lambda5**-1, self._lambda4**-1, self._lambda2**-1)
    M_D = M_D.reshape(self._D6, self._d, self._a, self._D8, self._D5, self._D7)
    self._gammaD = np.einsum('upardl,r,d,l->paurdl', M_D, self._lambda8**-1, self._lambda5**-1, self._lambda7**-1)


  def update_bond7(self):
    """
    update lambda7 between C and D by applying gate to C right bond
    """
    M_C = np.einsum('paurdl,u,d,l->audlpr', self._gammaC, self._lambda3, self._lambda1,
                    self._lambda8).reshape(self._a*self._D3*self._D1*self._D8, self._d*self._D7)
    M_D = np.einsum('paurdl,u,r,d->lpaurd', self._gammaD, self._lambda6, self._lambda8,
                    self._lambda5).reshape(self._D7*self._d, self._a*self._D6*self._D8*self._D5)

    M_C, self._lambda7, M_D = update_first_neighbor(M_C, M_D, self._lambda7, self._g1, self._d)
    M_C = M_C.reshape(self._a, self._D3, self._D1, self._D8, self._d, self._D7)
    self._gammaC = np.einsum('audlpr,u,d,l->paurdl', M_C, self._lambda3**-1, self._lambda1**-1, self._lambda8**-1)
    M_D = M_D.reshape(self._D7, self._d, self._a, self._D6, self._D8, self._D5)
    self._gammaD = np.einsum('lpaurd,u,r,d->paurdl', M_D, self._lambda6**-1, self._lambda8**-1, self._lambda5**-1)


  def update_bond8(self):
    """
    update lambda8 between C and D by applying gate to C left bond
    """
    M_C = np.einsum('paurdl,u,r,d->aurdpl', self._gammaC, self._lambda3, self._lambda7,
                    self._lambda1).reshape(self._a*self._D3*self._D7*self._D1, self._d*self._D8)
    M_D = np.einsum('paurdl,u,d,l->rpaudl', self._gammaD, self._lambda6, self._lambda5,
                    self._lambda7).reshape(self._D8*self._d, self._a*self._D6*self._D5*self._D7)

    M_C, self._lambda8, M_D = update_first_neighbor(M_C, M_D, self._lambda8, self._g1, self._d)
    M_C = M_C.reshape(self._a, self._D3, self._D7, self._D1, self._d, self._D8)
    self._gammaC = np.einsum('aurdpl,u,r,d->paurdl', M_C, self._lambda3**-1, self._lambda7**-1, self._lambda1**-1)
    M_D = M_D.reshape(self._D8, self._d, self._a, self._D6, self._D5, self._D7)
    self._gammaD = np.einsum('rpaudl,u,d,l->paurdl', M_D, self._lambda6**-1, self._lambda5**-1, self._lambda7**-1)


###############################################################################
# second neighbor updates
###############################################################################

  def update_bonds26(self):
    """
    update lambda2 and lambda6 by applying gate to A down-right next nearest
    neighbor bond with D through tensor B.
    """
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda1, self._lambda3,
                    self._lambda4).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = np.einsum('paurdl,u,r->ldpaur', self._gammaB, self._lambda5, self._lambda4).reshape(
                                           self._D2*self._D6, self._d*self._a*self._D5*self._D4)
    M_D = np.einsum('paurdl,r,d,l->upardl', self._gammaD, self._lambda8, self._lambda5,
                    self._lambda7).reshape(self._D6*self._d, self._a*self._D8*self._D5*self._D7)
    M_A, M_B, M_D, self._lambda2, self._lambda_6 = update_second_neighbor(
                                       M_A, M_B, M_D, self._lambda2, self._lambda6, g2, self._d)
    M_A = M_A.reshape(self._a, self._D1, self._D3, self._D4, self._d, self._D2)
    self._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, self._lambda1**-1, self._lambda3**-1, self._lambda4**-1)
    M_B = M_B.reshape(self._D2, self._D6, self._d, self._a, self._D5, self._D4)
    self._gammaB = np.einsum('ldpaur,u,r->paurdl', M_B, self._lambda5**-1, self._lambda4**-1)
    M_D = M_D.reshape(self._D6, self._d, self._a, self._D8, self._D5, self._D7)
    self._gammaD = np.einsum('upardl,r,d,l->paurdl', M_D, self._lambda8**-1, self._lambda5**-1, self._lambda7**-1)


