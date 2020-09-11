import numpy as np
import scipy.linalg as lg
from toolsU1 import default_color, dotU1, combine_colors, svdU1, checkU1, tensordotU1

# more convenient than [:,None,None,None,None,None] everywhere?
_sh = (-1,1,1,1,1,1)

def update_first_neighbor(M0_L, M0_R, lambda0, gate, d, col_L=default_color,
             col_R=default_color, col_bond=default_color, col_d=default_color):
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
  D = lambda0.size
  W_L, sL, M_L, col_sL = svdU1(M0_L, col_L, -combine_colors(col_d, col_bond))
  D_effL = sL.size
  M_L *= sL[:,None]
  #     \|            \|
  #     -R-    -> -MR==W-
  #      |\        /   |
  M_R, sR, W_R, col_sR = svdU1(M0_R, combine_colors(-col_bond, -col_d), col_R)
  D_effR = sR.size
  M_R *= sR

  # 2) construct matrix theta with gate g
  #
  #             =ML-l-MR=
  #                \  /
  #   theta =       gg
  #                /  \
  row_col = combine_colors(col_sL, col_d)
  col_gate = combine_colors(-col_d, col_d)  # opposite sublattice
  theta = M_L.reshape(D_effL*d, D)*lambda0
  theta = dotU1(theta, M_R.reshape(D, d*D_effR), row_col, col_bond, combine_colors(-col_d,-col_sR))
  theta = theta.reshape(D_effL, d, d, D_effR).transpose(0,3,1,2).reshape(D_effL*D_effR, d**2)
  theta = dotU1(theta, gate, combine_colors(col_sL,-col_sR), -col_gate, -col_gate)

  # 3) cut theta with SVD
  theta = theta.reshape(D_effL, D_effR, d, d).swapaxes(1,2).reshape(D_effL*d, D_effR*d)
  M_L, new_lambda, M_R, new_col_lambda = svdU1(theta,row_col,-combine_colors(-col_sR,-col_d))

  # 4) renormalize link dimension
  new_lambda = new_lambda[:D]
  new_lambda /= new_lambda.sum()  # singular values are positive
  new_col_lambda = -new_col_lambda[:D]

  # 5) start reconstruction of new gammaX and gammaY by unifying cst and eff parts
  M_L = (M_L[:,:D]*new_lambda).reshape(D_effL, d*D)
  M_L = dotU1(W_L, M_L, col_L, -col_sL, combine_colors(col_d,new_col_lambda))
  M_R = (M_R[:D]*new_lambda[:,None]).reshape(D, D_effR, d).swapaxes(1,2).reshape(D*d, D_effR)
  M_R = dotU1(M_R, W_R, combine_colors(-new_col_lambda,-col_d), -col_sR, -col_R)
  return M_L, new_lambda, M_R, new_col_lambda


def update_second_neighbor(M0_L, M0_mid, M0_R, lambda_L, lambda_R, gate, d,
             col_L=default_color, col_mid=default_color, col_R=default_color,
             col_bL=default_color, col_bR=default_color, col_d=default_color):
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

  D_L, D_R = lambda_L.size, lambda_R.size

  # 1) SVD cut between constant tensors and effective tensors to update
  #     \|        \|
  #     -L-    -> -cstL==effL-lambda_L- (D_L)
  #      |\        |       \
  cst_L, s_L, eff_L, col_sL = svdU1(M0_L, col_L, -combine_colors(col_d, col_bL))
  D_effL = s_L.size
  eff_L = (s_L[:,None]*eff_L).reshape(D_effL*d, D_L)*lambda_L  # add lambda
  #                       \|/|
  #                       cstM
  #     \|                 ||
  #     -M-   ->  (D_L) - effM - (D_R)
  #      |\
  eff_m, s_m, cst_m, col_sm = svdU1(M0_mid, -combine_colors(col_bL,col_bR), col_mid)
  D_effm = s_m.size
  eff_m = (eff_m*s_m).reshape(D_L, D_R*D_effm)
  #     \|                              \|
  #     -R-   ->  (D_R)  lambda_R-effR==cstR
  #      |\                              |\
  col_effR = combine_colors(col_bR, col_d)
  eff_R, s_R, cst_R, col_sR = svdU1(M0_R, -col_effR, col_R)
  D_effR = s_R.size
  eff_R = (eff_R*s_R).reshape(D_R, d, D_effR)*lambda_R[:,None,None]

  # contract tensor network
  #                         ||
  #    =effL-lambdaL -- eff_mid -- lambdaR-effR=
  #         \                             /
  #          \----------- gate ----------/
  theta = dotU1(eff_L, eff_m, combine_colors(col_sL,col_d), col_bL, -combine_colors(col_bR,col_sm))
  theta = theta.reshape(D_effL, d, D_R, D_effm)
  col_th = (col_sL, col_d, -col_bR, -col_sm)
  col_gate = combine_colors(col_d, col_d)  # same sublattice
  theta = tensordotU1(theta, eff_R, (2,), (0,), col_th, (col_bR, col_d, col_sR))
  theta = theta.transpose(0,2,4,1,3).reshape(D_effL*D_effm*D_effR, d**2)
  col_th = combine_colors(col_sL, -col_sm, col_sR)
  theta = dotU1(theta, gate, col_th, col_gate, col_gate).reshape(D_effL, D_effm, D_effR, d, d)
  theta = theta.transpose(0,3,1,2,4).reshape(D_effL*d, D_effm*D_effR*d)

  # first SVD: cut left part
  new_L, new_lambda_L, theta, col_nbL = svdU1(theta, combine_colors(col_sL, col_d), -combine_colors(-col_sm, col_sR, col_d))
  new_lambda_L = new_lambda_L[:D_L]
  new_lambda_L /= new_lambda_L.sum()
  col_nbL = col_nbL[:D_L]
  new_L = (new_L[:,:D_L]*new_lambda_L).reshape(D_effL, d*D_L)

  # second SVD: split middle and right parts
  theta = (new_lambda_L[:,None]*theta[:D_L]).reshape(D_L*D_effm, D_effR*d)
  col_th = combine_colors(col_nbL,-col_sm)
  new_mid, new_lambda_R, new_R, col_nbR = svdU1(theta, col_th, -combine_colors(col_sR,col_d))
  new_lambda_R = new_lambda_R[:D_R]
  new_lambda_R /= new_lambda_R.sum()
  col_nbR = col_nbR[:D_R]
  new_mid = new_mid[:,:D_R].reshape(D_L, D_effm, D_R)*new_lambda_R
  new_R = new_R[:D_R].reshape(D_R, D_effR, d)*new_lambda_R[:,None,None]

  # bring back constant parts
  new_L = dotU1(cst_L, new_L, col_L, -col_sL, combine_colors(col_d,-col_nbL))
  new_mid = new_mid.swapaxes(1,2).reshape(D_L*D_R, D_effm)
  new_mid = dotU1(new_mid, cst_m, combine_colors(col_nbL, -col_nbR), -col_sm, -col_mid)
  new_R = new_R.swapaxes(1,2).reshape(D_R*d, D_effR)
  new_R = dotU1(new_R, cst_R, combine_colors(col_nbR, col_d), col_sR, col_R)

  return new_L, new_mid, new_R, new_lambda_L, new_lambda_R, -col_nbL, col_nbR


class SimpleUpdate2x2(object):

  def __init__(self, d, a, Ds, h1, h2, tau, tensors=None, colors=None,
                verbosity=0):
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
    tensors : optional, enumerable of 4 ndarrays with shapes (d,a,D,D,D,D)
      Initial tensors. If not provided, random tensors are taken.
    colors : optional, quantum numbers for physical, ancila and virtual legs.
    verbosity : int
      level of log verbosity. Default is no log.


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
    self.verbosity = verbosity
    if self.verbosity > 0:
      print(f'construct SimpleUpdataABCD with d = {d}, a = {a} and Ds = {Ds}')

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

    if colors is not None:
      if len(colors) != 10:
        raise ValueError('colors must be [colors_p, colors_a, colors_1...8]')
      if len(colors[0]) != d:
        raise ValueError('physical leg colors length is not d')
      self._colors_p = np.asarray(colors[0], dtype=np.int8)
      if len(colors[1]) != a:
        raise ValueError('ancila leg colors length is not a')
      self._colors_a = np.asarray(colors[1], dtype=np.int8)
      if len(colors[2]) != self._D1:
        raise ValueError('virtual leg 1 colors length is not D1')
      self._colors1 = np.asarray(colors[2], dtype=np.int8)
      if len(colors[3]) != self._D2:
        raise ValueError('virtual leg 2 colors length is not D2')
      self._colors2 = np.asarray(colors[3], dtype=np.int8)
      if len(colors[4]) != self._D3:
        raise ValueError('virtual leg 3 colors length is not D3')
      self._colors3 = np.asarray(colors[4], dtype=np.int8)
      if len(colors[5]) != self._D4:
        raise ValueError('virtual leg 4 colors length is not D4')
      self._colors4 = np.asarray(colors[5], dtype=np.int8)
      if len(colors[6]) != self._D5:
        raise ValueError('virtual leg 5 colors length is not D5')
      self._colors5 = np.asarray(colors[6], dtype=np.int8)
      if len(colors[7]) != self._D6:
        raise ValueError('virtual leg 6 colors length is not D6')
      self._colors6 = np.asarray(colors[7], dtype=np.int8)
      if len(colors[8]) != self._D7:
        raise ValueError('virtual leg 7 colors length is not D7')
      self._colors7 = np.asarray(colors[8], dtype=np.int8)
      if len(colors[9]) != self._D8:
        raise ValueError('virtual leg 8 colors length is not D8')
      self._colors8 = np.asarray(colors[9], dtype=np.int8)

    else:
      self._colors_p = default_color
      self._colors_a = default_color
      self._colors1 = default_color
      self._colors2 = default_color
      self._colors3 = default_color
      self._colors4 = default_color
      self._colors5 = default_color
      self._colors6 = default_color
      self._colors7 = default_color
      self._colors8 = default_color

    if h1.shape != (self._d**2, self._d**2):
        raise ValueError('invalid shape for Hamiltonian h1')
    if h2.shape != (self._d**2, self._d**2):
        raise ValueError('invalid shape for Hamiltonian h2')
    self._h1 = h1
    self._h2 = h2

    self.tau = tau
    self._lambda1 = np.ones(self._D1)
    self._lambda2 = np.ones(self._D2)
    self._lambda3 = np.ones(self._D3)
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
    if self.verbosity > 0:
      print(f'set tau to  {tau}')
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

  @property
  def colors(self):
    """
    Tuple
    U(1) quantum numbers for non-equivalent legs
    Convention: sort as ((physical,ancila),leg_i) to have colors[i] = color_i.
    """
    return ( (self._colors_p, self._colors_a), self._colors1, self._colors2,
            self._colors3, self._colors4, self._colors5, self._colors6,
            self._colors7, self._colors8)

  @property
  def lambdas(self):
    """
    Tuple
    Simple update weights.
    Convention: return ((None,None),leg_i) to be consistent with colors.
    """
    return ( (None,None), self._lambda1, self._lambda2,
            self._lambda3, self._lambda4, self._lambda5, self._lambda6,
            self._lambda7, self._lambda8)


  def get_ABCD(self):
    """
    return optimized tensors A, B, C and D
    Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
    """
    sl1 = 1/np.sqrt(self._lambda1)
    sl2 = 1/np.sqrt(self._lambda2)
    sl3 = 1/np.sqrt(self._lambda3)
    sl4 = 1/np.sqrt(self._lambda4)
    sl5 = 1/np.sqrt(self._lambda5)
    sl6 = 1/np.sqrt(self._lambda6)
    sl7 = 1/np.sqrt(self._lambda7)
    sl8 = 1/np.sqrt(self._lambda8)
    A = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaA,sl1,sl2,sl3,sl4)
    B = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaB,sl5,sl4,sl6,sl2)
    C = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaC,sl3,sl7,sl1,sl8)
    D = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaD,sl6,sl8,sl5,sl7)
    A /= lg.norm(A)
    B /= lg.norm(B)
    C /= lg.norm(C)
    D /= lg.norm(D)
    return A,B,C,D


  def update_first_neighbor(self):
    """
    update all first neighbor links
    """
    if self.verbosity > 0:
      print('launch first neighbor update')
    self.update_bond1()   # AC up
    self.update_bond2()   # AB left
    self.update_bond3()   # AC down
    self.update_bond4()   # AB right
    self.update_bond5()   # BD up
    self.update_bond6()   # BD down
    self.update_bond7()   # CD left
    self.update_bond8()   # CD right


  def update_first_neighbor_reverse(self):
    """
    update all first neighbor links, reverse order
    """
    if self.verbosity > 0:
      print('launch reverse first neighbor update')
    self.update_bond8()   # CD right
    self.update_bond7()   # CD left
    self.update_bond6()   # BD down
    self.update_bond5()   # BD up
    self.update_bond4()   # AB right
    self.update_bond3()   # AC down
    self.update_bond2()   # AB left
    self.update_bond1()   # AC up


  def update_second_neighbor(self):
    """
    update all first neighbor links
    """
    if self.verbosity > 0:
      print('launch second neighbor update')
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


  def update_second_neighbor_reverse(self):
    """
    update all first neighbor links
    """
    if self.verbosity > 0:
      print('launch reverse second neighbor update')
    self.update_bonds57() # through D
    self.update_bonds21() # through A
    self.update_bonds67() # through D
    self.update_bonds23() # through A
    self.update_bonds68() # through D
    self.update_bonds43() # through A
    self.update_bonds58() # through D
    self.update_bonds41() # through A
    self.update_bonds18() # through C
    self.update_bonds45() # through B
    self.update_bonds38() # through C
    self.update_bonds46() # through B
    self.update_bonds37() # through C
    self.update_bonds26() # through B
    self.update_bonds17() # through C
    self.update_bonds25() # through B


  def update(self):
    """
    2nd order Trotter Suzuki step on all bonds
    """
    if self.verbosity > 0:
      print('launch 2nd order update for all bonds')
    # goes to 2nd order Trotter by reversing update order
    self.update_first_neighbor()
    self.update_second_neighbor()
    self.update_second_neighbor_reverse()
    self.update_first_neighbor_reverse()

###############################################################################
# first neighbor updates
###############################################################################
# due to pi-roation on B and C sites, colors are opposed on those sites
# the colors returned by update_neighbors need to be reversed
# mc{i} = minus color i

  def update_bond1(self):
    """
    update lambda1 between A and C by applying gate g1 to A upper bond
    """
    # add diagonal weights to gammaA and gammaC
    w = self._lambda1**-1
    M_A = (self._gammaA.transpose(1,3,4,5,0,2)*w).reshape(self._a*self._D2*self._D3*self._D4, self._d*self._D1)
    M_C = (w.reshape(_sh)*self._gammaC.transpose(4,0,1,2,3,5)).reshape(self._D1*self._d, self._a*self._D3*self._D7*self._D8)

    col_L = combine_colors(self._colors_a, self._colors2, self._colors3, self._colors4)
    col_R = combine_colors(self._colors_a, self._colors3, self._colors7, self._colors8)
    # construct matrix theta, renormalize bond dimension and get back tensors
    M_A, self._lambda1, M_C, self._colors1 = update_first_neighbor(M_A, M_C, self._lambda1, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=self._colors1, col_d=self._colors_p)

    # define new gammaA and gammaC from renormalized M_A and M_C
    self._gammaA = M_A.reshape(self._a, self._D2, self._D3, self._D4, self._d, self._D1).transpose(4,0,5,1,2,3)
    self._gammaC = M_C.reshape(self._D1, self._d, self._a, self._D3, self._D7, self._D8).transpose(1,2,3,4,0,5)

    if self.verbosity > 1:
      print('updated bond 1: new lambda1 =', self._lambda1)


  def update_bond2(self):
    """
    update lambda2 between A and B by applying gate to A right bond
    """
    w = self._lambda2**-1
    M_A = (self._gammaA.transpose(1,2,4,5,0,3)*w).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = (w.reshape(_sh)*self._gammaB.transpose(5,0,1,2,3,4)).reshape(self._D2*self._d, self._a*self._D5*self._D4*self._D6)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors3, self._colors4)
    col_R = combine_colors(self._colors_a, self._colors5, self._colors4, self._colors6)
    M_A, self._lambda2, M_B, self._colors2 = update_first_neighbor(M_A, M_B, self._lambda2, self._g1, self._d,
          col_L=col_L, col_R=col_R, col_bond=self._colors2, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D3, self._D4, self._d, self._D2).transpose(4,0,1,5,2,3)
    self._gammaB = M_B.reshape(self._D2, self._d, self._a, self._D5, self._D4, self._D6).transpose(1,2,3,4,5,0)

    if self.verbosity > 1:
      print('updated bond 2: new lambda2 =', self._lambda2)


  def update_bond3(self):
    """
    update lambda3 between A and C by applying gate to A down bond
    """
    w = self._lambda3**-1
    M_A = (self._gammaA.transpose(1,2,3,5,0,4)*w).reshape(self._a*self._D1*self._D2*self._D4, self._d*self._D3)
    M_C = (w.reshape(_sh)*self._gammaC.transpose(2,0,1,3,4,5)).reshape(self._D3*self._d, self._a*self._D7*self._D1*self._D8)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors4)
    col_R = combine_colors(self._colors_a, self._colors7, self._colors1, self._colors8)
    M_A, self._lambda3, M_C, self._colors3 = update_first_neighbor(M_A, M_C, self._lambda3, self._g1, self._d,
         col_L=col_L, col_R=col_R, col_bond=self._colors3, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D4, self._d, self._D3).transpose(4,0,1,2,5,3)
    self._gammaC = M_C.reshape(self._D3, self._d, self._a, self._D7, self._D1, self._D8).transpose(1,2,0,3,4,5)

    if self.verbosity > 1:
      print('updated bond 3: new lambda3 =', self._lambda3)


  def update_bond4(self):
    """
    update lambda4 between A and B by applying gate to A right bond
    """
    w = self._lambda4**-1
    M_A = (self._gammaA.transpose(1,2,3,4,0,5)*w).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D4)
    M_B = (w.reshape(_sh)*self._gammaB.transpose(3,0,1,2,4,5)).reshape(self._D4*self._d, self._a*self._D5*self._D6*self._D2)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors3)
    col_R = combine_colors(self._colors_a, self._colors5, self._colors6, self._colors2)
    M_A, self._lambda4, M_B, self._colors4 = update_first_neighbor(M_A, M_B, self._lambda4, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=self._colors4, col_d=self._colors_p)
    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D3, self._d, self._D4).transpose(4,0,1,2,3,5)
    self._gammaB = M_B.reshape(self._D4, self._d, self._a, self._D5, self._D6, self._D2).transpose(1,2,3,0,4,5)

    if self.verbosity > 1:
      print('updated bond 4: new lambda4 =', self._lambda4)


  def update_bond5(self):
    """
    update lambda5 between B and D by applying gate to B upper bond
    """
    w = self._lambda5**-1
    M_B = (self._gammaB.transpose(1,3,4,5,0,2)*w).reshape(self._a*self._D4*self._D6*self._D2, self._d*self._D5)
    M_D = (w.reshape(_sh)*self._gammaD.transpose(4,0,1,2,3,5)).reshape(self._D5*self._d, self._a*self._D6*self._D8*self._D7)

    col_L = -combine_colors(self._colors_a, self._colors4, self._colors6, self._colors2)
    col_R = -combine_colors(self._colors_a, self._colors6, self._colors8, self._colors7)
    M_B, self._lambda5, M_D, mc5 = update_first_neighbor(M_B, M_D, self._lambda5, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=-self._colors5, col_d=-self._colors_p)
    self._colors5 = -mc5 # B-D
    self._gammaB = M_B.reshape(self._a, self._D4, self._D6, self._D2, self._d, self._D5).transpose(4,0,5,1,2,3)
    self._gammaD = M_D.reshape(self._D5, self._d, self._a, self._D6, self._D8, self._D7).transpose(1,2,3,4,0,5)

    if self.verbosity > 1:
      print('updated bond 5: new lambda5 =', self._lambda5)


  def update_bond6(self):
    """
    update lambda6 between B and D by applying gate to B down bond
    """
    w = self._lambda6**-1
    M_B = (self._gammaB.transpose(1,2,3,5,0,4)*w).reshape(self._a*self._D5*self._D4*self._D2, self._d*self._D6)
    M_D = (w.reshape(_sh)*self._gammaD.transpose(2,0,1,3,4,5)).reshape(self._D6*self._d, self._a*self._D8*self._D5*self._D7)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors4, self._colors2)
    col_R = -combine_colors(self._colors_a, self._colors8, self._colors5, self._colors7)
    M_B, self._lambda6, M_D, mc6 = update_first_neighbor(M_B, M_D, self._lambda6, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=-self._colors6, col_d=-self._colors_p)
    self._colors6 = -mc6   # B-D
    self._gammaB = M_B.reshape(self._a, self._D5, self._D4, self._D2, self._d, self._D6).transpose(4,0,1,2,5,3)
    self._gammaD = M_D.reshape(self._D6, self._d, self._a, self._D8, self._D5, self._D7).transpose(1,2,0,3,4,5)

    if self.verbosity > 1:
      print('updated bond 6: new lambda6 =', self._lambda6)


  def update_bond7(self):
    """
    update lambda7 between C and D by applying gate to C right bond
    """
    w = self._lambda7**-1
    M_C = (self._gammaC.transpose(1,2,4,5,0,3)*w).reshape(self._a*self._D3*self._D1*self._D8, self._d*self._D7)
    M_D = (w.reshape(_sh)*self._gammaD.transpose(5,0,1,2,3,4)).reshape(self._D7*self._d, self._a*self._D6*self._D8*self._D5)

    col_L = -combine_colors(self._colors_a, self._colors3, self._colors1, self._colors8)
    col_R = -combine_colors(self._colors_a, self._colors6, self._colors8, self._colors5)
    M_C, self._lambda7, M_D, mc7 = update_first_neighbor(M_C, M_D, self._lambda7, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=-self._colors7, col_d=-self._colors_p)
    self._colors7 = -mc7  # C-D
    self._gammaC = M_C.reshape(self._a, self._D3, self._D1, self._D8, self._d, self._D7).transpose(4,0,1,5,2,3)
    self._gammaD = M_D.reshape(self._D7, self._d, self._a, self._D6, self._D8, self._D5).transpose(1,2,3,4,5,0)

    if self.verbosity > 1:
      print('updated bond 7: new lambda7 =', self._lambda7)


  def update_bond8(self):
    """
    update lambda8 between C and D by applying gate to C left bond
    """
    w = self._lambda8**-1
    M_C = (self._gammaC.transpose(1,2,3,4,0,5)*w).reshape(self._a*self._D3*self._D7*self._D1, self._d*self._D8)
    M_D = (w.reshape(_sh)*self._gammaD.transpose(3,0,1,2,4,5)).reshape(self._D8*self._d, self._a*self._D6*self._D5*self._D7)

    col_L = -combine_colors(self._colors_a, self._colors3, self._colors7, self._colors1)
    col_R = -combine_colors(self._colors_a, self._colors6, self._colors5, self._colors7)
    M_C, self._lambda8, M_D, mc8 = update_first_neighbor(M_C, M_D, self._lambda8, self._g1, self._d,
             col_L=col_L, col_R=col_R, col_bond=-self._colors8, col_d=-self._colors_p)
    self._colors8 = -mc8 # C-D
    self._gammaC = M_C.reshape(self._a, self._D3, self._D7, self._D1, self._d, self._D8).transpose(4,0,1,2,3,5)
    self._gammaD = M_D.reshape(self._D8, self._d, self._a, self._D6, self._D5, self._D7).transpose(1,2,3,0,4,5)

    if self.verbosity > 1:
      print('updated bond 8: new lambda8 =', self._lambda8)


###############################################################################
# second neighbor updates
###############################################################################

###############################   links A-D   #################################
  def update_bonds25(self):
    """
    update lambda2 and lambda6 by applying gate to A upper-right next nearest
    neighbor bond with D through tensor B. Twin of 17.
    """
    w2 = self._lambda2**-1
    w5 = (self._lambda5**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,4,5,0,3)*w2).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = (self._gammaB.transpose(5,2,0,1,3,4)*(w2.reshape(_sh)*w5[None])).reshape(self._D2*self._D5, self._d*self._a*self._D4*self._D6)
    M_D = (self._gammaD.transpose(4,0,1,2,3,5)*w5[:,None]).reshape(self._D5*self._d, self._a*self._D6*self._D8*self._D7)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors3, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors4, self._colors6)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors8, self._colors7)
    M_A, M_B, M_D, self._lambda2, self._lambda5, self._colors2, self._colors5 = update_second_neighbor(
                                       M_A, M_B, M_D, self._lambda2, self._lambda5, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors2, col_bR=self._colors5, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D3, self._D4, self._d, self._D2).transpose(4, 0, 1, 5, 2, 3)
    self._gammaB = M_B.reshape(self._D2, self._D5, self._d, self._a, self._D4, self._D6).transpose(2, 3, 1, 4, 5, 0)
    self._gammaD = M_D.reshape(self._D5, self._d, self._a, self._D6, self._D8, self._D7).transpose(1, 2, 3, 4, 0, 5)

    if self.verbosity > 1:
      print('updated bonds 2 and 5: new lambda2 =', self._lambda2)
      print('                       new lambda5 =', self._lambda5)


  def update_bonds17(self):
    """
    update lambda2 and lambda6 by applying gate to A upper-right next nearest
    neighbor bond with D through tensor C. Twin of 25.
    """
    w1 = self._lambda1**-1
    w7 = (self._lambda7**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,3,4,5,0,2)*w1).reshape(self._a*self._D2*self._D3*self._D4, self._d*self._D1)
    M_C = (self._gammaC.transpose(4,3,0,1,2,5)*(w1.reshape(_sh)*w7[None])).reshape(self._D1*self._D7, self._d*self._a*self._D3*self._D8)
    M_D = (self._gammaD.transpose(5,0,1,2,3,4)*w7[:,None]).reshape(self._D7*self._d, self._a*self._D6*self._D8*self._D5)

    col_L = combine_colors(self._colors_a, self._colors2, self._colors3, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors3, self._colors8)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors8, self._colors5)
    M_A, M_C, M_D, self._lambda1, self._lambda7, self._colors1, self._colors7 = update_second_neighbor(
                                       M_A, M_C, M_D, self._lambda1, self._lambda7, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors1, col_bR=self._colors7, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D2, self._D3, self._D4, self._d, self._D1).transpose(4, 0, 5, 1, 2, 3)
    self._gammaC = M_C.reshape(self._D1, self._D7, self._d, self._a, self._D3, self._D8).transpose(2, 3, 4, 1, 0, 5)
    self._gammaD = M_D.reshape(self._D7, self._d, self._a, self._D6, self._D8, self._D5).transpose(1, 2, 3, 4, 5, 0)

    if self.verbosity > 1:
      print('updated bonds 1 and 7: new lambda1 =', self._lambda1)
      print('                       new lambda7 =', self._lambda7)


  def update_bonds26(self):
    """
    update lambda2 and lambda6 by applying gate to A down-right next nearest
    neighbor bond with D through tensor B. Twin of 37.
    """
    w2 = self._lambda2**-1
    w6 = (self._lambda6**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,4,5,0,3)*w2).reshape(self._a*self._D1*self._D3*self._D4, self._d*self._D2)
    M_B = (self._gammaB.transpose(5,4,0,1,2,3)*(w2.reshape(_sh)*w6[None])).reshape(self._D2*self._D6, self._d*self._a*self._D5*self._D4)
    M_D = (self._gammaD.transpose(2,0,1,3,4,5)*w6[:,None]).reshape(self._D6*self._d, self._a*self._D8*self._D5*self._D7)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors3, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors5, self._colors4)
    col_R = combine_colors(self._colors_a, self._colors8, self._colors5, self._colors7)
    M_A, M_B, M_D, self._lambda2, self._lambda6, self._colors2, self._colors6 = update_second_neighbor(
                                       M_A, M_B, M_D, self._lambda2, self._lambda6, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors2, col_bR=self._colors6, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D3, self._D4, self._d, self._D2).transpose(4, 0, 1, 5, 2, 3)
    self._gammaB = M_B.reshape(self._D2, self._D6, self._d, self._a, self._D5, self._D4).transpose(2, 3, 4, 5, 1, 0)
    self._gammaD = M_D.reshape(self._D5, self._d, self._a, self._D6, self._D8, self._D7).transpose(1, 2, 0, 3, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 2 and 6: new lambda2 =', self._lambda2)
      print('                       new lambda6 =', self._lambda6)


  def update_bonds37(self):
    """
    update lambda2 and lambda6 by applying gate to A down-right next nearest
    neighbor bond with D through tensor C. Twin of 26.
    """
    w3 = self._lambda3**-1
    w7 = (self._lambda7**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,3,5,0,4)*w3).reshape(self._a*self._D1*self._D2*self._D4, self._d*self._D3)
    M_C = (self._gammaC.transpose(2,3,0,1,4,5)*(w3.reshape(_sh)*w7[None])).reshape(self._D3*self._D7, self._d*self._a*self._D1*self._D8)
    M_D = (self._gammaD.transpose(5,0,1,2,3,4)*w7[:,None]).reshape(self._D7*self._d, self._a*self._D6*self._D8*self._D5)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors1, self._colors8)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors8, self._colors5)
    M_A, M_C, M_D, self._lambda3, self._lambda7, self._colors3, self._colors7 = update_second_neighbor(
                                       M_A, M_C, M_D, self._lambda3, self._lambda7, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors3, col_bR=self._colors7, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D4, self._d, self._D3).transpose(4, 0, 1, 2, 5, 3)
    self._gammaC = M_C.reshape(self._D3, self._D7, self._d, self._a, self._D1, self._D8).transpose(2, 3, 0, 1, 4, 5)
    self._gammaD = M_D.reshape(self._D7, self._d, self._a, self._D6, self._D8, self._D5).transpose(1, 2, 3, 4, 5, 0)

    if self.verbosity > 1:
      print('updated bonds 3 and 7: new lambda3 =', self._lambda3)
      print('                       new lambda7 =', self._lambda7)


  def update_bonds46(self):
    """
    update lambda4 and lambda6 by applying gate to A down-left next nearest
    neighbor bond with D through tensor B. Twin of 38.
    """
    w4 = self._lambda4**-1
    w6 = (self._lambda6**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,3,4,0,5)*w4).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D4)
    M_B = (self._gammaB.transpose(3,4,0,1,2,5)*(w4.reshape(_sh)*w6[None])).reshape(self._D4*self._D6, self._d*self._a*self._D5*self._D2)
    M_D = (self._gammaD.transpose(2,0,1,3,4,5)*w6[:,None]).reshape(self._D6*self._d, self._a*self._D8*self._D5*self._D7)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors3)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors5, self._colors2)
    col_R = combine_colors(self._colors_a, self._colors8, self._colors5, self._colors7)
    M_A, M_B, M_D, self._lambda4, self._lambda6, self._colors4, self._colors6 = update_second_neighbor(
                                       M_A, M_B, M_D, self._lambda4, self._lambda6, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors4, col_bR=self._colors6, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D3, self._d, self._D4).transpose(4, 0, 1, 2, 3, 5)
    self._gammaB = M_B.reshape(self._D4, self._D6, self._d, self._a, self._D5, self._D2).transpose(2, 3, 4, 0, 1, 5)
    self._gammaD = M_D.reshape(self._D6, self._d, self._a, self._D8, self._D5, self._D7).transpose(1, 2, 0, 3, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 4 and 6: new lambda4 =', self._lambda4)
      print('                       new lambda6 =', self._lambda6)


  def update_bonds38(self):
    """
    update lambda2 and lambda6 by applying gate to A down-left next nearest
    neighbor bond with D through tensor C. Twin of 46.
    """
    w3 = self._lambda3**-1
    w8 = (self._lambda8**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,3,5,0,4)*w3).reshape(self._a*self._D1*self._D2*self._D4, self._d*self._D3)
    M_C = (self._gammaC.transpose(2,5,0,1,3,4)*(w3.reshape(_sh)*w8[None])).reshape(self._D3*self._D8, self._d*self._a*self._D7*self._D1)
    M_D = (self._gammaD.transpose(3,0,1,2,4,5)*w8[:,None]).reshape(self._D8*self._d, self._a*self._D6*self._D5*self._D7)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors7, self._colors1)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors5, self._colors7)
    M_A, M_C, M_D, self._lambda3, self._lambda8, self._colors3, self._colors8 = update_second_neighbor(
                                       M_A, M_C, M_D, self._lambda3, self._lambda8, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors3, col_bR=self._colors8, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D4, self._d, self._D3).transpose(4, 0, 1, 2, 5, 3)
    self._gammaC = M_C.reshape(self._D3, self._D8, self._d, self._a, self._D7, self._D1).transpose(2, 3, 0, 4, 5, 1)
    self._gammaD = M_D.reshape(self._D8, self._d, self._a, self._D6, self._D5, self._D7).transpose(1, 2, 3, 0, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 3 and 8: new lambda3 =', self._lambda3)
      print('                       new lambda8 =', self._lambda8)


  def update_bonds45(self):
    """
    update lambda4 and lambda5 by applying gate to A upper-left next nearest
    neighbor bond with D through tensor B. Twin of 18.
    """
    w4 = self._lambda4**-1
    w5 = (self._lambda5**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,2,3,4,0,5)*w4).reshape(self._a*self._D1*self._D2*self._D3, self._d*self._D4)
    M_B = (self._gammaB.transpose(3,2,0,1,4,5)*(w4.reshape(_sh)*w5[None])).reshape(self._D4*self._D5, self._d*self._a*self._D6*self._D2)
    M_D = (self._gammaD.transpose(4,0,1,2,3,5)*w5[:,None]).reshape(self._D5*self._d, self._a*self._D6*self._D8*self._D7)

    col_L = combine_colors(self._colors_a, self._colors1, self._colors2, self._colors3)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors6, self._colors2)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors8, self._colors7)
    M_A, M_B, M_D, self._lambda4, self._lambda5, self._colors4, self._colors5 = update_second_neighbor(
                                       M_A, M_B, M_D, self._lambda4, self._lambda5, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors4, col_bR=self._colors5, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D1, self._D2, self._D3, self._d, self._D4).transpose(4, 0, 1, 2, 3, 5)
    self._gammaB = M_B.reshape(self._D4, self._D5, self._d, self._a, self._D6, self._D2).transpose(2, 3, 1, 0, 4, 5)
    self._gammaD = M_D.reshape(self._D5, self._d, self._a, self._D6, self._D8, self._D7).transpose(1, 2, 3, 4, 0, 5)

    if self.verbosity > 1:
      print('updated bonds 4 and 5: new lambda4 =', self._lambda4)
      print('                       new lambda5 =', self._lambda5)


  def update_bonds18(self):
    """
    update lambda1 and lambda8 by applying gate to A upper-left next nearest
    neighbor bond with D through tensor C. Twin of 45.
    """
    w1 = self._lambda1**-1
    w8 = (self._lambda8**-1)[:,None,None,None,None]
    M_A = (self._gammaA.transpose(1,3,4,5,0,2)*w1).reshape(self._a*self._D2*self._D3*self._D4, self._d*self._D1)
    M_C = (self._gammaC.transpose(4,5,0,1,2,3)*(w1.reshape(_sh)*w8[None])).reshape(self._D1*self._D8, self._d*self._a*self._D3*self._D7)
    M_D = (self._gammaD.transpose(3,0,1,2,4,5)*w8[:,None]).reshape(self._D8*self._d, self._a*self._D6*self._D5*self._D7)

    col_L = combine_colors(self._colors_a, self._colors2, self._colors3, self._colors4)
    col_mid = combine_colors(self._colors_p, self._colors_a, self._colors3, self._colors7)
    col_R = combine_colors(self._colors_a, self._colors6, self._colors5, self._colors7)
    M_A, M_C, M_D, self._lambda1, self._lambda8, self._colors1, self._colors8 = update_second_neighbor(
                                       M_A, M_C, M_D, self._lambda1, self._lambda8, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=self._colors1, col_bR=self._colors8, col_d=self._colors_p)

    self._gammaA = M_A.reshape(self._a, self._D2, self._D3, self._D4, self._d, self._D1).transpose(4, 0, 5, 1, 2, 3)
    self._gammaC = M_C.reshape(self._D1, self._D8, self._d, self._a, self._D3, self._D7).transpose(2, 3, 4, 5, 0, 1)
    self._gammaD = M_D.reshape(self._D8, self._d, self._a, self._D6, self._D5, self._D7).transpose(1, 2, 3, 0, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 1 and 8: new lambda1 =', self._lambda1)
      print('                       new lambda8 =', self._lambda8)




###############################   links B-C   #################################
  def update_bonds41(self):
    """
    update lambda4 and lambda1 by applying gate to B upper-right next nearest
    neighbor bond with C through tensor A. Twin of 58.
    """
    w4 = self._lambda4**-1
    w1 = (self._lambda1**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,4,5,0,3)*w4).reshape(self._a*self._D5*self._D6*self._D2, self._d*self._D4)
    M_A = (self._gammaA.transpose(5,2,0,1,3,4)*(w4.reshape(_sh)*w1[None])).reshape(self._D4*self._D1, self._d*self._a*self._D2*self._D3)
    M_C = (self._gammaC.transpose(4,0,1,2,3,5)*w1[:,None]).reshape(self._D1*self._d, self._a*self._D3*self._D7*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors6, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors2, self._colors3)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors7, self._colors8)
    M_B, M_A, M_C, self._lambda4, self._lambda1, mc4, mc1 = update_second_neighbor(
                                       M_B, M_A, M_C, self._lambda4, self._lambda1, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors4, col_bR=-self._colors1, col_d=-self._colors_p)
    self._colors4 = -mc4
    self._colors1 = -mc1

    self._gammaB = M_B.reshape(self._a, self._D5, self._D6, self._D2, self._d, self._D4).transpose(4, 0, 1, 5, 2, 3)
    self._gammaA = M_A.reshape(self._D4, self._D1, self._d, self._a, self._D2, self._D3).transpose(2, 3, 1, 4, 5, 0)
    self._gammaC = M_C.reshape(self._D1, self._d, self._a, self._D3, self._D7, self._D8).transpose(1, 2, 3, 4, 0, 5)

    if self.verbosity > 1:
      print('updated bonds 4 and 1: new lambda4 =', self._lambda4)
      print('                       new lambda1 =', self._lambda1)


  def update_bonds58(self):
    """
    update lambda2 and lambda6 by applying gate to B upper-right next nearest
    neighbor bond with C through tensor D. Twin of 41.
    """
    w5 = self._lambda5**-1
    w8 = (self._lambda8**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,3,4,5,0,2)*w5).reshape(self._a*self._D4*self._D6*self._D2, self._d*self._D5)
    M_D = (self._gammaD.transpose(4,3,0,1,2,5)*(w5.reshape(_sh)*w8[None])).reshape(self._D5*self._D8, self._d*self._a*self._D6*self._D7)
    M_C = (self._gammaC.transpose(5,0,1,2,3,4)*w8[:,None]).reshape(self._D8*self._d, self._a*self._D3*self._D7*self._D1)

    col_L = -combine_colors(self._colors_a, self._colors4, self._colors6, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors6, self._colors7)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors7, self._colors1)
    M_B, M_D, M_C, self._lambda5, self._lambda8, mc5, mc8 = update_second_neighbor(
                                       M_B, M_D, M_C, self._lambda5, self._lambda8, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors5, col_bR=-self._colors8, col_d=-self._colors_p)
    self._colors5 = -mc5
    self._colors8 = -mc8

    self._gammaB = M_B.reshape(self._a, self._D4, self._D6, self._D2, self._d, self._D5).transpose(4, 0, 5, 1, 2, 3)
    self._gammaD = M_D.reshape(self._D5, self._D8, self._d, self._a, self._D6, self._D7).transpose(2, 3, 4, 1, 0, 5)
    self._gammaC = M_C.reshape(self._D8, self._d, self._a, self._D3, self._D7, self._D1).transpose(1, 2, 3, 4, 5, 0)

    if self.verbosity > 1:
      print('updated bonds 5 and 8: new lambda5 =', self._lambda5)
      print('                       new lambda8 =', self._lambda8)


  def update_bonds43(self):
    """
    update lambda4 and lambda3 by applying gate to B down-right next nearest
    neighbor bond with C through tensor A. Twin of 68.
    """
    w4 = self._lambda4**-1
    w3 = (self._lambda3**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,4,5,0,3)*w4).reshape(self._a*self._D5*self._D6*self._D2, self._d*self._D4)
    M_A = (self._gammaA.transpose(5,4,0,1,2,3)*(w4.reshape(_sh)*w3[None])).reshape(self._D4*self._D3, self._d*self._a*self._D1*self._D2)
    M_C = (self._gammaC.transpose(2,0,1,3,4,5)*w3[:,None]).reshape(self._D3*self._d, self._a*self._D7*self._D1*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors6, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors1, self._colors2)
    col_R = -combine_colors(self._colors_a, self._colors7, self._colors1, self._colors8)
    M_B, M_A, M_C, self._lambda4, self._lambda3, mc4, mc3 = update_second_neighbor(
                                       M_B, M_A, M_C, self._lambda4, self._lambda3, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors4, col_bR=-self._colors3, col_d=-self._colors_p)
    self._colors4 = -mc4
    self._colors3 = -mc3

    self._gammaB = M_B.reshape(self._a, self._D5, self._D6, self._D2, self._d, self._D4).transpose(4, 0, 1, 5, 2, 3)
    self._gammaA = M_A.reshape(self._D4, self._D3, self._d, self._a, self._D1, self._D2).transpose(2, 3, 4, 5, 1, 0)
    self._gammaC = M_C.reshape(self._D3, self._d, self._a, self._D7, self._D1, self._D8).transpose(1, 2, 0, 3, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 4 and 3: new lambda4 =', self._lambda4)
      print('                       new lambda3 =', self._lambda3)


  def update_bonds68(self):
    """
    update lambda2 and lambda6 by applying gate to B down-right next nearest
    neighbor bond with C through tensor D. Twin of 43.
    """
    w6 = self._lambda6**-1
    w8 = (self._lambda8**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,3,5,0,4)*w6).reshape(self._a*self._D5*self._D4*self._D2, self._d*self._D6)
    M_D = (self._gammaD.transpose(2,3,0,1,4,5)*(w6.reshape(_sh)*w8[None])).reshape(self._D6*self._D8, self._d*self._a*self._D5*self._D7)
    M_C = (self._gammaC.transpose(5,0,1,2,3,4)*w8[:,None]).reshape(self._D8*self._d, self._a*self._D3*self._D7*self._D1)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors4, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors5, self._colors7)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors7, self._colors1)
    M_B, M_D, M_C, self._lambda6, self._lambda8, mc6, mc8 = update_second_neighbor(
                                       M_B, M_D, M_C, self._lambda6, self._lambda8, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors6, col_bR=-self._colors8, col_d=-self._colors_p)
    self._colors6 = -mc6
    self._colors8 = -mc8

    self._gammaB = M_B.reshape(self._a, self._D5, self._D4, self._D2, self._d, self._D6).transpose(4, 0, 1, 2, 5, 3)
    self._gammaD = M_D.reshape(self._D6, self._D8, self._d, self._a, self._D5, self._D7).transpose(2, 3, 0, 1, 4, 5)
    self._gammaC = M_C.reshape(self._D8, self._d, self._a, self._D3, self._D7, self._D1).transpose(1, 2, 3, 4, 5, 0)

    if self.verbosity > 1:
      print('updated bonds 6 and 8: new lambda6 =', self._lambda6)
      print('                       new lambda8 =', self._lambda8)


  def update_bonds23(self):
    """
    update lambda2 and lambda3 by applying gate to B down-left next nearest
    neighbor bond with C through tensor A. Twin of 67.
    """
    w2 = self._lambda2**-1
    w3 = (self._lambda3**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,3,4,0,5)*w2).reshape(self._a*self._D5*self._D4*self._D6, self._d*self._D2)
    M_A = (self._gammaA.transpose(3,4,0,1,2,5)*(w2.reshape(_sh)*w3[None])).reshape(self._D2*self._D3, self._d*self._a*self._D1*self._D4)
    M_C = (self._gammaC.transpose(2,0,1,3,4,5)*w3[:,None]).reshape(self._D3*self._d, self._a*self._D7*self._D1*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors4, self._colors6)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors1, self._colors4)
    col_R = -combine_colors(self._colors_a, self._colors7, self._colors1, self._colors8)
    M_B, M_A, M_C, self._lambda2, self._lambda3, mc2, mc3 = update_second_neighbor(
                                       M_B, M_A, M_C, self._lambda2, self._lambda3, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors2, col_bR=-self._colors3, col_d=-self._colors_p)
    self._colors2 = -mc2
    self._colors3 = -mc3

    self._gammaB = M_B.reshape(self._a, self._D5, self._D4, self._D6, self._d, self._D2).transpose(4, 0, 1, 2, 3, 5)
    self._gammaA = M_A.reshape(self._D2, self._D3, self._d, self._a, self._D1, self._D4).transpose(2, 3, 4, 0, 1, 5)
    self._gammaC = M_C.reshape(self._D3, self._d, self._a, self._D7, self._D1, self._D8).transpose(1, 2, 0, 3, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 2 and 3: new lambda2 =', self._lambda2)
      print('                       new lambda3 =', self._lambda3)


  def update_bonds67(self):
    """
    update lambda6 and lambda7 by applying gate to B down-left next nearest
    neighbor bond with C through tensor D. Twin of 23.
    """
    w6 = self._lambda6**-1
    w7 = (self._lambda7**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,3,5,0,4)*w6).reshape(self._a*self._D5*self._D4*self._D2, self._d*self._D6)
    M_D = (self._gammaD.transpose(2,5,0,1,3,4)*(w6.reshape(_sh)*w7[None])).reshape(self._D6*self._D7, self._d*self._a*self._D8*self._D5)
    M_C = (self._gammaC.transpose(3,0,1,2,4,5)*w7[:,None]).reshape(self._D7*self._d, self._a*self._D3*self._D1*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors4, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors8, self._colors5)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors1, self._colors8)
    M_B, M_D, M_C, self._lambda6, self._lambda7, mc6, mc7 = update_second_neighbor(
                                       M_B, M_D, M_C, self._lambda6, self._lambda7, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors6, col_bR=-self._colors7, col_d=-self._colors_p)
    self._colors6 = -mc6
    self._colors7 = -mc7

    self._gammaB = M_B.reshape(self._a, self._D5, self._D4, self._D2, self._d, self._D6).transpose(4, 0, 1, 2, 5, 3)
    self._gammaD = M_D.reshape(self._D6, self._D7, self._d, self._a, self._D8, self._D5).transpose(2, 3, 0, 4, 5, 1)
    self._gammaC = M_C.reshape(self._D7, self._d, self._a, self._D3, self._D1, self._D8).transpose(1, 2, 3, 0, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 6 and 7: new lambda6 =', self._lambda6)
      print('                       new lambda7 =', self._lambda7)


  def update_bonds21(self):
    """
    update lambda2 and lambda1 by applying gate to B upper-left next nearest
    neighbor bond with C through tensor A. Twin of 57.
    """
    w2 = self._lambda2**-1
    w1 = (self._lambda1**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,2,3,4,0,5)*w2).reshape(self._a*self._D5*self._D4*self._D6, self._d*self._D2)
    M_A = (self._gammaA.transpose(3,2,0,1,4,5)*(w2.reshape(_sh)*w1[None])).reshape(self._D2*self._D1, self._d*self._a*self._D3*self._D4)
    M_C = (self._gammaC.transpose(4,0,1,2,3,5)*w1[:,None]).reshape(self._D1*self._d, self._a*self._D3*self._D7*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors5, self._colors4, self._colors6)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors3, self._colors4)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors7, self._colors8)
    M_B, M_A, M_C, self._lambda2, self._lambda1, mc2, mc1 = update_second_neighbor(
                                       M_B, M_A, M_C, self._lambda2, self._lambda1, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors2, col_bR=-self._colors1, col_d=-self._colors_p)
    self._colors2 = -mc2
    self._colors1 = -mc1

    self._gammaB = M_B.reshape(self._a, self._D5, self._D4, self._D6, self._d, self._D2).transpose(4, 0, 1, 2, 3, 5)
    self._gammaA = M_A.reshape(self._D2, self._D1, self._d, self._a, self._D3, self._D4).transpose(2, 3, 1, 0, 4, 5)
    self._gammaC = M_C.reshape(self._D1, self._d, self._a, self._D3, self._D7, self._D8).transpose(1, 2, 3, 4, 0, 5)

    if self.verbosity > 1:
      print('updated bonds 2 and 1: new lambda2 =', self._lambda2)
      print('                       new lambda1 =', self._lambda1)


  def update_bonds57(self):
    """
    update lambda6 and lambda7 by applying gate to B down-left next nearest
    neighbor bond with C through tensor D. Twin of 21.
    """
    w5 = self._lambda5**-1
    w7 = (self._lambda7**-1)[:,None,None,None,None]
    M_B = (self._gammaB.transpose(1,3,4,5,0,2)*w5).reshape(self._a*self._D4*self._D6*self._D2, self._d*self._D5)
    M_D = (self._gammaD.transpose(4,5,0,1,2,3)*(w5.reshape(_sh)*w7[None])).reshape(self._D5*self._D7, self._d*self._a*self._D6*self._D8)
    M_C = (self._gammaC.transpose(3,0,1,2,4,5)*w7[:,None]).reshape(self._D7*self._d, self._a*self._D3*self._D1*self._D8)

    col_L = -combine_colors(self._colors_a, self._colors4, self._colors6, self._colors2)
    col_mid = -combine_colors(self._colors_p, self._colors_a, self._colors6, self._colors8)
    col_R = -combine_colors(self._colors_a, self._colors3, self._colors1, self._colors8)
    M_B, M_D, M_C, self._lambda5, self._lambda7, mc5, mc7 = update_second_neighbor(
                                       M_B, M_D, M_C, self._lambda5, self._lambda7, self._g2, self._d, col_L=col_L, col_mid=col_mid, col_R=col_R, col_bL=-self._colors5, col_bR=-self._colors7, col_d=-self._colors_p)
    self._colors5 = -mc5
    self._colors7 = -mc7

    self._gammaB = M_B.reshape(self._a, self._D4, self._D6, self._D2, self._d, self._D5).transpose(4, 0, 5, 1, 2, 3)
    self._gammaD = M_D.reshape(self._D5, self._D7, self._d, self._a, self._D6, self._D8).transpose(2, 3, 4, 5, 0, 1)
    self._gammaC = M_C.reshape(self._D7, self._d, self._a, self._D3, self._D1, self._D8).transpose(1, 2, 3, 0, 4, 5)

    if self.verbosity > 1:
      print('updated bonds 5 and 7: new lambda5 =', self._lambda5)
      print('                       new lambda7 =', self._lambda7)
