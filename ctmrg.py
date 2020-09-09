import rdm
from ctm_environment import CTM_Environment
from ctm_contract import contract_u_half, contract_l_half, contract_d_half, contract_r_half
from ctm_renormalize import *
from toolsU1 import combine_colors, checkU1

class CTMRG(object):
  # convention: legs and tensors are taken clockwise starting from upper right.
  #    C1-T1-T1-C2
  #    |  |  |   |
  #    T4-a--a--T2
  #    |  |  |   |
  #    T4-a--a--T2
  #    |  |  |   |
  #    C4-T3-T3-C3

  # When passed as arguments to a function for contraction, tensors are sorted
  # from left to right, from up to down.

  def __init__(self, chi, tensors=(), cell=None, tiling=None, colors=None, saveFile=None, verbosity=0):
    self.verbosity = verbosity
    if self.verbosity > 0:
      print(f"initalize CTMRG with chi = {chi} and self.verbosity = {self.verbosity}")
    self.chi = chi
    self._env = CTM_Environment(tensors, cell=cell, tiling=tiling, colors=colors, saveFile=saveFile)
    self._neq_coords = self._env.neq_coords
    if self.verbosity > 0:
      print('CTMRG constructed')
      print("unit cell =", self._env.cell, sep="\n")
      if colors is not None:
        print("colors =", colors, sep="\n")

  def save_to_file(self, saveFile):
    self._env.save_to_file(saveFile)   # all the data is in env

  def load_from_file(self, saveFile):
    self._env.load_from_file(saveFile)

  @property
  def Lx(self):
    return self._env.Lx

  @property
  def Ly(self):
    return self._env.Ly

  @property
  def cell(self):
    return self._env.cell

  def set_tensors(self, tensors, colors=None, keep_env=True):
    if self.verbosity > 0:
      print("set new tensors")
    if keep_env:
      self._env.set_tensors(tensors, colors=colors)
    else:   # restart from fresh
      self._env = Env.Env(tensors, cell=self._env.cell.copy(), colors=colors)

  def print_tensor_shapes(self):
    print("C1 T1 C2 // T4 A T2 // C4 T3 C4")
    for (x,y) in self._neq_coords:
      print(f'({x},{y}):', self._env.get_C1(x,y).shape,
      self._env.get_T1(x+1,y).shape,    self._env.get_C2(x+2,y).shape,
      self._env.get_T4(x,y+1).shape,    self._env.get_A(x+1,y+1).shape,
      self._env.get_T2(x+2,y+1).shape,  self._env.get_C4(x,y+3).shape,
      self._env.get_T3(x+1,y+2).shape,  self._env.get_C3(x+2,y+2).shape)

  def check_symetries(self):
    for (x,y) in self._neq_coords:
      print(f'({x},{y}):',
       'C1', checkU1(self._env.get_C1(x,y),[self._env.get_color_C1_r(x,y), self._env.get_color_C1_d(x,y)]),
       'T1', checkU1(self._env.get_T1(x,y),[self._env.get_color_T1_r(x,y), self._env.get_color_T1_d(x,y), -self._env.get_color_T1_d(x,y), self._env.get_color_T1_l(x,y)]),
       'C2', checkU1(self._env.get_C2(x,y),[self._env.get_color_C2_d(x,y), self._env.get_color_C2_l(x,y)]),
       'T2', checkU1(self._env.get_T2(x,y),[self._env.get_color_T2_u(x,y), self._env.get_color_T2_d(x,y), self._env.get_color_T2_l(x,y), -self._env.get_color_T2_l(x,y)]),
       'C3', checkU1(self._env.get_C3(x,y),[self._env.get_color_C3_u(x,y), self._env.get_color_C3_l(x,y)]),
       'T3', checkU1(self._env.get_T3(x,y),[self._env.get_color_T3_u(x,y), -self._env.get_color_T3_u(x,y), self._env.get_color_T3_r(x,y), self._env.get_color_T3_l(x,y)]),
       'C4', checkU1(self._env.get_C4(x,y),[self._env.get_color_C4_u(x,y), self._env.get_color_C4_r(x,y)]),
       'T4', checkU1(self._env.get_T4(x,y),[self._env.get_color_T4_u(x,y), self._env.get_color_T4_r(x,y), -self._env.get_color_T4_r(x,y), self._env.get_color_T4_d(x,y)]))


  def iterate(self):
    self.up_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.right_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.down_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.left_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()


  def up_move(self):
    if self.verbosity > 0:
      print('\nstart up move')
    # 1) compute isometries for every non-equivalent sites
    # convention : for every move, leg 0 of R and Rt are to be contracted
    for x,y in self._neq_coords:
      R = contract_r_half(self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                           self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                           self._env.get_T3(x+2,y+3),self._env.get_C3(x+3,y+3))
      Rt = contract_l_half(self._env.get_C1(x,y),   self._env.get_T1(x+1,y),
                            self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                            self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                            self._env.get_C4(x,y+3),self._env.get_T3(x+1,y+3))
      #        L-0  == 1-R
      #        L         R
      #        L         R  => transpose R
      #        L-1     0-R
      col_A_l = self._env.get_colors_A(x+2,y+2)[5]
      color = combine_colors(self._env.get_color_T3_l(x+2,y+3), col_A_l, -col_A_l)
      P, Pt, color = construct_projectors(R.T, Rt, self.chi, color)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.store_projectors(x+2,y,P,Pt,color)# indices: Pt <=> renormalized T in R
      del R, Rt

    # 2) renormalize every non-equivalent C1, T1 and C2
    # need all projectors to be constructed at this time
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for x,y in self._neq_coords:
      P = self._env.get_P(x+1,y)
      Pt = self._env.get_Pt(x,y)
      color_P = self._env.get_color_P(x+1,y)
      color_Pt = -self._env.get_color_P(x,y)
      nC1 = renormalize_C1_up(self._env.get_C1(x,y), self._env.get_T4(x,y+1),P)

      nT1 = renormalize_T1(Pt,self._env.get_T1(x,y),self._env.get_A(x,y+1),P)

      nC2 = renormalize_C2_up(self._env.get_C2(x,y),self._env.get_T2(x,y+1),Pt)
      self._env.store_renormalized_tensors(x,y+1,nC1,nT1,nC2,color_P,color_Pt)

    # 3) store renormalized tensors in the environment
    # renormalization reads C1[x,y] but writes C1[x,y+1]
    # => need to compute every renormalized tensors before storing any of them
    self._env.fix_renormalized_up()
    if self.verbosity > 0:
      print('up move completed')


  def right_move(self):
    if self.verbosity > 0:
      print('\nstart right move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      0  1    0
      #      |  | => R
      #      DDDD    1
      R = contract_d_half(self._env.get_T4(x,y+2),   self._env.get_A(x+1,y+2),
                          self._env.get_A(x+2,y+2),  self._env.get_T2(x+3,y+2),
                          self._env.get_C4(x,y+3),   self._env.get_T3(x+1,y+3),
                          self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
      #      UUUU     0
      #      |  | =>  Rt
      #      1  0     1
      Rt = contract_u_half(self._env.get_C1(x,y),    self._env.get_T1(x+1,y),
                           self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_T4(x,y+1),  self._env.get_A(x+1,y+1),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1))
      col_A_u = self._env.get_colors_A(x+1,y+2)[2]
      color = combine_colors(self._env.get_color_T4_u(x,y+2), col_A_u, -col_A_u)
      P, Pt, color = construct_projectors(R.T,Rt,self.chi, color)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.store_projectors(x+3, y+2, P, Pt, color)
      del R, Rt

    # 2) renormalize tensors by absorbing column
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for x,y in self._neq_coords:
      P = self._env.get_P(x,y+1)
      Pt = self._env.get_Pt(x,y)
      color_P = self._env.get_color_P(x,y+1)
      color_Pt = -self._env.get_color_P(x,y)
      nC2 = renormalize_C2_right(self._env.get_C2(x,y),self._env.get_T1(x-1,y),
                                 P)

      nT2 = renormalize_T2(Pt,self._env.get_A(x-1,y),self._env.get_T2(x,y),P)

      nC3 = renormalize_C3_right(self._env.get_C3(x,y),self._env.get_T3(x-1,y),
                                 Pt)
      self._env.store_renormalized_tensors(x-1,y,nC2,nT2,nC3,color_P,color_Pt)

    # 3) store renormalized tensors in the environment
    self._env.fix_renormalized_right()
    if self.verbosity > 0:
      print('right move completed')


  def down_move(self):
    if self.verbosity > 0:
      print('\nstart down move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #        L-0      L-1
      #        L        L
      #        L    =>  L
      #        L-1      L-0
      R = contract_l_half(self._env.get_C1(x,y),     self._env.get_T1(x+1,y),
                            self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                            self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                            self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3))
      #      1-R
      #        R
      #        R
      #      0-R
      Rt = contract_r_half(self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                           self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                           self._env.get_T3(x+2,y+3),self._env.get_C3(x+3,y+3))

      col_A_r = self._env.get_colors_A(x+1,y+1)[3]
      color = combine_colors(self._env.get_color_T1_r(x+1,y), col_A_r, -col_A_r)
      P, Pt, color = construct_projectors(R.T,Rt,self.chi,color)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.store_projectors(x+3, y+3, P, Pt, color)
      del R, Rt

    # 2) renormalize every non-equivalent C3, T3 and C4
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for x,y in self._neq_coords:
      P = self._env.get_P(x-1,y)
      Pt = self._env.get_Pt(x,y)
      color_P = self._env.get_color_P(x-1,y)
      color_Pt = -self._env.get_color_P(x,y)
      nC3 = renormalize_C3_down(self._env.get_C3(x,y), self._env.get_T2(x,y-1),
                                P)

      nT3 = renormalize_T3(Pt,self._env.get_T3(x,y),self._env.get_A(x,y-1),P)

      nC4 = renormalize_C4_down(self._env.get_C4(x,y), self._env.get_T4(x,y-1),
                                Pt)
      self._env.store_renormalized_tensors(x,y-1,nC3,nT3,nC4,color_P,color_Pt)

    # 3) store renormalized tensors in the environment
    self._env.fix_renormalized_down()
    if self.verbosity > 0:
      print('down move completed')


  def left_move(self):
    if self.verbosity > 0:
      print('\nstart left move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      UUUU      1
      #      |  |  =>  R
      #      1  0      0
      R = contract_u_half(self._env.get_C1(x,y),    self._env.get_T1(x+1,y),
                           self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_T4(x,y+1),  self._env.get_A(x+1,y+1),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1))
      #      0  1
      #      |  |
      #      DDDD
      Rt = contract_d_half(self._env.get_T4(x,y+2),   self._env.get_A(x+1,y+2),
                          self._env.get_A(x+2,y+2),  self._env.get_T2(x+3,y+2),
                          self._env.get_C4(x,y+3),   self._env.get_T3(x+1,y+3),
                          self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
      col_A_d = self._env.get_colors_A(x+2,y+1)[4]
      color = combine_colors(self._env.get_color_T2_d(x+3,y+1), col_A_d, -col_A_d)
      P, Pt, color = construct_projectors(R.T, Rt, self.chi, color)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.store_projectors(x, y+1, P, Pt, color)
      del R, Rt

    # 2) renormalize every non-equivalent C4, T4 and C1
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for x,y in self._neq_coords:
      P = self._env.get_P(x,y-1)
      Pt = self._env.get_Pt(x,y)
      color_P = self._env.get_color_P(x,y-1)
      color_Pt = -self._env.get_color_P(x,y)
      nC4 = renormalize_C4_left(self._env.get_C4(x,y), self._env.get_T3(x+1,y),
                                P)

      nT4 = renormalize_T4(Pt,self._env.get_T4(x,y),self._env.get_A(x+1,y),P)

      nC1 = renormalize_C1_left(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                                Pt)
      self._env.store_renormalized_tensors(x+1,y,nC4,nT4,nC1,color_P,color_Pt)

    # 3) store renormalized tensors in the environment
    self._env.fix_renormalized_left()
    if self.verbosity > 0:
      print('left move completed')


  def compute_rdm1x1(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 1x1 with C1 coord = ({x},{y})')
    return rdm.rdm_1x1(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                         self._env.get_C2(x+2,y), self._env.get_T4(x,y+1),
                         self._env.get_A(x+1,y+1), self._env.get_T2(x+2,y+1),
                         self._env.get_C4(x,y+2), self._env.get_T3(x+1,y+2),
                         self._env.get_C3(x+2,y+2))


  def compute_rdm1x2(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 1x2 with C1 coord = ({x},{y})')
    return rdm.rdm_1x2(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                          self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                          self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                          self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                          self._env.get_C4(x,y+2), self._env.get_T3(x+1,y+2),
                          self._env.get_T3(x+2,y+2), self._env.get_C3(x+3,y+2))



  def compute_rdm2x1(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 2x1 with C1 coord = ({x},{y})')
    return rdm.rdm_2x1(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                          self._env.get_C2(x+2,y), self._env.get_T4(x,y+1),
                          self._env.get_A(x+1,y+1), self._env.get_T2(x+2,y+1),
                          self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                          self._env.get_T2(x+2,y+2), self._env.get_C4(x,y+3),
                          self._env.get_T3(x+1,y+3), self._env.get_C3(x+2,y+3))


  def compute_rdm2x2(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 2x2 with C1 coord = ({x},{y})')
    return rdm.rdm_2x2(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                       self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                       self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                       self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                       self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                       self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                       self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3),
                       self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))


  def compute_rdm_diag_dr(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm for down right diagonal sites ({x+1},{y+1}) and ({x+2},{y+2})')
    return rdm.rdm_diag_dr(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                       self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                       self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                       self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                       self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                       self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                       self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3),
                       self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))


  def compute_rdm_diag_ur(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm for upper right diagonal sites ({x+1},{y+2}) and ({x+2},{y+1})')
    return rdm.rdm_diag_ur(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                       self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                       self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                       self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                       self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                       self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                       self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3),
                       self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
