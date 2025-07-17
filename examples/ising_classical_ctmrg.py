#!/usr/bin/env python
"""
Compute the partition function of the square lattice classical Ising model
Implement a naive C4v-symmetric classical CTMRG
Use trivial and Z2 symmetries
"""

import numpy as np
import scipy.linalg as lg
import scipy.special

from frostspin import AsymmetricTensor, DiagonalTensor, U1SymmetricTensor

# ===============  simulation parameters  =================
D = 2  # bond dimension for Ising model
beta = 0.5  # inverse temperature
chi_target = 20  # maximal corner dimension
n_iter = 80  # number of CTMRG iteration
rng = np.random.default_rng(42)


# ===============  exact results  =================
beta_crit = 0.5 * np.log(1 + np.sqrt(2))  # around 0.441


def exact_magn(beta):
    if beta < beta_crit:
        return 0.0
    return (1 - ((np.sinh(2 * beta)) ** (-4))) ** (1.0 / 8)


def exact_energy(beta):
    k = np.sinh(2 * beta) ** -2
    x = scipy.special.ellipk(4 * k * (1 + k) ** -2)
    return -(1.0 + 2 / np.pi * (2 * np.tanh(2 * beta) ** 2 - 1) * x) / np.tanh(2 * beta)


# ===============  Initialize tensors  =================
# construct Boltzmann weights
h = -2 * np.eye(2) + 1  # ferromagnetic Ising model
bw = np.exp(-beta * h)
sq_bw = lg.sqrtm(bw)

# construct site tensor
delta = np.zeros((D, D, D, D))
delta[0, 0, 0, 0] = 1
delta[1, 1, 1, 1] = 1
a0 = np.einsum("abcd,ai,bj,ck,dl->ijkl", delta, sq_bw, sq_bw, sq_bw, sq_bw)

# construct magnetization operator
magn_op = delta.copy()
magn_op[1, 1, 1, 1] = -1
magn_op = np.einsum("abcd,ai,bj,ck,dl->ijkl", magn_op, sq_bw, sq_bw, sq_bw, sq_bw)

# cast to SymmetricTensor
bond_repr_asym = np.array([D])
row_asym = (bond_repr_asym,) * 2
col_asym = (bond_repr_asym,) * 2

# SymmetricTensor imposes bond orientation
# for classical CTMRG without symmetry, it does not really matter
siga = np.array([False, False, True, True])
a_asym = AsymmetricTensor.from_array(a0, row_asym, col_asym, signature=siga)
#       2*
#       ^
#       |
# 3* -<-a-<-  0
#       |
#       ^
#       1

# need to cheat a bit with arrows in classical C4v CTMRG
# does not matter for asymmetric or Z2
siga2 = np.array([False, True, True, True])
a_asym2 = AsymmetricTensor(
    a_asym.row_reps, a_asym.col_reps, a_asym.blocks, a_asym.block_irreps, siga2
)
#       2*
#       ^
#       |
# 3* -<-a2-<- 0
#       |
#       v
#       1

# initialize corner eigenvalues s
chi0 = chi_target
s0 = rng.normal(size=(chi0,))
s0 = s0[np.abs(s0).argsort()[::-1]] / lg.norm(s0)

# cast to SymmetricTensor
corner_repr_asym = np.array([chi0])
irrep_dim = 1  # trivial symmetry: there is only one "irrep" with dim = 1
s0_asym = DiagonalTensor(
    (s0,), corner_repr_asym, a_asym.block_irreps, [irrep_dim], a_asym.symmetry()
)

# initialize edge tensor T
T0 = rng.normal(size=(chi0, D, chi0))
T0 = T0 + T0.swapaxes(0, 2)  # impose symmetric T
T0 = T0 / lg.norm(T0)

# cast to SymmetricTensor
sigT = np.array([False, False, True])
#  2* -<-T-<- 0
#        |
#        ^
#        1*
T0_asym = AsymmetricTensor.from_array(
    T0, (corner_repr_asym,), (bond_repr_asym, corner_repr_asym), signature=sigT
)


# ===============  Implement C4v classical CTMRG  =================
def iterate_ctm(s, T, a, a2):
    """
    Iterate classical C4v CTMRG
    Need to supply site tensor a with 2 different signatures
    Not an issue for classical CTMRG
    """
    #  2* -<-Tp-<- 0
    #        |
    #        ^
    #        1
    Tp = T.permute((0, 1), (2,))
    matC = Tp * s

    #  s-----T-<- 0
    #  |     ^
    #  |     1
    #  T-<2*
    #  v
    #  3*
    matC = matC @ T

    #  s-----T-<- 2*
    #  |     ^
    #  |     0
    #  T-<1
    #  v
    #  3*
    matC = matC.permute((1, 2), (0, 3))

    #  s-----T-<- 2*
    #  |     |
    #  |     |
    #  T-----a--< 0
    #  v     v
    #  3*    1
    matC = a2 @ matC

    #  s-----T-<- 1
    #  |     |
    #  |     |
    #  T-----a--< 0
    #  v     v
    #  3*    2*
    matC = matC.permute((0, 2), (1, 3))

    news, U = matC.truncated_eigh(chi_target)
    news /= news.norm()
    #  1     0
    #  v     v
    #  |--U--|
    #     v
    #     2*

    #      |<0    2*-<-Tp-<-0
    #  2*<-U           ^
    #      |<1*        1
    newT = U.permute((1,), (0, 2))

    #      |----T-<-0
    #  3*<-U    ^
    #      |<2* 1
    newT = Tp @ newT

    #      |----T-<-2*
    #  3*<-U    ^
    #      |<1  0
    newT = newT.permute((1, 2), (0, 3))

    #      |---T-<-2*
    #  3*<-U   |
    #      |---a-<-0
    #          ^
    #          1
    newT = a @ newT

    #      |----T-<-1
    #  3*<-U    |
    #      |----a-<-0
    #          ^
    #          2*
    newT = newT.permute((0, 2), (1, 3))

    #      |---T---|
    #  2*<-U   |   U*<0
    #      |---a---|
    #          ^
    #          1*
    newT = U.dagger() @ newT
    newT /= newT.norm()
    return news, newT


def env_1site(s, T):
    """
    Compute CTMRG envrionement for one site
    """
    Tp = T.permute((0, 1), (2,))
    corner = Tp * s
    #   s-----T<-0
    #   |     ^
    #   T<2*  1
    #   |
    #   s->3*
    half = corner @ corner.permute((0,), (1, 2))

    #   s-----T<-2*
    #   |     ^
    #   T<1   0
    #   |
    #   s->3*
    left = half.permute((1, 2), (0, 3))

    #           0-<-s
    #               |
    #            3*>T
    #          2*   |
    #          v    |
    #      1->-T----s
    right = half.permute((3, 0), (1, 2))

    #   s-----T-----s
    #   |     ^     |
    #   |     0     |
    #   T<1      3*>T
    #   |     2*    |
    #   |     ^     |
    #   s-----T-----s
    full = left @ right

    #   s-----T-----s
    #   |     |     |
    #   |     0     |
    #   T-3*      1-T
    #   |     2*    |
    #   |     |     |
    #   s-----T-----s
    return full.permute((0, 3), (2, 1))


def env_2sites(s, T):
    """
    Compute CTMRG envrionement for two sites
    """
    Tp = T.permute((0, 1), (2,))
    env = Tp * s
    #   s-----T<-0
    #   |     ^
    #   T<2*  1
    #   |
    #   s->3*
    env = env @ env.permute((0,), (1, 2))

    #   s-----T----T-<-0
    #   |     ^    ^
    #   T<3*  2*   1
    #   |
    #   s->4*
    env = Tp @ env.permute((0,), (1, 2, 3))

    #   s-----T----T-<-3*
    #   |     ^    ^
    #   T<2   1    0
    #   |
    #   s->4*
    left = env.permute((1, 2, 3), (0, 4))

    #             4*-<-s
    #                  |
    #               3*>T
    #        1    2*   |
    #        v    v    |
    #    0->-T----T----s
    # rotation for half

    #              0-<-s
    #                  |
    #               4*>T
    #        2*   3*   |
    #        v    v    |
    #    1->-T----T----s
    right = env.permute((4, 0), (1, 2, 3))

    #   s----T----T-----s
    #   |    ^    ^     |
    #   |    1    0     |
    #   T<2          5*>T
    #   |    3*   4*    |
    #   |    ^    ^     |
    #   s----T----T-----s
    return left @ right


# ==========  Run simulation without symmetry  =======
print("#" * 88)
print("Contract 2D classical Ising model partition function using CTMRG")
print(f"inverse temperature beta = {beta}   (critical beta is {beta_crit:.5f})")
print("CTMRG paramters:")
print(f"D = {D}")
print(f"chi_target = {chi_target}")
print(f"n_iter = {n_iter}")

# initialize CTMRG
s, T, a, a2 = s0_asym, T0_asym, a_asym, a_asym2

# iterate CTMRG
print("\n" + "#" * 88)
print("Begin CTMRG with trivial symmetry")
print(f"Iterate CTMRG for n_iter = {n_iter}...")
for _ in range(n_iter):
    s, T = iterate_ctm(s, T, a, a2)

print("Done with CTMRG iterations.")

# compute observables
print("Compute 1 and 2 site environments...")
env1 = env_1site(s, T)
env2 = env_2sites(s, T)
print("Done with environment.")

# more convenient to cast back to array to contract all legs
env1 = env1.toarray()
env2 = env2.toarray()

# compute magnetization
mr = env1.ravel() @ magn_op.ravel()  # unnormalized magnetization
nr = env1.ravel() @ a0.ravel()  # norm
m = mr / nr  # magnetization
m_ex = exact_magn(beta)
m_err = abs(m_ex - abs(m))

# compute energy
nr = np.einsum("abcdef,bgdc,afeg->", env2, a0, a0)
sdsr = np.einsum("abcdef,bgdc,afeg->", env2, magn_op, magn_op)
sds = sdsr / nr
energy = -2 * sds  # ferromagnetic
e_ex = exact_energy(beta)
e_err = abs(e_ex - energy)

print("\nresults without symmetry:")
print(f"magnetization / exact / error:  {m: .5f}   /  {m_ex:.5f}   /  {m_err:.1e}")
print(f"energy / exact / error       :  {energy:.5f}   / {e_ex:.5f}   /  {e_err:.1e}")


# ==========  Initialize Z2 symmetric tensors  =======
# now we can do the same while imposing Z2 symmetry
# Z2 = {1, -1} acts on a bond as {Id, Sx}
# we can diagonalize this group action with
K = np.array([[-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(2)
# K is symmetric and squares to 1:
assert lg.norm(K - K.T) < 1e-15
assert lg.norm(K @ K - np.eye(2)) < 1e-15
# we are now going to insert K @ K = Id on every bond
# K diagonalizes a bond such that a it now carries Z2 irreps (odd, even)

# Z2 symmetry is not explicitly implemented in frostspin
# however we can use U(1) symmetry as Z2, using int8 and the rules
# 0 + 0 = 0
# 0 + -128 = -128
# -128 + -128 = 0

# define Z2 representations
bond_repr_Z2 = np.array([-128, 0], dtype=np.int8)
row_Z2 = (bond_repr_Z2,) * 2
col_Z2 = (bond_repr_Z2,) * 2

a0_diag = np.einsum("abcd,ai,bj,ck,dl->ijkl", a0, K, K, K, K)
a_Z2 = U1SymmetricTensor.from_array(a0_diag, row_Z2, col_Z2, signature=siga)

# signature still does not matter as Z2 action is self-dual
a_Z2_2 = U1SymmetricTensor(row_Z2, col_Z2, a_Z2.blocks, a_Z2.block_irreps, siga2)

# initialize symmetric corner eigenvalues s with even and odd blocks
d0o = chi0 // 2
s_odd = rng.normal(size=(d0o,))
s_odd = s_odd[np.abs(s_odd).argsort()[::-1]] / lg.norm(s_odd)
d0e = chi0 // 2  # could be different from d0o
s_even = rng.normal(size=(d0e,))
s_even = s_even[np.abs(s_even).argsort()[::-1]] / lg.norm(s_even)
s_blocks = (s_odd, s_even)

# cast to SymmetricTensor
corner_repr_Z2 = np.array([-128] * d0o + [0] * d0e, dtype=np.int8)
s_block_irreps = np.array([-128, 0], dtype=np.int8)
irrep_dims = [1, 1]  # Z2 odd and even irreps have dimension 1
s0_Z2 = DiagonalTensor(
    s_blocks, corner_repr_Z2, s_block_irreps, irrep_dims, a_Z2.symmetry()
)

# initialize edge tensor T directly as a SymmetricTensor
row_reps = (corner_repr_Z2,)
col_reps = (bond_repr_Z2, corner_repr_Z2)
T0_Z2 = U1SymmetricTensor.random(row_reps, col_reps, signature=sigT, rng=rng)
temp = U1SymmetricTensor(
    row_reps, col_reps, T0_Z2.blocks, T0_Z2.block_irreps, [1, 0, 0]
)
temp = temp.permute((2,), (1, 0))
T0_Z2 = T0_Z2 + temp  # impose symmetric T
T0_Z2 /= T0_Z2.norm()

# with these new tensors, we can run the simulation exactly as before.
# There is no difference in the interface between AsymmetricTensor and
# U1SymmetricTensor


# ==========  Simulation with Z2 symmetry  =======
print("\n" + "#" * 88)

# initialize CTMRG
s, T, a, a2 = s0_Z2, T0_Z2, a_Z2, a_Z2_2

# iterate CTMRG
print("Begin CTMRG with Z2 symmetry")
print(f"Iterate CTMRG for n_iter = {n_iter}...")
for _ in range(n_iter):
    s, T = iterate_ctm(s, T, a, a2)

print("Done with CTMRG iterations")

# compute observables
print("Compute 1 and 2 site environments...")
env1 = env_1site(s, T)
env2 = env_2sites(s, T)
print("Done with environment.")

# more convenient to cast back to array to contract all legs
env1 = env1.toarray()
env2 = env2.toarray()

# compute magnetization
# now that we imposed Z2, magnetization is always 0 as we have
# exactly the same probability for up and down sectors
# first we need to cancel K on all bonds
magn_op_diag = np.einsum("abcd,ai,bj,ck,dl->ijkl", magn_op, K, K, K, K)
mr = env1.ravel() @ magn_op_diag.ravel()  # unnormalized magnetization
nr = env1.ravel() @ a0_diag.ravel()  # norm
m_Z2 = mr / nr  # magnetization

# compute energy
nr = np.einsum("abcdef,bgdc,afeg->", env2, a0_diag, a0_diag)
sdsr = np.einsum("abcdef,bgdc,afeg->", env2, magn_op_diag, magn_op_diag)
sds = sdsr / nr
energy_Z2 = -2 * sds  # ferromagnetic
e_err_Z2 = abs(e_ex - energy_Z2)

print("\nresults with Z2 symmetry:")
print(f"magnetization (has to be 0)  :  {m_Z2: .1e}")
print("energy / exact / error       :", end="")
print(f"  {energy_Z2:.5f}   / {e_ex:.5f}   /  {e_err_Z2:.1e}")

# Note: performances are better without symmetry than with Z2
# this is to be expected: tensors are small and Z2 is a small group
# Contraction and diagonlization are fast enough without symmetry
# the gain obtained with Z2 there does not compensate the extra cost in Z2 permute

# concerning numerial accuracy, for a fixed corner dimension in the symmetry broken
# phase one may get a better precision by converging to a Z2 broken fixed point
# (either up or down), instead of converging to the Z2 symmetric fixed point
