import scipy.linalg as lg

from frostspin import ASSERT_TOL

symmetric_tensor_types = {}


def get_symmetric_tensor_type(symmetry):
    """
    Get SymmetricTensor subclass implementing symmetry specified by 'symmetry'.

    Parameters
    ----------
    symmetry : str
        Symmetry group. Must match implemented symmetry. Currently implemented
        symmetries are 'trivial', 'U1' and 'SU2'.
    """
    try:
        st_type = symmetric_tensor_types[symmetry]
    except KeyError:
        msg = f"Unknown symmetry '{symmetry}'"
        raise RuntimeError(msg) from None
    return st_type


def check_norm(t1, t2, *, tol=ASSERT_TOL):
    try:
        n1 = t1.norm()
    except AttributeError:
        n1 = lg.norm(t1)
    try:
        n2 = t2.norm()
    except AttributeError:
        n2 = lg.norm(t2)
    dn = abs(n1 - n2)
    b = dn > tol * max(n1, n2)
    if b:
        print(f"WARNING: norm is different: {dn:.1e} > {tol * max(n1, n2):.1e}")
    return not b
