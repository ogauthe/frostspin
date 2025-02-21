from .asymmetric_tensor import AsymmetricTensor
from .o2_symmetric_tensor import O2_SymmetricTensor
from .su2_symmetric_tensor import SU2_SymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor

symmetric_tensor_types = {
    "trivial": AsymmetricTensor,
    "U1": U1_SymmetricTensor,
    "O2": O2_SymmetricTensor,
    "SU2": SU2_SymmetricTensor,
}


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
