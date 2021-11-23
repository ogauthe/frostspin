from .asymmetric_tensor import AsymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor
from .su2_symmetric_tensor import SU2_SymmetricTensor

symmetric_tensor_types = {
    "{e}": AsymmetricTensor,
    "U(1)": U1_SymmetricTensor,
    "SU(2)": SU2_SymmetricTensor,
}


def get_symmetric_tensor_type(symmetry):
    """
    Get SymmetricTensor subclass implementing symmetry specified by 'symmetry'.

    Parameters
    ----------
    symmetry : str
        Symmetry group. Must match implemented symmetry. Currently implemented
        symmetries are '{e}', 'U(1)' and 'SU(2)'.
    """
    try:
        st_type = symmetric_tensor_types[symmetry]
    except KeyError:
        raise ValueError(f"Unknown symmetry '{symmetry}'")
    return st_type
