import numpy as np


def check_tensor_bond_indices(tensor_bond_indices):
    """
    Check tensor_bond_indices abides by conventions:
    - virtual legs are labelled by integers from 0 to n_bonds
    - each virtual leg appears on exactly two different tensors
    - for a given tensor, all virtual legs differ

    Note that the number of virtual legs is allowed to be different on different
    tensors.
    """
    n_bonds = max(max(tbi) for tbi in tensor_bond_indices) + 1
    count = np.zeros((n_bonds,), dtype=int)
    for i, tbi in enumerate(tensor_bond_indices):
        for j, leg in enumerate(tbi):
            if not 0 <= leg < n_bonds:
                raise ValueError("Bond indices must be 0 to n_bonds - 1")
            if leg in tbi[j + 1 :]:
                raise ValueError(f"Leg {leg} apperars twice in tensor {i}")
            count[leg] += 1

    for i, c in enumerate(count):
        if c != 2:
            raise ValueError(f"Virtual bond {i} appears {c} times")


def check_hamiltonians(hamiltonians):
    ST = type(hamiltonians[0])
    for i, h in enumerate(hamiltonians):
        if type(h) is not ST:
            raise ValueError(f"Invalid type for Hamiltonian {i}")
        if h.ndim != 4 or h.n_row_reps != 2:
            raise ValueError(f"Hamiltonian {i} has invalid shape")
        for a in range(2):
            if h.row_reps[a].shape != h.col_reps[a].shape:
                raise ValueError(f"Hamiltonian {i} has invalid representations")
            if (h.row_reps[a] != h.col_reps[a]).any():
                raise ValueError(f"Hamiltonian {i} has invalid representations")
            if not h.signature[a] ^ h.signature[a + 2]:
                raise ValueError(f"Hamiltonian {i} has invalid signature")
