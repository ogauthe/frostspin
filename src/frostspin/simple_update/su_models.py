import numpy as np

"""
Define update data for common models.
"""


_models = ["J1-J2", "checkerboard", "Shastry-Sutherland"]


j1_j2_models = {
    # J1-J2 model on the square lattice using an AB//CD plaquette.
    # The first Hamiltonian is the first neighbor Hamiltonian h1, the second Hamiltonian
    # is h2/2. The factor 2 comes from the two possible intermediate sites.
    "J1-J2": np.array(
        [
            # b1 b2 h iL iR  im
            [0, 0, 0, 0, 2, -1],
            [1, 1, 0, 0, 1, -1],
            [2, 2, 0, 0, 2, -1],
            [3, 3, 0, 0, 1, -1],
            [4, 4, 0, 3, 1, -1],
            [5, 5, 0, 3, 1, -1],
            [6, 6, 0, 3, 2, -1],
            [7, 7, 0, 3, 2, -1],
            [0, 7, 1, 0, 3, 2],
            [4, 3, 1, 3, 0, 1],
            [6, 0, 1, 3, 0, 2],
            [1, 4, 1, 0, 3, 1],
            [5, 1, 1, 3, 0, 1],
            [2, 6, 1, 0, 3, 2],
            [7, 2, 1, 3, 0, 2],
            [3, 5, 1, 0, 3, 1],
            [4, 6, 1, 1, 2, 3],
            [0, 1, 1, 2, 1, 0],
            [3, 0, 1, 1, 2, 0],
            [7, 4, 1, 2, 1, 3],
            [5, 7, 1, 1, 2, 3],
            [2, 3, 1, 2, 1, 0],
            [1, 2, 1, 1, 2, 0],
            [6, 5, 1, 2, 1, 3],
        ]
    ),
    # J1-J2 model on the checkerboard lattice using an AB//CD plaquette
    # The first Hamiltonian is the first neighbor Hamiltonian h1, the second Hamiltonian
    # is h2/2. The factor 2 comes from the two possible intermediate sites.
    "checkerboard": np.array(
        [
            [0, 0, 0, 0, 2, -1],
            [1, 1, 0, 0, 1, -1],
            [2, 2, 0, 0, 2, -1],
            [3, 3, 0, 0, 1, -1],
            [4, 4, 0, 3, 1, -1],
            [5, 5, 0, 3, 1, -1],
            [6, 6, 0, 3, 2, -1],
            [7, 7, 0, 3, 2, -1],
            [0, 7, 1, 0, 3, 2],
            [4, 3, 1, 3, 0, 1],
            [5, 1, 1, 3, 0, 1],
            [2, 6, 1, 0, 3, 2],
            [3, 0, 1, 1, 2, 0],
            [7, 4, 1, 2, 1, 3],
            [1, 2, 1, 1, 2, 0],
            [6, 5, 1, 2, 1, 3],
        ]
    ),
    # J1-J2 model on the Shastry-Sutherland lattice using an AB//CD plaquette
    # The first Hamiltonian is the first neighbor Hamiltonian h1, the second Hamiltonian
    # is h2/2. The factor 2 comes from the two possible intermediate sites.
    "Shastry-Sutherland": np.array(
        [
            [0, 0, 0, 0, 2, -1],
            [1, 1, 0, 0, 1, -1],
            [2, 2, 0, 0, 2, -1],
            [3, 3, 0, 0, 1, -1],
            [4, 4, 0, 3, 1, -1],
            [5, 5, 0, 3, 1, -1],
            [6, 6, 0, 3, 2, -1],
            [7, 7, 0, 3, 2, -1],
            [0, 7, 1, 0, 3, 2],
            [4, 3, 1, 3, 0, 1],
            [1, 2, 1, 1, 2, 0],
            [6, 5, 1, 2, 1, 3],
        ]
    ),
}
