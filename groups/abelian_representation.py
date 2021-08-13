# import bisect

import numpy as np


class AbelianRepresentation(object):
    """
    Minimalist class for Abelian representations. A non-abelian symmetry is a much more
    complicate object with quite different features. Implementation becomes simpler
    without defining a common Representation base class.
    """

    _symmetry = NotImplemented  # string to label symmetry group

    def __init__(self, degen, irreps):
        """
        Construct an abelian representation.

        Parameters
        ----------
        degen : integer array
            Degeneracy of given irreps.
        irreps : integer array
            Irreps of the symmetry group, labeled by integers.
        """
        assert degen.shape == irreps.shape == (degen.size,)
        assert degen.any()  # finite groups => easier to allow zeros
        assert (irreps.argsort() == np.arange(irreps.size)).all()
        assert np.issubdtype(degen.dtype, np.integer)
        assert np.issubdtype(irreps.dtype, np.integer)
        self._degen = degen
        self._irreps = irreps
        self._dim = degen.sum()
        self._n_irr = degen.size

    @property
    def dim(self):
        return self._dim

    @property
    def degen(self):
        return self._degen

    @property
    def irreps(self):
        return self._irreps

    @property
    def n_irr(self):
        return self._n_irr

    def __eq__(self, other):
        assert type(self) == type(other)
        return (
            self._n_irr == other._n_irr
            and (self._irreps == other._irreps).all()
            and (self._degen == other._degen).all()
        )

    # define interface for subclasses
    def conjugate(self):  # conjugate representation
        return NotImplemented

    def __add__(self, other):  # sum of 2 representations
        return NotImplemented

    def __mul__(self, other):  # product of 2 representations
        return NotImplemented

    @classmethod  # product of several representation
    def combine_representations(cls, *reps):
        return NotImplemented


class FiniteGroupAbelianRepresentation(AbelianRepresentation):
    # for finite groups, keeping all irreps and allowing 0 as degen is much simpler.
    # However, __init__ must keep its signature to be used in SymmetricTensor.
    _symmetry = NotImplemented
    _irreps = NotImplemented  # class member
    _n_irr = NotImplemented  # class member

    def __init__(self, degen, irreps=None):  # keep irreps for parent compatibility
        assert np.issubdtype(degen.dtype, np.integer)
        assert irreps is None or (irreps == self._irreps).all()
        self._degen = degen
        self._dim = degen.sum()

    def __add__(self, other):
        assert type(self) == type(other)
        return type(self)(self._degen + other._degen)


class AsymRepresentation(FiniteGroupAbelianRepresentation):
    _symmetry = "{e}"
    _irreps = np.zeros((1,), dtype=np.int8)
    _n_irr = 1

    def conjugate(self):
        return self

    def __mul__(self, other):
        assert other._symmetry == "{e}"
        return AsymRepresentation(self._degen * other._degen)

    @classmethod
    def combine_representations(cls, *reps):
        return AsymRepresentation(np.product([rep._degen for rep in reps]))


class Z2_Representation(FiniteGroupAbelianRepresentation):
    _symmetry = "Z2"
    _irreps = np.array([0, 1], dtype=np.int8)
    _n_irr = 2

    def conjugate(self):
        return self

    def __mul__(self, other):
        assert other._symmetry == "Z2"
        degen = np.array(
            [
                self._degen[0] * other._degen[0] + self._degen[1] * other._degen[1],
                self._degen[1] * other._degen[0] + self._degen[0] * other._degen[1],
            ]
        )
        return Z2_Representation(degen)


class U1_Representation(AbelianRepresentation):
    _symmetry = "U1"

    def conjugate(self):
        return U1_Representation(self._degen[::-1], -self._irreps[::-1])

    def __mul__(self, other):
        irrmin = self._irreps[0] + other._irreps[0]
        irrmax = self._irreps[-1] + other._irreps[-1]
        degen = np.zeros((irrmax - irrmin + 1,))
        for d1, irr1 in zip(self._degen, self._irreps):
            for d2, irr2 in zip(other._degen, other._irreps):
                degen[irr1 + irr2 - irrmin] += d1 * d2
        irreps = degen.nonzero()[0]
        degen = np.ascontiguousarray(degen[irreps])
        irreps += irrmin
        return U1_Representation(degen, irreps)
