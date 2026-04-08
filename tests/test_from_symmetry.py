#!/usr/bin/env python

import pytest

from frostspin import (
    AsymmetricTensor,
    O2SymmetricTensor,
    SU2SymmetricTensor,
    SymmetricTensor,
    U1SymmetricTensor,
)

SYMMETRY_TYPES = [
    ("U1", U1SymmetricTensor),
    ("SU2", SU2SymmetricTensor),
    ("O2", O2SymmetricTensor),
    ("Trivial", AsymmetricTensor),
]


@pytest.mark.parametrize(("symmetry", "expected"), SYMMETRY_TYPES)
def test_from_symmetry(symmetry, expected):
    assert SymmetricTensor.from_symmetry(symmetry) is expected


def test_from_symmetry_unknown():
    with pytest.raises(RuntimeError, match="Unknown symmetry"):
        SymmetricTensor.from_symmetry("unknown")


if __name__ == "__main__":
    for symmetry, expected in SYMMETRY_TYPES:
        test_from_symmetry(symmetry, expected)
    test_from_symmetry_unknown()
