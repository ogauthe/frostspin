# froSTspin
A Symmetric Tensor library for lattice spin systems at finite temperature.

## Symmetries
frostspin incorporates both abelian and non-abelian symmetries. It features a unified interface to interact with a SymmetricTensor, irrespective of the specific symmetry group ruling it.

Currently, frostspin supports the following symmetry groups:
- Z<sub>2</sub>
- U(1)
- O(2)
- SU(2)

## Algorithms
frostspin includes the following tensor network algorithms:
- simple update
- CTMRG

## Dependencies
- python >= 3.10
- numpy >= 1.23
- scipy >= 1.10
- sympy >= 1.11
- numba >= 0.57

## Install
The simplest is to git clone this repository and to add it to your PYTHONPATH.
```shell
git clone git@framagit.org:ogauthe/frostspin.git
export PYTHONPATH=$(pwd)/froSTspin:$PYTHONPATH
python -uO ./froSTspin/tests/test_u1_tensor.py
```

## Usage
See examples and tests on how to use it.
