# froSTspin
A Symmetric Tensor library for lattice spin systems at finite temperature.

## Note
This library is no more actively maintained. Consider using more complete libraries such as [ITensors](https://github.com/ITensor/ITensors.jl), [TensorKit](https://github.com/Jutho/TensorKit.jl) or [TeNPy](https://github.com/tenpy/tenpy).

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
Clone this repository and install it with locally with pip
```shell
git clone git@github.com:ogauthe/frostspin.git
uv pip install -e /path/to/frostspin
uv sync
python /path/to/frostspin/examples/ising_classical_ctmrg.py
```

## Usage
See examples and tests on how to use it.
To improve performances, consider running with `python -O` to disable `assert` statements.
