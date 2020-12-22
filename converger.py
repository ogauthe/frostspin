import time

import numpy as np


class Converger(object):
    """
    Wrapper class to detect convergence success or failure.
    """

    def __init__(self, iterate, get_value, distance=None, verbosity=0):
        """
        Parameters
        ----------
        iterate : function with signature iterate() -> None
            function performing an iteration
        get_value : function with signature get_value() -> value
            function computing the value v whose convergence is searched. value is
            typically a numpy array.
        distance : function with signature (value, value) -> float
            function to evaluate distance between two values. If not provided, Frobenius
            norm of the difference is used.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        self.verbosity = verbosity
        if self.verbosity > 2:
            print("Initialize Converger")
        self._iterate = iterate
        self.get_value = get_value
        self.reset()
        if distance is None:
            self._distance = lambda a, b: np.linalg.norm(a - b)
        else:
            self._distance = distance

    @property
    def niter(self):
        return self._niter

    @property
    def delta_list(self):
        return self._delta_list

    @property
    def value(self):
        return self._value

    @property
    def last_value(self):
        return self._last_value

    def iterate(self):
        self._iterate()
        self._niter += 1

    def reset(self):
        self._niter = 0
        self._delta_list = []

    def converge(self, tol=1e-8, warmup=0, shift=5, stuck=0.99, maxiter=2147483647):
        """
        Converge process.

        Parameters
        ----------
        tol : float
            Tolerance to estimate convergence occurred.
        warmup : int
            Number of warmup iteration before any convergence check.
        maxiter : int
            Maximal number of iteration. Default is 2**31 - 1.
        """
        if self.verbosity > 0:
            print(f"Converge with tol = {tol}, warmup = {warmup}, maxiter = {maxiter}")

        t0 = time.time()
        for i in range(warmup):
            self.iterate()
            if self.verbosity > 1:
                print(f"niter = {self._niter}, t = {time.time()-t0:.1f}")

        if self.verbosity > 0:
            print(f"{warmup} warmup iterations done, t = {time.time()-t0:.1f}")

        shift_warm = shift + warmup
        self._value = self.get_value()
        while 1:
            self.iterate()
            self._last_value = self._value
            self._value = self.get_value()
            delta = self._distance(self._last_value, self.value)
            self._delta_list.append(delta)
            if self.verbosity > 1:
                print(
                    f"niter = {self._niter}, delta = {delta:.2e},",
                    f"t = {time.time()-t0:.1f}",
                )

            if delta < tol:
                return True, "success!"

            if self._niter > maxiter:
                return False, "Convergence did not occur before maxiter"

            if self._niter > shift_warm and delta / self._delta_list[-shift] > stuck:
                return False, "delta has stabilized higher than tol"
