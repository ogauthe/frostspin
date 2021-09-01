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
        if self.verbosity > 1:
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
        self._value = None
        self._last_value = None
        self._delta_list = []
        self._total_time = 0.0

    def warmup(self, warmup_iter):
        if self.verbosity > 0:
            print(f"Converge for warmup = {warmup_iter} iterations")

        t0 = time.time() - self._total_time
        for i in range(warmup_iter):
            self.iterate()
            if self.verbosity > 1:
                print(f"i = {self._niter}, t = {time.time()-t0:.1f}")

        self._total_time = time.time() - t0
        if self.verbosity > 0:
            print(f"{warmup_iter} warmup iterations finished.")

    def converge(self, tol, shift=5, stuck=0.99, maxiter=2147483647):
        """
        Converge process. Convergence is reached if delta = distance(value, last_value)
        gets smaller than tol. Convergence fails if niter > maxiter or if the ratio
        delta / delta(shift iterations before) > stuck.

        Parameters
        ----------
        tol : float
            Tolerance to estimate convergence occurred.
        shift : int
            Number of iterations between current delta and delta used to test if stuck.
        stuck : float
            Value to consider process is stuck.
        maxiter : int
            Maximal number of iteration. Default is 2**31 - 1. Note that maxiter is
            is compared to niter, which includes warmup iterations.
        """
        if self.verbosity > 0:
            print(f"Converge up to tol = {tol}")
            print(f"{maxiter - self._niter} iterations left before reaching maxiter.")

        t0 = time.time() - self._total_time
        self._value = self.get_value()
        while 1:
            self.iterate()
            self._last_value = self._value
            self._value = self.get_value()
            delta = self._distance(self._last_value, self.value)
            self._delta_list.append(delta)
            if self.verbosity > 1:
                t = time.time() - t0
                print(f"i = {self._niter}, t = {t:.1f}, delta = {delta:.3e}")
            if delta < tol:
                msg = "Convergence succeded!"
                break
            if self._niter > maxiter:
                msg = (
                    f"Convergence failed: maxiter = {maxiter} reached before "
                    "convergence!"
                )
                break
            if (
                len(self._delta_list) > shift
                and delta / self._delta_list[-shift] > stuck
            ):
                msg = "Convergence failed: delta has stabilized higher than tol!"
                break

        if self.verbosity > 0:
            print(msg)
            print(
                f"Converger exit parameters: niter = {self._niter},",
                f"t = {time.time()-t0:.1f}, delta = {delta:.3e}",
            )
        return delta
