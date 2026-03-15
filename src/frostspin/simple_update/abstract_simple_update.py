import numpy as np


class AbstractSimpleUpdate:
    @property
    def Dmax(self):
        return max(max(t.shape[2:]) for t in self._tensors)

    @property
    def beta(self):
        return self._beta

    @property
    def n_bonds(self):
        return self._n_bonds

    @property
    def n_tensors(self):
        return self._n_tensors

    @property
    def tau(self):
        return self._tau

    @property
    def logZ(self):
        return self._logZ / self._n_tensors

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        self._gates = tuple((-tau * h).expm() for h in self._hamiltonians)
        # 2nd order Trotter Suzuki + rho is quadratic in psi
        self._dbeta = 2 * (1 + self._is_second_order) * tau

    def __str__(self):
        s = repr(self)
        s = s + f"\nn_tensors = {self._n_tensors}, n_bonds = {self._n_bonds}"
        s = s + f"\nDmax = {self.Dmax}, tau = {self._tau}, rcutoff = {self.rcutoff}, "
        return s + f"degen_ratio = {self.degen_ratio}"

    # do not define it as property to mimic ST.symmetry() behavior
    def symmetry(self):
        return self._symmetry

    def get_bond_representation(self, i):
        for j in range(self._n_tensors):
            ind = (self._tensor_bond_indices[j] == i).nonzero()[0]
            if ind.size:
                return self._tensors[j].col_reps[ind[0]]
        raise ValueError(f"Unknown bond {i}")

    def get_bond_representations(self):
        """
        Obtain representations associated with all unit cell bonds.
        """
        return [self.get_bond_representation(i) for i in range(self._n_bonds)]

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as s_ent = -sum_i p_i log_p_i
        """
        return np.array([-w @ np.log(w) for w in self.get_weights()])

    def get_weights(self, *, sort=True):
        """
        Return simple update weights for each bond with degeneracies.
        """
        return [w.toarray(sort=sort) for w in self._weights]

    def get_tensors(self):
        """
        Returns
        -------
        tensors : tuple of _n_tensors SymmetricTensor
            Optimized tensors, with sqrt(weights) on all virtual legs.
        """
        sqw = [w**-0.5 for w in self._weights]
        tensors = []
        for i, t0 in enumerate(self._tensors):
            # we already imposed the two first legs to be physical and ancilla in the
            # default configuration. Add weights on the virtual legs.
            rswap = (0, 1, *range(3, t0.ndim))
            t = t0
            for leg in self._tensor_bond_indices[i]:
                t = t.permute(rswap, (2,))
                t = t * sqw[leg]
            t = t.permute((0, 1), tuple(range(2, t0.ndim)))
            tensors.append(t)
        return tensors

    def evolve(self, beta_evolve):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
        if self.verbosity > 0:
            print(
                f"Evolve in imaginary time for beta from {self._beta:.6g} to "
                f"{self._beta + beta_evolve:.6g}...",
                flush=True,
            )
        if beta_evolve < -1e-12:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        # 2nd order update: update 0 is special as it may be squared or not
        # depending whether it is at the very beginning / end of update sequence
        self._initialize_update()
        for _ in range(niter - 1):  # there is 1 step out of the loop
            for j in range(self._n_updates):
                self._elementary_update(j)
        self._finalize_update()
        self._beta += niter * self._dbeta
        return

    def _initialize_update(self):
        # For a given tensor, its structure is defined by tensor_bond_indices between
        # updates. This structure needs to change at each update the tensor is involved
        # in. To avoid swappping back and forth for each elementary update, the tensor
        # structure constraint is relaxed inside evolve and changes within updates, as
        # defined by lperm, rperm and mperm. One cycle of updates will leave tensors
        # in a given final state structure, one needs to start from this state to be
        # able to loop over it.
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permute(*self._initial_swap[i])

        # run last elementary update with time step tau instead of 2*tau
        self._elementary_update(self._n_updates)

        # run one cycle of update but the first one
        for j in range(1, self._n_updates):
            self._elementary_update(j)

    def _finalize_update(self):
        if self._is_second_order:
            # run the very last/first update with time step tau
            self._elementary_update(self._n_updates)

        # come back to default form, as defined in tensor_bond_indices
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permute(*self._final_swap[i])

    def save_to_file(self, savefile, **additional_data):
        """
        Save SimpleUpdate in given file.

        Parameters
        ----------
        savefile : str
            Path to savefile.
        additional_data : dic
            Additional data to save together.
        """
        data = {
            "_SimpleUpdate_symmetry": self._symmetry,
            "_SimpleUpdate_classname": self._classname,
            "_SimpleUpdate_D": self.D,
            "_SimpleUpdate_beta": self._beta,
            "_SimpleUpdate_tau": self._tau,
            "_SimpleUpdate_logZ": self._logZ,
            "_SimpleUpdate_is_second_order": self._is_second_order,
            "_SimpleUpdate_rcutoff": self.rcutoff,
            "_SimpleUpdate_degen_ratio": self.degen_ratio,
            "_SimpleUpdate_raw_update_data": self._raw_update_data,
            "_SimpleUpdate_n_bonds": self._n_bonds,
            "_SimpleUpdate_n_hamiltonians": len(self._raw_hamilts),
            "_SimpleUpdate_n_tensors": self._n_tensors,
        }
        for i, h in enumerate(self._raw_hamilts):
            data |= h.get_data_dic(prefix=f"_SimpleUpdate_hamiltonian_{i}")

        for i, tbi in enumerate(self._tensor_bond_indices):
            data |= self._tensors[i].get_data_dic(prefix=f"_SimpleUpdate_tensor_{i}")
            data[f"_SimpleUpdate_tensor_bond_indices_{i}"] = tbi

        for i, w in enumerate(self._weights):
            data |= w.get_data_dic(prefix=f"_SimpleUpdate_weights_{i}")

        np.savez_compressed(savefile, **data, **additional_data)
        if self.verbosity > 0:
            print("Simple update saved in file", savefile)
