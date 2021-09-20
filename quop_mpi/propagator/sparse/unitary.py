import numpy as np
from ...Unitary import Unitary
from ...__lib import fMPI

class unitary(Unitary):
    """Implements a mixing unitary with a circulant matrix operator exponent.

    See :class:`Unitary` for more information.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.unitary_n_params > 1:
            self.propagate = self.evolve_group
        else:
            self.propagate = self.evolve_single

        self.precision = "dp"

        self.W_row_starts = None
        self.W_col_indexes = None
        self.W_values = None

        self.one_norms = None
        self.num_norms = None
        self.W_num_rec_inds = None
        self.W_rec_disps = None
        self.W_num_send_inds = None
        self.W_send_disps = None
        self.W_local_col_inds = None
        self.W_rhs_send_inds = None


    def plan(self, system_size, MPI_COMM):

        size = MPI_COMM.Get_size()
        rank = MPI_COMM.Get_size()

        local_i = system_size // size

        if local_i * size != system_size:
            remainder = system_size - local_i * size
            if rank < remainder:
                local_i += 1
                print(local_i, flush = True)
        return local_i, local_i

    def copy_plan(self, ex_unitary):
        pass

    def gen_operator(self, *args):

        super().gen_operator(*args)

        self.W_row_starts, self.W_col_indexes, self.W_values = self.operator

        for i in range(len(self.W_values)):
            self.W_values[i][:] = -1j*self.W_values[i]

        self.W_num_rec_inds = []
        self.W_rec_disps = []
        self.W_num_send_inds = []
        self.W_send_disps = []
        self.W_local_col_inds = []
        self.W_rhs_send_inds = []
        self.one_norms = []
        self.num_norms = []

        for w_row_starts, w_col_indexes, w_values in zip(self.W_row_starts, self.W_col_indexes, self.W_values):

            w_num_rec_inds, w_rec_disps, w_num_send_inds, w_send_disps = fMPI.rec_a(
                    self.system_size,
                    w_row_starts,
                    w_col_indexes,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.W_num_rec_inds.append(w_num_rec_inds)
            self.W_rec_disps.append(w_rec_disps)
            self.W_num_send_inds.append(w_num_send_inds)
            self.W_send_disps.append(w_send_disps)

            w_local_col_inds, w_rhs_send_inds = fMPI.rec_b(
                    self.system_size,
                    np.sum(self.W_num_send_inds),
                    w_row_starts,
                    w_col_indexes,
                    w_num_rec_inds,
                    w_rec_disps,
                    w_num_send_inds,
                    w_send_disps,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.W_local_col_inds.append(w_local_col_inds)
            self.W_rhs_send_inds.append(w_rhs_send_inds)

            one_norms, num_norms = fMPI.one_norm_series(
                    self.system_size,
                    w_row_starts,
                    w_col_indexes,
                    w_values,
                    w_num_rec_inds,
                    w_rec_disps,
                    w_num_send_inds,
                    w_send_disps,
                    w_local_col_inds,
                    w_rhs_send_inds,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.one_norms.append(one_norms)
            self.num_norms.append(num_norms)

    def evolve_single(self, x):
        """Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        for i in range(len(self.W_row_starts)):

            self.final_state[:self.local_i] = fMPI.step(
                    self.system_size,
                    self.local_i,
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                    self.W_values[i],
                    self.W_num_rec_inds[i],
                    self.W_rec_disps[i],
                    self.W_num_send_inds[i],
                    self.W_send_disps[i],
                    self.W_local_col_inds[i],
                    self.W_rhs_send_inds[i],
                    np.abs(x),
                    self.initial_state[:self.local_i],
                    self.partition_table,
                    self.num_norms[i],
                    self.one_norms[i],
                    self.MPI_COMM.py2f(),
                    self.precision)

            self.initial_state[:] = self.final_state

    def evolve_group(self, x):
        """Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        for i in range(len(self.W_row_starts)):

            self.final_state[:self.local_i] = fMPI.step(
                    self.system_size,
                    self.local_i,
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                    self.W_values[i],
                    self.W_num_rec_inds[i],
                    self.W_rec_disps[i],
                    self.W_num_send_inds[i],
                    self.W_send_disps[i],
                    self.W_local_col_inds[i],
                    self.W_rhs_send_inds[i],
                    np.abs(x[i]),
                    self.initial_state[:self.local_i],
                    self.partition_table,
                    self.num_norms[i],
                    self.one_norms[i],
                    self.MPI_COMM.py2f(),
                    self.precision)

            self.initial_state[:] = self.final_state
