import numpy as np
from quop_mpi import Ansatz
from quop_mpi.propagator import discrete, composite, momentum
from quop_mpi.state import position_grid
from quop_mpi.param.rand import uniform
from quop_mpi.__lib import fCQAOA
from quop_mpi.state import equal

class multivariable(Ansatz):

    def set_independent_t(self, independent):
        if independent:
            self.UW_n_params = len(Ns)
        else:
            self.UW_n_params = 1
            self.UW.unitary_n_params = 1

        self.set_unitaries([self.UQ, self.UW])

    def set_qualities(self, function):

        self.UQ.operator_kwargs = {"function":function}
        self.set_observables(0)

    def set_mixer(self, Cs):
        self.UW.operator_kwargs = {"Ns":self.Ns, "Cs":Cs}

    def get_maximum_point(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            probs = np.abs(self.final_state) ** 2
            indx = np.argmax(probs)
            maxes = self.COMM_OPT.gather(probs[indx], root=0)
            indxs = self.COMM_OPT.gather(indx + self.local_i_offset, root=0)

            if self.COMM_OPT.Get_rank() == 0:

                best = np.argmax(maxes)
                inds = self.unitaries[1].fCQAOA.continuous.get_index(
                    indxs[best] + 1, self.Ns, self.UW.strides
                )
                grid_points = (inds - 1) * self.deltas
                grid_points += self.mins
                best_value = self.UQ.operator_kwargs['function'](grid_points)
                return best_value, grid_points, maxes[best]

    def grid_point_from_index(self, index):

        if self.COMM_OPT.Get_rank() == 0:
        
            inds = self.unitaries[1].fCQAOA.continuous.get_index(
                index + 1, self.Ns, self.UW.strides
            )
            grid_points = (inds - 1) * self.deltas
            grid_points += self.mins
            return grid_points

    def save(self, file_name, config_name, action="a"):

        if self.colours[self.COMM.Get_rank()] == 0:
            self.observables = np.real(self.observables)

        super().save(file_name, config_name, action)

        if self.colours[self.COMM.Get_rank()] == 0:
            self.observables = np.real(self.observables)

            from quop_mpi.__lib import fqwoa_mpi
            import h5py

            if self.COMM_OPT.Get_rank() == 0:

                qgrid = fCQAOA.continuous.gen_local_grid(
                    self.system_size,
                    self.Ns,
                    self.UW.strides,
                    self.deltas,
                    self.mins,
                    0,
                    self.system_size,
                )

                File = h5py.File(file_name + ".h5", "a")

                File[self.config_name].attrs["Ns"] = self.Ns


                File.create_dataset(
                    self.config_name + "/position_grid",
                    data=qgrid,
                    dtype=np.float64,
                )

                File.close()

class qmoa(multivariable):

    def __init__(self, Ns, deltas, mins, *args, **kwargs):

        self.Ns = Ns
        self.deltas = deltas
        self.mins = mins

        system_size = 1 
        for N in Ns:
            system_size *= N

        super().__init__(system_size, *args, **kwargs)

        self.continuous_function = None # must be defined using set_qualities
        self.graphs = Ns # complete graphs by default
        self.UW_n_params = len(Ns)

        self.UQ = discrete.unitary(
            Ns,
            deltas,
            mins,
            discrete.operator.grid,
            operator_kwargs = {"function": None},
            parameter_function=uniform,
        )
        
        self.UW = composite.unitary(
            Ns,
            composite.operator.ith,
            operator_kwargs={
                "Ns": Ns,
                "Cs": Ns,
            },
            parameter_function=uniform,
            unitary_n_params = self.UW_n_params,
        )

        self.set_unitaries([self.UQ, self.UW])

class qowe(multivariable):

    def __init__(self, Ns, deltas, mins, *args, **kwargs):

        self.Ns = Ns
        self.deltas = deltas
        self.mins = mins

        system_size = 1 
        for N in Ns:
            system_size *= N

        super().__init__(system_size, *args, **kwargs)

        self.continuous_function = None # must be defined using set_qualities
        self.graphs = Ns # complete graphs by default
        self.UW_n_params = len(Ns)
        
        self.deltask = np.array([
                2*np.pi/(n*delta) for (delta, n) in zip(self.deltas, self.Ns)
                ], dtype = np.float64)
        
        self.minsk = np.array([
                -(n/2)*delta for (delta, n) in zip(self.deltask, self.Ns)
                ], dtype = np.float64)

        self.UQ = discrete.unitary(
            self.Ns,
            self.deltas,
            self.mins,
            discrete.operator.grid,
            operator_kwargs = {"function": None},
            parameter_function=uniform,
        )
        
        self.UW = momentum.unitary(
            self.Ns,
            self.mins,
            self.minsk,
            self.deltas,
            self.deltask,
            momentum.operator.magnitude_squared,
            parameter_function=uniform,
            unitary_n_params = len(self.Ns),
        )
        
        self.set_unitaries([self.UQ, self.UW])

        self.set_initial_state(
                position_grid,
                kwargs = {
                    "Ns": self.Ns,
                    "deltasq":self.deltas,
                    "minsq":self.mins
                    }
                )
