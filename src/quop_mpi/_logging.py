# cspell:words subcomm logfile
"""Logging and I/O mixin for QVA simulation results."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from mpi4py import MPI

from ._scope import scope
from ._utils._filenames import ensure_path_and_extension

if TYPE_CHECKING:
    from ._lib.context import Context
    from ._utils._comm_size import QuopMpiLayout


class Logging:
    """Mixin providing logging and I/O functionality for :class:`~quop_mpi.ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for CSV logging of simulation results and HDF5 saving of quantum states.
    """

    # Type hints for attributes provided by Ansatz
    if TYPE_CHECKING:
        subcomms: QuopMpiLayout | None
        MPI_COMM_WORLD: MPI.Intracomm
        system_size: int
        ansatz_depth: int
        local_i: int
        local_i_offset: int
        context: Context | None
        local_observables: np.ndarray
        ansatz_initial_state: np.ndarray | None
        variational_parameters: np.ndarray | None
        result: dict[str, Any] | None
        quop_result: dict[str, Any]
        optimiser_log: list[str] | None
        sampling: bool
        total_shots: int
        minimum_sampled: float
        shots_to_global_minimum: int | str
        state_norm: float | None
        time: float | None
        neval_mpi_jac: int

        def get_state_norm(self) -> float | None: ...

    def _init_logging(self):
        """Initialize logging-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.log: bool = False
        self.filename: str | None = None
        self.label: str | None = None
        self.log_action: str = "a"
        self.logfile: TextIO | None = None
        self.logfile_csv: Any | None = None
        self.n_log_fields: int = 6
        self.repeat: int = 1
        self.config_name: str | None = None

    @scope("world")
    def set_log(self, filename: str, label: str, action: str = "a") -> None:
        """Creates a CSV in which to save simulation results after a call to
        :meth:`~quop_mpi.ansatz.execute`.

        Parameters
        ----------
        filename : str
            path to the log file
        label : str
           simulation identifier
        action : {'a', 'w'}, optional
            'a' to append or 'w' overwrite, by default 'a'
        """

        self.filename = ensure_path_and_extension(filename, "csv")
        self.label = label
        self.log_action = action

        self.repeat = 1  # needed if logging results from the execute method

        self.setup_log = True

    @scope("world")
    def _gen_log(self) -> None:
        """Create or open a log file."""

        self.n_log_fields = 6

        if self.MPI_COMM_WORLD.Get_rank() == 0:

            if os.path.exists(self.filename) and self.log_action == "a":
                self.logfile = open(self.filename, "a", newline="")
                self.logfile_csv = csv.writer(self.logfile)
            else:

                self._create_new_logfile()

        self.log = True

    @scope("world")
    def _create_new_logfile(self) -> None:
        """Create a new log file, called by rank 0 at :attr:`~quop_mpi.ansatz.MPI_COMM_WORLD`
        only."""

        headings = [
            "label",
            "system_size",
            "ansatz_depth",
            "repeat",
            "state_norm",
            "simulation_time",
            "MPI_nodes",
            "MPI_jacobian_evaluations",
        ]

        if self.sampling:
            headings.extend(("total_shots", "minimum_sampled", "shots_to_global_minimum"))

        if self.optimiser_log is not None:
            headings.extend(iter(self.optimiser_log))

        self.logfile = open(self.filename, "w", newline="")
        self.logfile_csv = csv.writer(self.logfile)
        self.logfile_csv.writerow(headings)

    @scope("world")
    def _log_update(self) -> None:
        """Write simulation information to an active log file."""

        state_norm = self.get_state_norm()

        if self.MPI_COMM_WORLD.Get_rank() != 0:
            return

        log_output = [
            self.label,
            self.system_size,
            self.ansatz_depth,
            self.repeat,
            state_norm,
            self.time,
            self.subcomms.SUBCOMM.size,
            self.neval_mpi_jac,
        ]

        if self.sampling:
            log_output.extend(
                (
                    self.total_shots,
                    self.minimum_sampled,
                    self.shots_to_global_minimum,
                )
            )
        if self.optimiser_log is not None:
            log_output.extend(self.result[optimiser_log] for optimiser_log in self.optimiser_log)

        self.logfile_csv.writerow(log_output)

        self.logfile.flush()

    @scope("world")
    def _post_log(self) -> None:
        """Close the results log file on simulation completion."""

        if self.MPI_COMM_WORLD.Get_rank() == 0 and self.log:
            self.logfile.close()

    @scope("subcomm", returns="none")
    def save(self, file_name: str, config_name: str, action: str = "a") -> None:
        """Write the :term:`final state`, :term:`observables` and results
        summary to a HDf5 file.

        Parameters
        ----------
        file_name : str
            file path to saved data
        config_name : str
            simulation identifier
        action : {'a', 'w'}, optional
            'a' to append or 'w' to overwrite, by default 'a'

        Notes
        -----

        Data is saved into a :literal:`*.h5` file with the following structure.

        ::

            config_name/
                initial_state
                final_state
                observables

        The saved ``initial_state`` and ``final_state`` datasets are the
        algorithm-level ansatz states: ``initial_state`` comes from
        :attr:`ansatz_initial_state`, and ``final_state`` comes from the current
        context state after evolution. They are not backend work buffers.

        The minimization result is saved in the 'minimize_result' attribute of
        'config_name' as a formatted string.

        Multiple configurations with a unique config_name can be stored in the
        same .h5 file. HDF5 files are supported in python by the `h5py
        <https://www.h5py.org/>`_ package. With it, a saved configuration can be
        accessed as follows:

        .. code-block:: python

            import h5py

            config_name = "my_simulation"

            f = h5py.File(file_name + ".h5", "r")
            initial_state = np.array(f[config_name]["initial_state"]).view(np.complex128)
            final_state = np.array(f[config_name]["final_state"]).view(np.complex128)
            observables = np.array(f[config_name]["observables"]).view(np.float64)

            print(f["my_simulation"].attrs["minimize_result"])

        .. warning::

            The :literal:`"final_state"` and :literal:`"observables"`
            datasets are saved using Fortran subroutines which make
            use of parallel HDF5.

            The complex values of the final_state array are saved as a compound
            datatype consisting of contiguous double precision reals. This is
            equivalent to the np.complex128 NumPy datatype. To access this data
            without a loss of precision in python, the user must set the
            **view** of the NumPy array to np.complex128, rather than casting it
            to np.complex128 using the dtype keyword.

            Similarly, the observables array, which is saved as an array of
            double-precision reals, should have its view set to np.float64.
        """

        if self.subcomms.get_subcomm_index() != 0:
            return

        from quop_mpi._lib import parallel_io

        if self.subcomms.SUBCOMM.Get_rank() == 0:

            import h5py

            self.config_name = config_name

            file_name = ensure_path_and_extension(file_name, "h5")
            h5_file = h5py.File(file_name, action)

            # If the config_name already exists in the target file, add an underscore.
            duplicate = True
            while duplicate:
                if self.config_name in h5_file:
                    self.config_name += "_"
                else:
                    duplicate = False

            config = h5_file.create_group(self.config_name)

            if self.result is not None:
                config.attrs["minimize_result"] = str(self.result)

            h5_file.create_dataset(
                f"{self.config_name}/initial_phases",
                data=self.variational_parameters,
                dtype=np.float64,
            )
            h5_file.close()
        else:
            self.config_name = None

        file_name = self.subcomms.SUBCOMM.bcast(file_name, root=0)
        self.config_name = self.subcomms.SUBCOMM.bcast(self.config_name, root=0)

        parallel_io.io.save_dist_complex(
            file_name,
            f"{self.config_name}/",
            "final_state",
            "a",
            self.system_size,
            self.local_i_offset,
            self.context.state[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )

        parallel_io.io.save_dist_complex(
            file_name,
            f"{self.config_name}/",
            "initial_state",
            "a",
            self.system_size,
            self.local_i_offset,
            self.ansatz_initial_state[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )

        parallel_io.io.save_dist_real(
            file_name,
            f"{self.config_name}/",
            "observables",
            "a",
            self.system_size,
            self.local_i_offset,
            self.local_observables[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )
