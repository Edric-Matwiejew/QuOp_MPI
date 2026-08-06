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

        # Log file lives on the optimiser leader (subcomm 0, SUBCOMM rank 0)
        # so that _log_update can write to it directly without an extra
        # cross-comm transfer.  This rank is not necessarily world rank 0.
        if self.subcomms is not None and self.subcomms.is_optimiser_leader():

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

        # Only the optimiser leader holds the open logfile handle.
        if self.subcomms is None or not self.subcomms.is_optimiser_leader():
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

        if self.subcomms is not None and self.subcomms.is_optimiser_leader() and self.log:
            self.logfile.close()

    @scope("subcomm", returns="none")
    def save(
        self,
        file_name: str,
        config_name: str,
        action: str = "a",
        *,
        problem_name: str | None = None,
        save_initial_state: bool = True,
        save_observables: bool = True,
        save_final_state: bool = True,
    ) -> None:
        """Write ansatz states, observables and result metadata to HDF5.

        Parameters
        ----------
        file_name : str
            Path to the saved data.
        config_name : str
            Simulation identifier.
        action : {'a', 'w'}, optional
            ``'a'`` to append or ``'w'`` to overwrite, by default ``'a'``.
        problem_name : str, optional
            Problem-instance identifier. Group ``config_name`` together with
            other configurations assumed to share the same ``initial_state``
            and ``observables``.
        save_initial_state : bool, optional
            Ensure that the algorithm-level initial state is saved, by default
            ``True``.
        save_observables : bool, optional
            Ensure that the observables vector is saved, by default ``True``.
        save_final_state : bool, optional
            Save the final state, by default ``True``.

        Notes
        -----
        Without ``problem_name``, the HDF5 group layout is ::

            config_name/
                initial_parameters
                initial_state
                final_state
                observables

        With ``problem_name``, shared problem data and run-specific data are
        separated::

            problems/
                problem_name/
                    initial_state
                    observables
                    configs/
                        config_name/
                            initial_parameters
                            final_state

        When ``problem_name`` is supplied, ``save_initial_state=True`` and
        ``save_observables=True``, if a requested dataset is already present with the
        expected one-dimensional shape, it is not written again. If its shape
        is incompatible with ``system_size``, a :class:`ValueError` is raised.

        The saved ``initial_state`` and ``final_state`` datasets are the
        algorithm-level ansatz states: ``initial_state`` comes from
        :attr:`ansatz_initial_state`, and ``final_state`` comes from the current
        context state after evolution. They are not backend work buffers.

        The minimization result is stored as the ``minimize_result`` attribute
        of the configuration group.

        The distributed datasets are written using Fortran subroutines and
        parallel HDF5. Complex arrays are stored as compound double-precision
        values equivalent to ``np.complex128``. When reading them with NumPy,
        use ``.view(np.complex128)`` rather than casting with ``dtype``.
        Observables are stored as double-precision real values and can be read
        with ``.view(np.float64)``.
        """

        if self.subcomms.get_subcomm_index() != 0:
            return

        from quop_mpi._lib import parallel_io

        file_name = ensure_path_and_extension(file_name, "h5")
        config_path: str | None = None
        shared_path: str | None = None
        validation_error: str | None = None

        write_initial_state = save_initial_state
        write_observables = save_observables

        if self.subcomms.SUBCOMM.Get_rank() == 0:

            import h5py

            self.config_name = config_name

            with h5py.File(file_name, action) as h5_file:
                if problem_name is None:
                    # Configuration names are top-level.
                    config_parent = h5_file
                    while self.config_name in config_parent:
                        self.config_name += "_"

                    config = config_parent.create_group(self.config_name)
                    shared_group = config
                else:
                    # Problem names are top level.
                    # Create intermediate groups.
                    problems = h5_file.require_group("problems")
                    shared_group = problems.require_group(problem_name)
                    configs = shared_group.require_group("configs")

                    expected_shape = (self.system_size,)
                    validation_errors: list[str] = []

                    if save_initial_state and "initial_state" in shared_group:
                        existing = shared_group["initial_state"]
                        if not isinstance(existing, h5py.Dataset):
                            validation_errors.append(
                                f"Existing object "
                                f"{shared_group.name}/initial_state is not a dataset."
                            )
                        elif existing.shape != expected_shape:
                            validation_errors.append(
                                f"Existing initial_state for problem "
                                f"{problem_name!r} has shape {existing.shape}; "
                                f"expected {expected_shape}."
                            )
                        else:
                            write_initial_state = False

                    if save_observables and "observables" in shared_group:
                        existing = shared_group["observables"]
                        if not isinstance(existing, h5py.Dataset):
                            validation_errors.append(
                                f"Existing object "
                                f"{shared_group.name}/observables is not a dataset."
                            )
                        elif existing.shape != expected_shape:
                            validation_errors.append(
                                f"Existing observables for problem "
                                f"{problem_name!r} has shape {existing.shape}; "
                                f"expected {expected_shape}."
                            )
                        else:
                            write_observables = False

                    if validation_errors:
                        validation_error = " ".join(validation_errors)
                    else:
                        # Configuration names need only be unique within a problem
                        while self.config_name in configs:
                            self.config_name += "_"

                        config = configs.create_group(self.config_name)
                        config.attrs["problem_name"] = problem_name

                shared_path = shared_group.name.lstrip("/")

                if validation_error is None:
                    config_path = config.name.lstrip("/")

                    if self.result is not None:
                        config.attrs["minimize_result"] = str(self.result)

                    config.create_dataset(
                        "initial_parameters",
                        data=self.variational_parameters,
                        dtype=np.float64,
                    )

                h5_file.flush()

                try:
                    file_handle = h5_file.id.get_vfd_handle()
                except (AttributeError, OSError, ValueError):
                    file_handle = None

                if isinstance(file_handle, int):
                    os.fsync(file_handle)
        else:
            self.config_name = None

        self.subcomms.SUBCOMM.barrier()
        file_name = self.subcomms.SUBCOMM.bcast(file_name, root=0)
        self.config_name = self.subcomms.SUBCOMM.bcast(self.config_name, root=0)
        config_path = self.subcomms.SUBCOMM.bcast(config_path, root=0)
        shared_path = self.subcomms.SUBCOMM.bcast(shared_path, root=0)
        write_initial_state = self.subcomms.SUBCOMM.bcast(
            write_initial_state,
            root=0,
        )
        write_observables = self.subcomms.SUBCOMM.bcast(
            write_observables,
            root=0,
        )
        validation_error = self.subcomms.SUBCOMM.bcast(
            validation_error,
            root=0,
        )

        if validation_error is not None:
            raise ValueError(validation_error)

        if save_final_state:
            parallel_io.io.save_dist_complex(
                file_name,
                f"{config_path}/",
                "final_state",
                "a",
                self.system_size,
                self.local_i_offset,
                self.context.state[: self.local_i],
                self.subcomms.SUBCOMM.py2f(),
            )

        if write_initial_state:
            parallel_io.io.save_dist_complex(
                file_name,
                f"{shared_path}/",
                "initial_state",
                "a",
                self.system_size,
                self.local_i_offset,
                self.ansatz_initial_state[: self.local_i],
                self.subcomms.SUBCOMM.py2f(),
            )

        if write_observables:
            parallel_io.io.save_dist_real(
                file_name,
                f"{shared_path}/",
                "observables",
                "a",
                self.system_size,
                self.local_i_offset,
                self.local_observables[: self.local_i],
                self.subcomms.SUBCOMM.py2f(),
            )
