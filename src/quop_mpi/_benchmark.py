# cspell:words ansatz subcomm subcomms
"""Benchmark mixin for systematic QVA depth studies."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable

import numpy as np
from mpi4py import MPI

from ._scope import scope
from ._utils._filenames import ensure_path_and_extension
from ._utils._tracker import JobTracker

if TYPE_CHECKING:
    from ._utils._comm_size import QuopMpiLayout


class Benchmark:
    """Mixin providing benchmarking functionality for :class:`~quop_mpi.ansatz`.

    This class is not intended to be instantiated directly. It provides the
    `benchmark` method for systematically studying QVA performance across
    different ansatz depths with support for job tracking, parameter persistence,
    and suspend/resume capabilities.
    """

    # Type hints for attributes provided by Ansatz
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm
        subcomms: QuopMpiLayout | None
        variational_parameters: np.ndarray | None
        ansatz_depth: int
        total_params: int
        seed: int
        benchmarking: bool
        repeat: int
        result: dict[str, Any] | None
        quop_result: dict[str, Any]
        tracker: JobTracker | None
        _has_param_map: bool
        _n_free_params: int | None

        def setup(self) -> None: ...
        def set_seed(self, seed: int) -> None: ...
        def set_depth(self, depth: int) -> None: ...
        def execute(self) -> None: ...
        def print_result(self) -> None: ...
        def save(self, filename: str, label: str, action: str) -> None: ...
        def _Ansatz__pre(self) -> None: ...  # noqa: N802
        def _Ansatz__gen_initial_params(self, depth: int = None) -> np.ndarray: ...  # noqa: N802

    def _init_benchmark(self) -> None:
        """Initialize benchmark-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.benchmarking: bool = False
        self.tracker: JobTracker | None = None
        self.repeat: int = 0

    @scope("world")
    def benchmark(
        self,
        ansatz_depths: Iterable[int],
        repeats: int,
        initial_parameters: list[float] | np.ndarray[float] | None = None,
        param_persist: bool = False,
        verbose: bool = True,
        filename: str | None = None,
        label: str = "test",
        save_action: str = "a",
        time_limit: float | None = None,
        suspend_path: str | None = None,
    ) -> None:
        """A method by which to study how a QVA performs as the number
        of ansatz iterations<ansatz depth> increases.

        Parameters
        ----------
        ansatz_depths : iterable[int]
            integers specifying a sequence of ansatz depths<ansatz depth>
        repeats : int
            number of repeats at each ansatz depth
        initial_parameters: list[float] or ndarray[float], optional
            ** Must be defined if a parameter mapping function is set. **
            initial variational parameter values, if not present these are generated
            using the default parameter generation methods of the ansatz unitaries.
        param_persist : bool, optional
            if True the optimised variational parameter values which achieved
            the lowest objective function value for all repeats at ansatz_depth
            will be used as starting parameters for the first
            ansatz_depth * total_params at ansatz_depth += 1. if a parameter
            map is set, the initial parameters will update whenever the
            objective function reaches a new minimum.
        verbose : bool, optional
            if True, print current the ansatz depth, repeat number and
            optimisation results (default True)
        filename : str or None, optional
            name of \\*.h5 file in which to save the optimised system state and observables
        label : str, optional
            if filename is not None, \\*.h5 data will be saved as
            "filename/label_depth_repeat" (default "test")
        save_action : {'a', 'w'}, optional
            action taken during first file write: 'a' to append, 'w' to overwrite (default 'a')
        time_limit : int or None, optional
            total allocated in-program time in seconds; if exceeded, the benchmark is suspended
        suspend_path : str or None, optional
            path to the suspend file if time_limit is not None
        """

        if self._has_param_map and initial_parameters is None:
            # Auto-generate random initial parameters for the free vector
            rng = np.random.default_rng(self.seed)
            initial_parameters = rng.uniform(0, 2 * np.pi, self._n_free_params)

        best_obj = np.inf
        previous_params = None

        if initial_parameters is not None:
            self.variational_parameters = np.asarray(initial_parameters, dtype=np.float64)

        self.setup()
        ansatz_depth_temp = deepcopy(self.ansatz_depth)
        self.benchmarking = True
        suspend_path = "suspend" if suspend_path is None else suspend_path
        self.tracker = JobTracker(
            repeats,
            list(ansatz_depths)[-1],
            time_limit,
            self.MPI_COMM_WORLD,
            seed=self.seed,
            suspend_path=suspend_path,
        )
        first = not self.tracker.got_match

        while not self.tracker.complete:
            repeat, depth = self.tracker.get_job()
            self.set_seed(self.tracker.get_seed())
            self.ansatz_depth = depth
            self.set_depth(depth)

            if repeat == 1 or first:
                self.set_depth(depth)
                first = False
                if (
                    self.subcomms.get_subcomm_index() == 0
                    and verbose
                    and self.subcomms.SUBCOMM.Get_rank() == 0
                ):
                    print(f"Starting depth = {depth}:", flush=True)

            self._Ansatz__pre()
            self.repeat = repeat

            if self.subcomms.get_subcomm_index() == 0:
                # Choose starting vector
                if self._has_param_map:
                    # With a parameter map, the free vector size is constant across all depths.
                    # The mapping function handles expansion via bound ansatz_depth/total_params.
                    # param_persist just means "use the best free vector found so far".
                    if param_persist and previous_params is not None:
                        self.variational_parameters = previous_params.copy()
                    else:
                        # Start fresh from the original supplied free-vector
                        self.variational_parameters = np.asarray(
                            initial_parameters, dtype=np.float64
                        )
                else:
                    # Unmapped case
                    if (not param_persist) or (depth == 1):
                        self.variational_parameters = self._Ansatz__gen_initial_params(depth)
                    else:
                        # Persist full-vector between repeats/depths
                        if self.subcomms.SUBCOMM.Get_rank() == 0:
                            n_previous = len(self.tracker.results_dict[depth - 1])
                        else:
                            n_previous = None
                        n_previous = self.subcomms.SUBCOMM.bcast(n_previous, root=0)

                        if n_previous > 0:
                            if self.subcomms.SUBCOMM.Get_rank() == 0:
                                if (
                                    self.tracker.job_list[self.tracker.job_index][1]
                                    != self.tracker.job_list[self.tracker.job_index - 1][1]
                                ) or (previous_params is None):
                                    funs = [
                                        result["fun"]
                                        for result in self.tracker.results_dict[depth - 1]
                                    ]
                                    xs = [
                                        result["variational_parameters"]
                                        for result in self.tracker.results_dict[depth - 1]
                                    ]
                                    previous_params = xs[np.argmin(funs)]
                            else:
                                previous_params = None

                            previous_params = self.subcomms.SUBCOMM.bcast(previous_params, root=0)

                            self.variational_parameters = np.empty(
                                depth * self.total_params, dtype=np.float64
                            )
                            # fill with best from last depth
                            self.variational_parameters[: len(previous_params)] = previous_params
                            # new parameters for the final layer
                            new_params = self._Ansatz__gen_initial_params(1)
                            self.variational_parameters[-self.total_params :] = new_params
                        else:
                            self.variational_parameters = self._Ansatz__gen_initial_params()

                if verbose and self.subcomms.SUBCOMM.Get_rank() == 0:
                    print(f"{repeat} of {repeats} at depth {depth}...", flush=True)

                self.execute()

                # If mapped, capture and persist the improved initial parameters on improvement
                if self._has_param_map:
                    if self.subcomms.SUBCOMM.Get_rank() == 0:
                        current_free = self.result["x"]
                        current_obj = self.quop_result["fun"]
                        if current_obj < best_obj:
                            best_obj = current_obj
                            previous_params = current_free.copy()
                    # Broadcast updated previous_params to all ranks for next iteration
                    previous_params = self.subcomms.SUBCOMM.bcast(previous_params, root=0)

                if verbose:
                    self.print_result()

                if filename is not None:
                    if first:
                        self.save(
                            ensure_path_and_extension(filename, "h5"),
                            f"{label}_{depth}_{repeat}",
                            action=save_action,
                        )
                    else:
                        self.save(
                            ensure_path_and_extension(filename, "h5"),
                            f"{label}_{depth}_{repeat}",
                            action="a",
                        )

                self.tracker.update(self.quop_result)
                first = False

            else:
                self.execute()
                self.tracker.update(None)

        self.benchmarking = False
        self.ansatz_depth = ansatz_depth_temp
