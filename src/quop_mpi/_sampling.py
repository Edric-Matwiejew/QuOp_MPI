# cspell:words subcomm
"""Sampling mixin for simulated measurement of QVA objective functions."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Callable

import numpy as np
from mpi4py import MPI

from ._scope import scope
from ._utils._interface import Interface

if TYPE_CHECKING:
    pass


class Sampling:
    """Mixin providing simulated sampling functionality for :class:`~quop_mpi.ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for computing the objective function using simulated quantum measurements
    rather than exact expectation values.
    """

    # Type hints for attributes provided by Ansatz
    if TYPE_CHECKING:
        subcomms: object
        observables: np.ndarray
        local_i: int
        local_i_offset: int
        context: object
        variational_parameters: np.ndarray
        quop_result: dict
        pre_execution_methods: list
        post_execution_methods: list
        unitaries: list
        MPI_COMM_WORLD: MPI.Intracomm

    def _init_sampling(self):
        """Initialize sampling-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.sampling_dict: dict = {}
        self.sample_indexes: list = []
        self.sample_minimum_indexes: list = []
        self.variational_parameter_history: list = []

        # sampling variables
        self.samples: list | None = None
        self.sample_block_size: int | None = None
        self.max_sample_iterations: int | None = None
        self.sampling_function: Callable | None = None
        self.sampling_function_input: Callable | None = None
        self.sampling: bool = False
        self.global_minimum: float | None = None
        self.minimum_sampled: float = np.inf
        self.shots_to_global_minimum: int | str = "not found"
        self.global_minimum_found: bool = False
        self.total_shots: int = 0

    @scope("world")
    def set_sampling(
        self,
        sample_block_size: int,
        function: Callable = None,
        max_sample_iterations: int = 100,
        sampling_dict: dict = None,
    ):
        """Compute the :term:`objective function` using simulated sampling.

        Samples are taken in blocks of `sample_block_size`. These are passed as
        a list of lists to :literal:`function` (a :term:`Sampling Function`),
        which returns a value for expectation value/objective function and a
        boolean that indicates whether the sampled result should be passed to
        the classical optimiser.

        If :literal:`function` is :literal:`None`, the :term:`objective function` is
        computed as the mean of :literal:`sample_block_size` shots.

        Parameters
        ----------
        sample_block_size : int
            number of shots taken between successive computation of the
            expectation value/objective function
        function : callable, optional
            :term:`Sampling Function`
        max_sample_iterations : int, optional
            maximum number of sample blocks per computation of the expectation
            value/objective function,  overrides the boolean returned by
            :literal:`function`, by default 100
        sampling_dict : FunctionDict, optional
            :term:`FunctionDict` for the Sampling Function
        """

        self.__parse_function_dict__(sampling_dict, "sampling_dict")

        if function is None:

            def function(samples):
                return np.mean(samples), True

        self.sample_block_size = sample_block_size
        self.max_sample_iterations = max_sample_iterations
        self.sampling_function_input = function

        if not self.setup_sampling:
            self.pre_execution_methods.append(self._pre_sampling)
            self.post_execution_methods.append(self._post_sampling)

        self.setup_sampling = True

    @scope("world")
    def unset_sampling(self):
        """Revert to simulation using exact computation of the
        :term:`objective function`.
        """
        self.setup_sampling = False
        self.sampling = False
        self.pre_execution_methods.remove(self._pre_sampling)
        self.post_execution_methods.remove(self._post_sampling)

    @scope("world")
    def _pre_sampling(self):
        """Preparation for simulated sampling."""

        self.minimum_sampled = np.inf
        self.total_shots = 0
        self.global_minimum_found = False
        self.shots_to_global_minimum = "not found"

        if self.MPI_COMM_WORLD.Get_rank() == 0:
            print("Executing with simulated sampling.")

    @scope("world")
    def _post_sampling(self):
        """Post simulation steps for simulated sampling."""

        if self.MPI_COMM_WORLD.Get_rank() == 0:

            self.quop_result["sampling total shots"] = self.total_shots
            self.quop_result["sampling minimum measured"] = self.minimum_sampled
            self.quop_result["sampling shots to minimum measured"] = self.shots_to_global_minimum
            self.quop_result["observables global minimum"] = self.global_minimum

    @scope("subcomm")
    def _gen_sampling(self):
        """Setup for simulated sampling."""

        self.sampling = True

        self._parse_sampling_function()

        self.global_minimum = self.subcomms.SUBCOMM.reduce(
            np.min(np.real(self.local_observables)), op=MPI.MIN
        )

    @scope("subcomm")
    def _parse_sampling_function(self):
        """Bind the arguments of a QuOp Sampling Function to the attributes of
        and :class:`~quop_mpi.ansatz` instance.
        """

        self.sampling_function = Interface(
            [self, self.unitaries],
            self.sampling_function_input,
            "sampling test function",
            self.subcomms.SUBCOMM,
            call_args=self.sampling_dict["args"],
            call_kwargs=self.sampling_dict["kwargs"],
        )

    @scope("subcomm")
    def _sample_expectation_value(self) -> float:
        """Returns the expectation value of QVA with solution quality values
        sampled according to the probability distribution of the system
        state vector.

        Returns
        -------
        float
            expectation value of the sampled solution qualities
        """

        if self.subcomms.SUBCOMM.Get_rank() == 0:
            self.samples = []
            self.sample_indexes = []
        else:
            self.samples = [None]
            self.sample_indexes = [None]

        for _ in range(self.max_sample_iterations):

            # Get the probability from each node in MPI_COMM
            self._get_local_probabilities()
            total_local_probability = np.array([self.local_probabilities.sum()], dtype=np.float32)

            comm_opt_size = self.subcomms.SUBCOMM.Get_size()

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                process_probabilities = np.empty(comm_opt_size, dtype=np.float32)
            else:
                process_probabilities = None

            self.subcomms.SUBCOMM.Gather(total_local_probability, process_probabilities, root=0)

            if self.subcomms.SUBCOMM.Get_rank() == 0:

                rank_samples = np.random.choice(
                    list(range(comm_opt_size)),
                    size=self.sample_block_size,
                    replace=True,
                    p=process_probabilities,
                )

                ranks, counts = np.unique(rank_samples, return_counts=True)

                samples_per_rank = np.zeros(comm_opt_size, dtype=int)

                for rank, count in zip(ranks, counts, strict=True):
                    samples_per_rank[rank] = count

            else:
                samples_per_rank = np.empty(comm_opt_size, dtype=int)

            self.subcomms.SUBCOMM.Bcast(samples_per_rank, root=0)

            local_normed_probabilities = self.local_probabilities / self.local_probabilities.sum()

            local_samples_inds = np.random.choice(
                list(range(self.local_i)),
                size=samples_per_rank[self.subcomms.SUBCOMM.Get_rank()],
                replace=True,
                p=local_normed_probabilities,
            ).astype(np.int32)

            local_samples = np.real(self.local_observables[local_samples_inds]).astype(np.float64)

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                self.samples.append(np.empty(self.sample_block_size, dtype=np.float64))
                self.sample_indexes.append(np.empty(self.sample_block_size, dtype=np.int32))

            self.subcomms.SUBCOMM.Gatherv(local_samples, [self.samples[-1], samples_per_rank], 0)

            self.subcomms.SUBCOMM.Gatherv(
                local_samples_inds + self.local_i_offset,
                [self.sample_indexes[-1], samples_per_rank],
                0,
            )

            if self.subcomms.SUBCOMM.Get_rank() == 0:

                self.sampling_function.update_parameters()

                sampling_function_result = self.sampling_function.call(
                    *self.sampling_dict["args"], **self.sampling_dict["kwargs"]
                )

                self.total_shots += len(self.samples[-1])

            else:
                sampling_function_result = None

            expectation, sample_test = self.subcomms.SUBCOMM.bcast(sampling_function_result, root=0)

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                argmin = np.argmin(self.samples[-1])
                self.sample_minimum_indexes.append(self.sample_indexes[-1][argmin])
                self.variational_parameter_history.append(self.variational_parameters)
                sample_min = self.samples[-1][argmin]
                if self.minimum_sampled > sample_min:
                    self.minimum_sampled = sample_min
                    self.shots_to_global_minimum = copy(self.total_shots)
            if sample_test:
                break

        return expectation
