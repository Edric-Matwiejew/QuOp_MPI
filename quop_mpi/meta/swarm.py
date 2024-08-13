from __future__ import annotations
import inspect
from time import time
import numpy as np
from mpi4py import MPI
from quop_mpi import Ansatz
from functools import wraps
from mpi4py import MPI
from logging import warn
from ..__utils.__filenames import ensure_path_and_extension
from ..__utils.__mpi import subcomms, MPI_COMM_type
from ..__utils.__tracker import swarm_tracker

#### Type hints ######################
from ..Unitary import Unitary
from ..Ansatz import Ansatz as _Ansatz
from typing import Callable, Union, Iterable
Ansatz = type(_Ansatz)
######################################

def is_list_of_lists(args):
    if len(args) > 0:
        return (
            all(isinstance(args[0][i], list) for i in range(len(args[0])))
            if isinstance(args[0], list)
            else False
        )
    else:
        return False


def is_len_swarm(self, args):
    return len(args[0]) == self.subcomms.get_n_subcomms()


def parse_args_list_of_lists(self, args, kwargs):
    a = None
    k = None
    if not is_len_swarm(self, args):
        raise RuntimeError(
            (
                f"Input list of argument lists of len={len(args[0])} "
                f"does not equal the swarm size of {self.subcomms.get_n_subcomms()}"
            )
        )
    last_arg = args[0][self.subcomms.get_subcomm_index()][-1]
    k = last_arg if isinstance(last_arg, dict) else {}
    a = (
        args[0][self.subcomms.get_subcomm_index()][:-1]
        if len(k) != 0
        else args[0][self.subcomms.get_subcomm_index()]
    )
    if (len(kwargs) >= 0) and (len(k) == 0):
        k = kwargs
    else:
        warn(
            (
                f"Arguments list equal to swarm size "
                f"{self.subcomms.get_n_subcomms()} identified with dictionary "
                f"in last position of sublists, taking keyword arguments from list."
            )
        )
    return a, k


def iterate_subcomms_1(method):
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if self.subcomms.in_subcomm():
            if is_list_of_lists(args):
                a, k = parse_args_list_of_lists(self, args, kwargs)
            else:
                a = args
                k = kwargs
            return method(self, *a, **k)

    return wrapped_method


def iterate_subcomms_2(name, self):
    def wrapped_method(*args, **kwargs):
        if self.subcomms.in_subcomm():
            if is_list_of_lists(args):
                a, k = parse_args_list_of_lists(self, args, kwargs)
            else:
                a = args
                k = kwargs
            return getattr(self.ansatz, name)(*a, **k)

    return wrapped_method


class swarm:
    """Create and operate on a swarm of identical :literal:`Ansatz` instances.

    Each :literal:`Ansatz` instance is associated with an MPI subcommunicator
    such that they can carry out :term:`QVA` simulation independently.

    The :literal:`Ansatz` instance is initialised by the :literal:`swarm`
    instance as,

    .. code-block:: python
        
        Ansatz(*args, MPI_COMM, **kwargs)


    Parameters
    ----------
    nodes_per_subcomm : int
        number of compute nodes associated with each :literal:`Ansatz`
        subcommunicator, if :literal:`nodes_per_subcomm == 1` create
        :literal:`maxcomm` subcommunicators per available compute node.
    processes_per_node : int
        number of MPI processes per compute node, must be the same for all nodes
    maxcomm : int
        target number of :literal:`Ansatz` subcommunicators
    MPI_COMM : Intracomm
        MPI communicator from which to create the :literal:`Ansatz`
        subcommunicators
    alg : Ansatz
        an :literal:`Ansatz`, :class:`quop_mpi.Ansatz` or a predefined algorithm
        (see :mod:`quop_mpi.algorithm`)

    """
    def __init__(
        self,
        nodes_per_subcomm: int,
        processes_per_node: int,
        maxcomm: int,
        MPI_COMM: MPI_COMM_type,
        alg: 'Ansatz',
        *args,
        **kwargs,
    ):
        self.filename = None
        self.label = None
        self.time = None
        self.last_execution_time = None
        self.suspend_path = None
        self.results = None
        self.MPI_COMM = MPI_COMM

        self.subcomms = subcomms(
            nodes_per_subcomm, processes_per_node, maxcomm, MPI_COMM
        )
        self.__set_ansatz(alg, *args, **kwargs)

    @iterate_subcomms_1
    def __set_ansatz(self, alg, *args, **kwargs):

        self.alg = alg
        self.ansatz = self.alg(*args, self.subcomms.SUBCOMM, **kwargs)

        swarm_methods = inspect.getmembers(swarm, predicate=inspect.isfunction)
        swarm_method_names = [method[0] for method in swarm_methods]
        for alg_method in inspect.getmembers(self.alg, predicate=inspect.isfunction):
            if alg_method[0] not in swarm_method_names:
                setattr(self, alg_method[0], iterate_subcomms_2(alg_method[0], self))

    def set_unitaries(self, unitaries: Union[list[Unitary], list[list[Unitary]]]):
        """Set the unitaries of an :literal:`Ansatz` swarm.

        Parameters
        ----------
        unitaries :list[Unitary] or list[list[Unitary]]
            a list of :literal:`unitary` instances (broadcast to all swarm instances)
            or a list of :literal:`unitary` instances for each :literal:`Ansatz` instance in
            the :literal:`swarm`

        Raises
        ------
        RuntimeError
            if :literal:`unitaries`is a list of lists that is not equal to the
            :literal:`swarm` size
        """
        if self.subcomms.in_subcomm():
            if not is_list_of_lists(unitaries):
                self.ansatz.set_unitaries(unitaries)
            elif is_len_swarm(self, unitaries):
                self.ansatz.set_unitaries(unitaries[0][self.subcomms.get_subcomm_index()])
            else:
                raise RuntimeError(
                    (
                        f"Input list of unitary lists of len={len(unitaries[0])} does "
                        f"not equal the swam size of {self.subcomms.get_n_subcomms()}."
                    )
                )

    def set_log(self, *args, **kwargs):
        """Log simulation information.

        See Also
        --------
        :meth:`~quop_mpi.Ansatz.set_log`

        Parameters
        ----------
        args: list[Any] or list[list[Any]] 
            positional arguments for :meth:`quop_mpi.ansatz.set_log`, or a list
            of positional arguments specifying unique input for
            :meth:`quop_mpi.ansatz.set_log` for each :literal:`Ansatz` instance.
        kwargs: dict
            keyword arguments for :meth:`quop_mpi.ansatz.set_log`, or keywords
            pointing to a list of positional arguments specifying unique input
            for :meth:`quop_mpi.ansatz.set_log` for each :literal:`Ansatz`
            instance.
        """
        if self.subcomms.in_subcomm():

            if is_list_of_lists(args):
                a, k = parse_args_list_of_lists(self, args, kwargs)
            else:
                a = [
                    ensure_path_and_extension(
                        args[0], "csv", modifier=self.subcomms.get_subcomm_index()
                    ),
                    f"{args[1]}_{self.subcomms.get_subcomm_index()}",
                ]
                k = kwargs

            self.ansatz.set_log(*a, **k)

    def save(self, *args, **kwargs):
        """Save simulation results.

        See Also
        --------
        :meth:`~quop_mpi.Ansatz.save`

        Parameters
        ----------
        args: 
            positional arguments for :meth:`quop_mpi.ansatz.save`, or a list of
            positional arguments specifying unique input for
            :meth:`quop_mpi.ansatz.save` for each :literal:`Ansatz` instance.
        kwargs:
            keyword arguments for :meth:`quop_mpi.ansatz.save`, or keywords
            pointing to a list of positional arguments specifying unique input
            for :meth:`quop_mpi.ansatz.save` for each :literal:`Ansatz`
            instance.
        """
        if self.subcomms.in_subcomm():

            if is_list_of_lists(args):
                a, k = parse_args_list_of_lists(self, args, kwargs)
            else:
                a = [
                    ensure_path_and_extension(
                        args[0], "h5", modifier=self.subcomms.get_subcomm_index()
                    ),
                    args[1],
                ]
                k = kwargs

            self.ansatz.save(*a, **k)

    def benchmark(self, *args, **kwargs):
        """Test :term:`QVA` performance as a function of :term:`Ansatz Depth`.

        See Also
        --------
        :meth:`~quop_mpi.Ansatz.benchmark`

        Parameters
        ----------
        args: list[Ans] or list[list[Any]]
            positional arguments for :meth:`quop_mpi.ansatz.benchmark`, or a
            list of positional arguments specifying unique input for
            :meth:`quop_mpi.ansatz.benchmark` for each :literal:`Ansatz`
            instance.
        kwargs: dict
            keyword arguments for :meth:`quop_mpi.ansatz.benchmark`, or keywords
            pointing to a list of positional arguments specifying unique input
            for :meth:`quop_mpi.ansatz.benchmark` for each :literal:`Ansatz`
            instance.
        """
        if self.subcomms.in_subcomm():

            if is_list_of_lists(args):
                a, k = parse_args_list_of_lists(self, args, kwargs)
            else:
                a = args
                k = kwargs
                if "filename" in k.keys():
                    k["filename"] = ensure_path_and_extension(
                        k["filename"], "h5", modifier=self.subcomms.get_subcomm_index()
                    )
                if "suspend_path" in k.keys():
                    k[
                        "suspend_path"
                    ] = f"{k['suspend_path']}_{self.subcomms.get_subcomm_index()}"
                if "label" in k.keys():
                    k["label"] = f"{k['label']}_{self.subcomms.get_subcomm_index()}"
                else:
                    k["label"] = f"test_{self.subcomms.get_subcomm_index()}"

            self.ansatz.benchmark(*a, **k)

    def get_optimal_result(self) -> dict:
        """Retrieve the result with the lowest :term:`objective function` value
        out of the last set of simulations executed by the :literal:`swarm`.

        Returns
        -------
        dict
            simulation result
        """
        if self.subcomms.in_rootcomm():

            min_fun_rank = self.subcomms.ROOTCOMM.allreduce(
                (self.ansatz.quop_result["fun"], self.subcomms.ROOTCOMM.Get_rank()),
                op=MPI.MINLOC,
            )[1]

            result = self.subcomms.ROOTCOMM.bcast(
                self.ansatz.quop_result, root=min_fun_rank
            )
            result["swarm_index"] = min_fun_rank

        else:
            result = None

        return self.MPI_COMM.bcast(result, root=0)

    def execute_swarm(
        self,
        param_lists: list[np.ndarray[np.float64]],
        basename: str,
        log_path: str = None,
        h5_path: str = None,
        labels: Union[str, list[str]] = None,
        save_action: str ="a",
        time_limit: float = None,
        verbose: bool =True,
        suspend_path: str = None,
    ):
        """Parallel simulation of :term:`QVAs <QVA>` over a set of initial
        :term:`variational parameters`.

        Parameters
        ----------
        param_lists : list[np.ndarray[np.float64]]
            list of 1-D real arrays containing initial :term:`variational
            parameters`
        basename : str
            folder in which to store simulation results and suspend data unless
            otherwise specified, by default :literal:`None`
        log_path : str, optional
            folder in which to write simulation log files, by default
            :literal:`None`
        h5_path : str, optional
            folder in which to write simulation results, by default
            :literal:`None`
        labels : str or list[str], optional
            labels(s) for each simulation, by default :literal:`None`
        save_action : {"a", "w"}
            "a" to append "w" to (over)write, by default :literal:`"a"`
        time_limit : float, optional
            suspend if the time remaining is less than the time taken by the
            last :term:`QVA` simulation, by default :literal:`None`
        verbose : bool, optional
            print the simulation results and simulation progress, by default
            :literal:`True`
        suspend_path : str, optional
            folder in which to store suspend data, by default :literal:`None`

        Returns
        -------
        dict
            a dictionary of optimisation results with keys
            :literal:`str(params_list[i])`
        """
        logging = True

        labels = ["test" for _ in range(len(param_lists))] if labels is None else labels

        tasks = [(labels[i], param_lists[i]) for i in range(len(param_lists))]

        suspend_path = basename if suspend_path is None else suspend_path

        tracker = swarm_tracker(tasks, time_limit, self.subcomms, suspend_path=suspend_path)

        if basename is not None and h5_path is None:
            h5path = ensure_path_and_extension(
                f"{tracker.suspend_path[:-5]}_{self.subcomms.get_subcomm_index()}",
                "h5",
            )
        elif h5_path is not None:
            h5path = ensure_path_and_extension(h5_path, 'h5')


        if basename is not None and log_path is None:
            logpath = ensure_path_and_extension(
                f"{tracker.suspend_path[:-5]}_{self.subcomms.get_subcomm_index()}",
                "csv",
            )
        elif log_path is not None:
            logpath = ensure_path_and_extension(log_path, 'csv')

        first = True

        while not tracker.complete:

            if first:
                action = save_action
                first = False
            else:
                action = "a"

            task = tracker.get_task()

            seed = tracker.get_seed()

            if task is not None:

                label, params = task

                if logging and basename is not None:
                    self.set_log(logpath, label, action)

                self.execute(params)
                result = self.ansatz.quop_result

                if basename is not None:
                    self.save(h5path, label, action)

                if verbose and self.subcomms.SUBCOMM.Get_rank() == 0:
                    print(f"Completed task: {label}", flush=True)

            else:
                result = None

            tracker.update(result)

        return self.subcomms.MPI_COMM.bcast(tracker.results_dict, root=0)

    def benchmark_swarm(
        self,
        ansatz_depths: Iterable[int],
        repeats: int,
        basename: str,
        param_persist: bool = True,
        verbose: bool = True,
        save_action: str = "a",
        time_limit: float = None,
        logging: bool = True,
        suspend_path: str = None,
    ):
        """Test :term:`QVA` performance with increasing :term:`ansatz depth`
        with repeats at each depth computed in parallel over the
        :literal:`swarm`.

        Parameters
        ----------
        ansatz_depths : Iterable[int]
            simulated :term:`ansatz depths <ansatz depth>`
        repeats : int
            number of repeats at each ansatz depth
        basename : str
            path to directory in which to write simulation results and logs, by
            default :literal:`None`
        param_persist : bool, optional
            if :literal:`True` the :term:`optimised<optimiser>`
            :term:`variational parameter <variational parameters>` values which
            achieved the lowest :term:`objective function` value  for all
            repeats at :literal:`ansatz_depth` will be used as starting
            parameters for the first :literal:`ansatz_depth * total_params` at
            :literal:`ansatz_depth += 1`, by default :literal:`True`
        verbose : bool, optional
            if :literal:`True`, print current the :term:`ansatz depth`, repeat
            number and :term:`optimisation<optimiser>` results, by default
            :literal:`True`
        save_action : {'a', 'w'}
            'a' to append, 'w' to (over)write, by default "a"
        time_limit : float, optional
            time limit in seconds, program will suspend if the time remaining is
            less than the time taken by the last simulation, by default
            :literal:`None`
        logging : bool, optional
            write simulation results to log files, by default :literal:`True`
        suspend_path : str, optional
            path to directory in which to write suspend data, by default
            :literal:`None`

        Returns
        -------
        list[dict] 
            optimisation results ordered by ansatz depth
        """
        self.set_seed(self.subcomms.get_subcomm_index())

        first = True

        results = []

        for depth in ansatz_depths:

            if basename is not None:
                depth_basename = f"{basename}_depth_{depth}"

            if suspend_path is not None:
                depth_suspend_path = f"{suspend_path}/depth_{depth}"
            else:
                depth_suspend_path = f"{basename}/depth_{depth}"


            self.ansatz.set_depth(depth)
            self.ansatz.destroy()
            self.ansatz.setup()

            labels = [f"repeat_{repeat}_depth_{depth}" for repeat in range(repeats)]

            if self.subcomms.get_subcomm_index() == 0:

                if (first or depth == 1) or (not param_persist):
                    params_list = [self.ansatz.gen_initial_params(depth) for _ in range(repeats)]
                    first = False
                else:
                    params_list = [
                        np.concatenate(
                            [optimal_x, self.ansatz.gen_initial_params(1)]
                        )
                        for _ in range(repeats)
                    ]

            else:

                params_list = None

            params_list = self.subcomms.MPI_COMM.bcast(params_list, root=0)

            start_time = time()

            if verbose and self.subcomms.MPI_COMM.Get_rank() == 0:
                print(f"Starting {repeats} repeats at depth {depth}...", flush=True)

            results.append(
                self.execute_swarm(
                    params_list,
                    basename=depth_basename,
                    labels=labels,
                    save_action=save_action,
                    time_limit=time_limit,
                    log_path = f'{basename}',
                    h5_path = f'{basename}',
                    verbose=verbose,
                    suspend_path = depth_suspend_path,
                )
            )

            if time_limit is not None:
                time_limit -= time() - start_time

            if self.subcomms.MPI_COMM.Get_rank() == 0:
                funs = [results[-1][key]["fun"] for key in results[-1].keys()]
                xs = [
                    results[-1][key]["variational parameters"]
                    for key in results[-1].keys()
                ]
                optimal_x = xs[np.argmin(funs)]
            else:
                optimal_x = None

            optimal_x = self.subcomms.MPI_COMM.bcast(optimal_x, root=0)

            if verbose and results:

                best_result = {"fun": np.inf}
                for key in results[-1].keys():
                    best_result = (
                        results[-1][key]
                        if results[-1][key]["fun"] < best_result["fun"]
                        else best_result
                    )
                self.ansatz.quop_result = best_result

                if self.subcomms.MPI_COMM.Get_rank() == 0:
                    print(f"Optimial result at depth {depth}:")
                    self.print_result()

        return results
