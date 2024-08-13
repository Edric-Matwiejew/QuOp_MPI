# cspell:words cartesian coeff
from __future__ import annotations
import numpy as np
from mpi4py import MPI
from ....__utils.__mpi import __scatter_1D_array
#from ....__lib import fCQAOA
from ....__lib import cartesian as cart

####################################
# imports and classes for type hints
####################################

from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm
iterable = Iterable

####################################

def serial(
    partition_table: list[int],
    MPI_COMM: Intracomm,
    variational_parameters: np.ndarray[np.float64],
    function: Callable,
    *args,
    **kwargs,
) -> Union[np.ndarray[np.float64], list[np.ndarray[np.float64]]]:
    """Generate the diagonal of the :term:`operator` for one or more sequential
    :term:`phase-shift unitaries<phase-shift unitary>` using a serial Python
    function.

    An :term:`Operator Function` for the
    :class:`quop_mpi.propagator.diagonal.unitary` class. The :literal:`function`
    argument must be defined in a corresponding :term:`FunctionDict` on
    initialisation of the :literal:`unitary` instance. Additional positional and
    keyword arguments contained in the FunctionDict are passed to :literal:`function`. 

    The :literal:`function` argument must conform to the signature, 

        .. code-block:: python

            def function(*args, *kwargs) -> (ndarray[float64] | list[ndarray[float64]])

    where the output is a 1-D real array of type :literal:`ndarray[float64]` 
    and length :term:`system size`, or :literal:`list` containing one or more 1-D 
    real arrays of type :literal:`ndarray[float64]` and length :literal:`system_size`.

    Parameters
    ----------
    partition_table : list[int]
        describes the parallel partitioning scheme, :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    variational_parameters : ndarray[float64]
        :term:`operator parameters <operator parameter>`, passed to :literal:`function` if
        :literal:`unitary.operator_n_params > 0`, :class:`quop_mpi.Unitary` attribute 
    function : Callable
        returns one or more :literal:`ndarray[float64]` of size :term:`system size`
        corresponding to the diagonal of the operator of one or more phase-shift
        unitaries

    Returns
    -------
    ndarray[float64] or list[ndarray[float64]]
        a 1-d real array or list of 1-D real arrays containing a :literal:`local_i`
        elements of the :term:`operator` diagonal with global index offset
        :literal:`local_i_offset`
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if MPI_COMM.Get_rank() == 0:

        operator = function(*variational_parameters, *args, **kwargs)

        operator_array = isinstance(operator[0], np.ndarray)

        n_terms = len(operator) if operator_array else 1
    else:
        operator = None
        n_terms = None

    n_terms = MPI_COMM.bcast(n_terms, 0)

    if n_terms <= 1:
        return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)

    terms = []

    for i in range(n_terms):
        if MPI_COMM.Get_rank() == 0:
            terms.append(
                __scatter_1D_array(operator[i], partition_table, MPI_COMM, np.float64)
            )
        else:
            terms.append(
                __scatter_1D_array(None, partition_table, MPI_COMM, np.float64)
            )

    return terms


def csv(
    partition_table: list[int], MPI_COMM: Intracomm, filename: Callable, *args, **kwargs
) -> np.ndarray[np.float64]:
    """Load the diagonal of a :term:`phase-shift unitary` using 
    `pandas <https://pandas.pydata.org/>`_.

    An :term:`Operator Function` for the
    :class:`quop_mpi.propagate.diagonal.unitary` class. The :literal:`filename` argument must
    be defined in a corresponding :term:`FunctionDict` on initialisation of the
    :literal:`unitary` instance. Additional keyword arguments in the :literal:`FunctionDict`
    are passed to the 
    `pandas.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
    method.

    Parameters
    ----------
    partition_table : list[int]
        describes the parallel partitioning scheme, :class:`quop_mpi.Ansatz`
        attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    filename : Callable
        path to a :literal:`*.csv` file

    Returns
    -------
    ndarray[float64]
        a 1-D real array containing :literal:`local_i` elements of the operator
        diagonal with global index offset :literal:`local_i_offset`
    """
    if MPI_COMM.Get_rank() == 0:
        import pandas as pd

        data_df = pd.read_csv(filename, *args, **kwargs)
        diagonals = data_df.to_numpy(dtype=np.complex128)
    else:
        diagonals = None

    return __scatter_1D_array(diagonals, partition_table, MPI_COMM, np.float64)


def hdf5(
    partition_table: int, MPI_COMM: Intracomm, filename: str, dataset_name: str
) -> np.ndarray[np.float64]:
    """Load the diagonal of a :term:`phase-shift unitary` using 
    `HDF5 for Python <https://docs.h5py.org/en/latest/index.html>`_.


    An :term:`Operator Function` for the
    :class:`quop_mpi.propagate.diagonal.unitary` class. The :literal:`filename` and
    :literal:`dataset_name` arguments must be defined in a corresponding
    :term:`FunctionDict` on initialisation of the :literal:`unitary` instance.
    Additional positional and keyword arguments in the :literal:`FunctionDict` are
    passed to the `h5py.File <https://docs.h5py.org/en/latest/high/file.html>`_
    method.

    Parameters
    ----------
    partition_table : int
        describes the parallel partitioning scheme, :class:`quop_mpi.Ansatz`
        attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    filename : str
        path to a :literal:`*.h5 file` 
    dataset_name : str
        path to the dataset containing a :literal:`ndarray[float64]` of size
        :term:`system size`

    Returns
    -------
    np.ndarray[np.float64]
        a 1-D real array containing :literal:`local_i` elements of the operator
        diagonal with global index offset :literal:`local_i_offset`
    """
    if MPI_COMM.Get_rank() == 0:
        import h5py as h5

        f = h5.File(filename, "r")
        operator = np.array(f[dataset_name]).view(np.float64)
        f.close()
    else:
        operator = None

    return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)


def array(
    partition_table: list[int], MPI_COMM: Intracomm, array: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """Define the diagonal of the :term:`phase-shift unitary` using a Numpy
    array.

    An :term:`Operator Function` for the
    :class:`quop_mpi.propagate.diagonal.unitary` class. 

    .. note::

        For memory efficiency, :literal:`array` can be present as an :literal:`ndarray[float64]` at 
        :literal:`MPI_COMM.rank == 0` only and :literal:`None` at all other ranks in :literal:`MPI_COMM`.  

    Parameters
    ----------
    partition_table : list[int]
        describes the parallel partitioning scheme, :class:`quop_mpi.Ansatz`
        attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    array : np.ndarray[np.float64]
        a 1-D real array of size :term:`system size`

    Returns
    -------
    ndarray[float64]
        a 1-D real array containing :literal:`local_i` elements of the operator
        diagonal with global index offset :literal:`local_i_offset`
    """
    return __scatter_1D_array(array, partition_table, MPI_COMM, np.float64)


def setup_cartesian(Ns: list[int], bounds: list[list[float]]) -> list[list[float]]:
    """Compute the step-size and minimum coordinate values in each dimension of
    a Cartesian grid.

    See Also
    --------

    cartesian
    
    cartesian_scaled

    Parameters
    ----------
    Ns : list[int]
        the number of qubits assigned to each dimension of the Cartesian grid
        such that there is :literal:`2 ** Ns[d]` grid points per dimension :literal:`d`
    bounds : list[list[float]]
        the lower and upper bounds of each dimension where 
        :literal:`len(Ns) == len(bounds)`

    Returns
    -------
    list[list[float]]
        the step-size, :literal:`deltas`, and the minimum, :literal:`mins`, in each Cartesian
        coordinate
    """
    d = len(Ns)
    Ls = np.array(
        [bound[1] - bound[0] for bound in bounds], dtype=np.float64
    )  # length in each dimension
    Ns = np.array([2**n for n in Ns], dtype=int)  # number of grid point per dimension
    deltas = Ls / (Ns - 1)  # grid step-size in each dimension
    mins = np.array(
        [bound[0] for bound in bounds], dtype=np.float64
    )  # minimum grid value in each dimension

    return [deltas, mins]


def cartesian(
    system_size: int,
    local_i: int,
    local_i_offset: int,
    Ns: list[int],
    deltas: list[float],
    mins: list[float],
    function: Callable,
    *args,
    **kwargs,
) -> np.ndarray[np.float64]:
    """TODO:UPDATE Generate the diagonal of a :term:`phase-shift unitary` :term:`operator`
    using a Python function defined in discrete Cartesian coordinates.

    An :term:`Observables Function`. Depending on wether :term:`QVA` simulation
    is defined using the :literal:`Ansatz` class directly or with a predefined
    :literal:`Ansatz` subclass from the :literal:`algorithm` submodule, the following
    arguments must be defined in a corresponding :term:`FunctionDict` on
    initialisation of the :literal:`unitary` instance:


        * :class:`quop_mpi.Ansatz`: :literal:`Ns`, :literal:`deltas`, :literal:`mins` and :literal:`function`

        * algorithms in :mod:`quop_mpi.algorithm.combinatorial`: :literal:`Ns`, :literal:`deltas`, :literal:`mins` and :literal:`function`

        * :class:`quop_mpi.algorithm.multivariable.qmoa`: :literal:`deltas`, :literal:`mins` and :literal:`function`

        * :class:`quop_mpi.algorithm.multivariable.qowe`: :literal:`function`

    Additional positional and keyword arguments in the :literal:`FunctionDict` are
    passed to :literal:`function`.

    The :literal:`function` argument must conform to the signature, 

        .. code-block:: python

            def function(x: ndarray[float64], *args, *kwargs) -> float

        where :literal:`x` is a 1-D array containing a :literal:`len(Ns)` -dimensional grid
        point.

    See Also
    --------
    setup_cartesian
        compute :literal:`deltas` and :literal:`mins`
    cartesian_scaled 
        alternative to :literal:`cartesian`, scales :literal:`function` between :literal:`0` and an
        upper bound.


    Parameters
    ----------
    system_size : int
        size of the simulated QVA
    local_i : int
        size of the local :term:`system state` partition,
        :class:`quop_mpi.Unitary` attribute
    local_i_offset : int
        global index offset of the local system state partition,
        :class:`quop_mpi.Unitary` attribute
    Ns : list[int]
        the number of qubits assigned to each dimension of the cartesian grid
        such that there is :literal:`2 ** Ns[d]` grid points per dimension :literal:`d`
    deltas : list[float]
        step size in each Cartesian coordinate
    mins : list[float]
        lower bound of each Cartesian coordinate 
    function : Callable
        a Python function that takes a list of :literal:`len(Ns)` real coordinate
        values and returns a :literal:`float`


    Returns
    -------
    ndarray[float64]
        a 1-D real array containing :literal:`local_i` elements of the operator
        diagonal with global index offset :literal:`local_i_offset`
    """

    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]

    x = cart.cartesian.gen_local_grid(
        system_size, Ns, strides, deltas, mins, local_i_offset, local_i
    )

    return np.array(function(x, *args, *kwargs), dtype = np.float64)


def cartesian_scaled(
    system_size: int,
    local_i: int,
    local_i_offset: int,
    MPI_COMM: Intracomm,
    Ns: list[int],
    deltas: list[float],
    mins: list[float],
    function: Callable,
    coeff: float,
    *args,
    **kwargs,
) -> np.ndarray[np.float64]:
    """Generate the diagonal of a :term:`phase-shift unitary` :term:`operator`
    using a Python function defined in discrete Cartesian coordinates with the
    function scaled between :literal:`0` and :literal:`coeff`.

    An :term:`Observables Function`. Depending on wether :term:`QVA` simulation
    is defined using the :literal:`Ansatz` class directly or with a predefined
    :literal:`Ansatz` subclass from the :literal:`algorithm` submodule, the following
    arguments must be defined in a corresponding :term:`FunctionDict` on
    initialisation of the :literal:`unitary` instance:


        * :class:`quop_mpi.Ansatz`: :literal:`Ns`, :literal:`deltas`, :literal:`mins`, :literal:`function` and :literal:`coeff`

        * algorithms in :mod:`quop_mpi.algorithm.combinatorial`: :literal:`Ns`, :literal:`deltas`, :literal:`mins`, :literal:`function` and :literal:`coeff`

        * :class:`quop_mpi.algorithm.multivariable.qmoa`: :literal:`deltas`, :literal:`mins`, :literal:`function` and :literal:`coeff`

        * :class:`quop_mpi.algorithm.multivariable.qowe`: :literal:`function` and :literal:`coeff`

    Additional positional and keyword arguments in the :literal:`FunctionDict` are
    passed to :literal:`function`.

    The :literal:`function` argument must conform to the signature, 

        .. code-block:: python

            def function(x: ndarray[float64], *args, *kwargs) -> float

        where :literal:`x` is a 1-D array containing a :literal:`len(Ns)` -dimensional grid
        point.

    See Also
    --------
    setup_cartesian
        compute :literal:`deltas` and :literal:`mins`
    cartesian_scaled 
        alternative to :literal:`cartesian_scaled`,  does not scale :literal:`function`

    Parameters
    ----------
    system_size : int
        size of the simulated QVA
    local_i : int
        size of the local :term:`system state` partition,
        :class:`quop_mpi.Unitary` attribute
    local_i_offset : int
        global index offset of the local system state partition,
        :class:`quop_mpi.Unitary` attribute
    Ns : list[int]
        the number of qubits assigned to each dimension of the cartesian grid
        such that there is :literal:`2 ** Ns[d]` grid points per dimension :literal:`d`
    deltas : list[float]
        step size in each Cartesian coordinate
    mins : list[float]
        lower bound of each Cartesian coordinate 
    function : Callable
        a Python function that takes a list of :literal:`len(Ns)` real coordinate
        values and returns a :literal:`float`
    coeff : float
        a positive real number, the upper bound of the scaling range

    Returns
    -------
    ndarray[float64]
        a 1-D real array containing :literal:`local_i` elements of the operator
        diagonal with global index offset :literal:`local_i_offset`
    """
    f = cartesian(
        system_size,
        local_i,
        local_i_offset,
        MPI_COMM,
        Ns,
        deltas,
        mins,
        function,
        *args,
        **kwargs,
    )

    f_max = MPI_COMM.allreduce(np.max(f), op=MPI.MAX)
    f_min = MPI_COMM.allreduce(np.min(f), op=MPI.MIN)

    f = coeff * (f - f_min) / (f_max - f_min)

    return f

def observables():
    return np.empty(1, dtype = np.float64)
