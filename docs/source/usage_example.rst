Usage Example
=============

As MPI based programs are parallelized through running multiple instances of the same program, they do not support the use of interactive environments such as the Jupyter notebook. The following code is to be placed in 'example.py', which is then run by issuing the terminal command:

.. code-block:: bash

   mpirun -N n python example.py

where n is a user specified parameter equal to the number of MPI nodes. Another possible workflow is to save the simulation results to disc and carry out visualization and analysis interactively, without calls to :mod:`~qwao_mpi.MPI`.

QWAO Simulation
###############

Import required modules.

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    import qwao_mpi as qw

Next, set up the MPI environment by creating an MPI communicator object.

.. code-block:: python

    comm = MPI.COMM_WORLD

Define QWAO parameters, the depth of the circuit, p, the number of qubits, n_qubits, and an array of starting angles of size 2p, x0. x0 defines the starting angles of the :math:`\vec{\gamma}` and :math:`\vec{t}` arrays.

.. code-block:: python

    p = 2
    n_qubits = 3
    np.random.seed(1)
    x0 = np.random.rand(2*p)

Create a :class:`~qwao_mpi.MPI.qwao` object. This contains the methods needed to perform parallel simulation of the QWAO algorithm.
qwao = qw.MPI.qwao(n_qubits, comm)

Define the circulant graph by passing a :mod:`~qwao_mpi.graph_array` to :meth:`~qwao_mpi.MPI.qwao.graph`.

.. code-block:: python

    qwao.graph(qw.graph_array.complete(qwao.size))

Define the solution qualities by passing a :mod:`~qwao_mpi.quality` method to :meth:`~qwao_mpi.MPI.qualities`

.. code-block:: python

    qwao.qualities(qw.qualities.ordered_integers)

The QWAO algorithm may then be executed. Note that :meth:`~qwao_mpi.MPI.plan` and :meth:`~qwao_mpi.MPI.destory_plan()` are necessary to create and free ancillary arrays and pointers used by external libraries.

.. code-block:: python

    qwao.plan()
    qwao.execute(x0)
    qwao.destroy_plan()

Simulation results can then be saved to disc as a HDF5 file using :meth:`~qwao_mpi.MPI.save`. This file type can be accessed in python using the `h5py <https://www.h5py.org/>` module.

.. code-block:: python

    qwao.save("example", "example_config", action = "w")

The results of the optimzation process can also be examined as follows:

.. code-block:: python

    if comm.Get_rank() == 0:
    print(qwao.result)
