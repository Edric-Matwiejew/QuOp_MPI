Usage Examples
==============

As MPI based programs are parallelized through running multiple instances of the same program, they do not support the use of interactive environments such as the Jupyter notebook. The following code is to be placed in 'example.py', which is then run by issuing the terminal command:

.. code-block:: bash

   mpirun -N n python example.py

where n is a user specified parameter equal to the number of MPI nodes. Another possible workflow is to save the simulation results to disc and carry out visualization and analysis interactively, without calls to :mod:`~quop_mpi.MPI`.


QWAO Simulation
###############

Import required modules.

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    import quop_mpi as qu

Next, set up the MPI environment by creating an MPI communicator object.

.. code-block:: python

    comm = MPI.COMM_WORLD

Define QWAO parameters, the depth of the circuit, p, the number of qubits, n_qubits, and an array of starting angles of size 2p, x0, which defines the starting angles of the :math:`\vec{\gamma}` and :math:`\vec{t}` arrays.

.. code-block:: python

    p = 2
    n_qubits = 3
    np.random.seed(1)
    x0 = np.random.rand(2*p)

Create a :class:`~quop_mpi.MPI.qwao` object. This contains the methods needed to perform parallel simulation of the QWAO algorithm.

.. code-block:: python

    quop = qu.MPI.qwao(n_qubits, comm)

Define the circulant graph by passing a :mod:`~quop_mpi.graph_array` to :meth:`~quop_mpi.MPI.quop.graph`.

.. code-block:: python

    quop.graph(qu.graph_array.complete(quop.size))

Define the solution qualities by passing a :mod:`~quop_mpi.qualities` method to :meth:`~quop_mpi.MPI.quop.set_qualities`

.. code-block:: python

    quop.qualities(qu.set_qualities.ordered_integers)

The QWAO algorithm may then be executed. Note that :meth:`~quop_mpi.MPI.quop.plan` and :meth:`~quop_mpi.MPI.quop.destroy_plan` are necessary to create and free ancillary arrays and pointers used by external libraries.

.. code-block:: python

    quop.plan()
    quop.execute(x0)
    quop.destroy_plan()

Simulation results can then be saved to disc as a HDF5 file using :meth:`~quop_mpi.MPI.quop.save`. This file type can be accessed in python using the `h5py <https://www.h5py.org/>`_ module.

.. code-block:: python

    quop.save("example", "example_config", action = "w")

The results of the optimzation process can also be examined as follows:

.. code-block:: python

    if comm.Get_rank() == 0:
    print(quop.result)
