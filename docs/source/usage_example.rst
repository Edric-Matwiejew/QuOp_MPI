Usage Examples
==============

This section aims to give an overview of the usage and functionalities of QuOp_MPI. For more detail, please see the documentation the method-specific documentation in the `Modules` section.

MPI
---

As MPI based programs are parallelized through running multiple instances of the same program, they do not support the use of interactive environments such as the Jupyter notebook. The following code is to be placed in 'example.py', which is then run by issuing the terminal command:

.. code-block:: bash

   mpirun -N n python example.py

where n is a user specified parameter equal to the number of MPI *nodes* (or *processes*). Another possible workflow is to save the simulation results to disc and carry out visualization and analysis interactively, without calls to :mod:`~quop_mpi.MPI`.


QWAO Simulation
---------------------

Import required modules.

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    import quop_mpi as qu

Next, set up the MPI environment by creating an MPI communicator object.

.. code-block:: python

    comm = MPI.COMM_WORLD

Define QWAO parameters, the depth of the circuit, p, the number of qubits, n_qubits, and a function to generate arrays of size 2p, x0, which defines the starting values of the :math:`\vec{\gamma}` and :math:`\vec{t}` parameters.

.. code-block:: python

    p = 2
    n_qubits = 3

    rng.np.random.RandomState(1)

    def x0(p):
        return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)  

Create a :class:`~quop_mpi.MPI.qwao` object. This contains the methods needed to perform parallel simulation of the QWAO algorithm.

.. code-block:: python

    qwoa = qu.MPI.qwao(n_qubits, comm)

By default this will use the adjacency maxtrix of a complex graph as :math:`W` in the mixing unitary :math:`U_W`.

Define the solution qualities, :math:`\vec{q}`, by passing a :mod:`~quop_mpi.qualities` method to :meth:`~quop_mpi.MPI.quop.set_qualities`

.. code-block:: python

    quop.set_qualities(qu.qualities.random_floats)

The QWAO algorithm may then be executed. Note that :meth:`~quop_mpi.MPI.quop.plan` and :meth:`~quop_mpi.MPI.quop.destroy_plan` are necessary to create and free ancillary arrays and pointers used by FFTW.

.. code-block:: python

    quop.plan()
    quop.execute(x0)
    quop.destroy_plan()

Simulation results can then be saved to disc as a HDF5 file using :meth:`~quop_mpi.MPI.quop.save`. This file type can be accessed in python using the `h5py <https://www.h5py.org/>`_ module.

.. code-block:: python

    quop.save("qwoa", "example_config", action = "w")

The results of the optimization process can also be examined as follows:

.. code-block:: python

    quop.print_result()

QAOA Simulation + Real-Time Data Logging
----------------------------------------

QAOA simulation begins much the same as QWOA:

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    import quop_mpi as qu
    import networkx as nx

    comm = MPI.COMM_WORLD

    p = 2
    n_qubits = 3

    rng.np.random.RandomState(1)

    def x0(p):
        return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)  

    qaoa = qu.MPI.qaoa(n_qubits, comm)    

By defualt the :class:`~quop_mpi.MPI.qaoa` class uses a :math:`2^n` dimensional hypercube. See :math:`~quop_mpi.MPI.qaoa.set_graph` for how to define a custom mixing operator.

To set up real-time logging of QAOA or QWOA results a log file must be defined:

.. code-block:: python

    qaoa.log_results("log", "qaoa", action = "a")

When the QAOA is executed the :meth:`n, p, \tilde{q}_\text{cutoff}`, the final value of :math:`f(\vec{\gamma},\vec{t}), \langle \vec{\gamma}, \vec{t} | \vec{\gamma}, \vec{t} \rangle` and, the in-program simulation time will be saved to log.csv with the identifier "qaoa". The same log file can be used for multiple simulations.

Simulation then proceeds as with the QWOA, excluding class to :meth:`~quop_mpi.MPI.qwoa.plan` and :meth:`~quop_mpi.MPI.qwoa.destory`, as the :class:`~quop_mpi.MPI.qaoa` class does not use FFTW libraries.

.. code-block:: python

    qaoa.set_initial_state(name = "equal")
    qaoa.set_qualities(qu.qualities.random_floats)
    qaoa.execute(x0(p))
    qaoa.save("qaoa", "example_config", action = "w")
    qaoa.print_result()

Automated Benchmarking
----------------------

It is often the case that one wishes to see how a given system responds as a function of :math:`p`. To assist with this QuOp_MPI provides the :meth:`~quop_mpi.MPI.system.benchmark` method. Note that :meth:`~quop_mpi.MPI.system.log_results` can be used to log the results for each value of :meth:`p` and repetition.

.. note:: 
    `param_func`,`qual_func and `state_func` each require an integer `seed` keyword argument.

.. code-block:: python

    import quop_mpi as qu
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    qubits_min = 2
    qubits_max = 4
    ps = list(range(1,6))
    repeats = 5

    def x0(p,seed):
        return np.random.uniform(low = 0, high = , size = 2*p)

    for n_qubits in range(qubits_min, qubits_max):
        qaoa = qu.MPI.qaoa(n_qubits,comm)
        qaoa.set_initial_state(name = "equal")
        qaoa.log_results("benchmark_example","qaoa_equal",action="a")
        qaoa.set_qualities(qu.qualities.random_floats)
        qaoa.benchmark(
                ps,
                repeats,
                param_func = x0,
                qual_func = qu.qualities.random_floats,
                filename = "qaoa_complete_equal",
                label = "qaoa_" + str(qubits))

User Defined Quality Function
-----------------------------

QuOp_MPI supports user defined quality functions, as detailed in :mod:`~quop_mpi.MPI.system.set_qualities`.

Working With HDF5 Files
-----------------------

HDF5 is a highly portable data format widely used in scientific computing. For comprehensive information on working with this format see the HDF5 documentation, or most applicably, the documentation for its python interface h5py.

The following is an example covering how to access simulation data saved via :meth:`~quop_mpi.MPI.system.save`. The final distribution of the first QAOA example will be imported as a numpy array and its probability distribution visualized using Matplotlib. This may be carried out in an interactive python environment.

First, import the required modules:

.. code-block:: python

    import h5py as h5
    import numpy as np
    import matplotlib.pyplot as plt

Open "qwoa.h5" as a read-only file

.. code-block:: python

   f = h5.File("qwao.h5", "r")


Load the final state into a numpy array:

.. code-block:: python

    final_state = np.array(f['example_config/final_state']).view(dtype = np.complex128)

.. note::
    The use of *view* ensures precision is not lost durring a datatype conversion.

Finally, lets examine the probability distribution:

.. code-block:: python

    probs = np.multiply(final_state, final_state)
    plt.plot(probs, '.')
    plt.show()
