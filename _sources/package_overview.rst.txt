Package Overview
================

QuOp_MPI provides an object-oriented framework for the design and simulation of :term:`QVAs <QVA>`. It enables researchers with any level of parallel programming experience to design simulation workflows that are efficiently scalable on massively parallel systems.

QVA Simulation
--------------

Predefined Algorithms
^^^^^^^^^^^^^^^^^^^^^

For combinatorial optimisation problems:

* :class:`~quop_mpi.algorithm.combinatorial.qaoa` (see :ref:`QAOA` and the :ref:`maxcut with QAOA <maxcut>`)
* :class:`~quop_mpi.algorithm.combinatorial.qwoa` (see :ref:`QWOA` and the :ref:`portfolio rebalancing with QWOA <portfolio>` example)

For the optimisation of continuous multivariable functions:

* :class:`~quop_mpi.algorithm.multivariable.qowe` (see :ref:`QOWE`)
* :class:`~quop_mpi.algorithm.multivariable.qmoa` (see :ref:`QMOA`)

User-Defined Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

Novel QVAs may be designed by working directly with the :class:`~quop_mpi.Ansatz` class and :mod:`~quop_mpi.propagator` submodules. See,

* :ref:`Maxcut with the Ex-QAOA <maxcut>` (also :ref:`Ex-QAOA`).
* :ref:`Portfolio optimisation with the QAOAz <portfolio>` (also :ref:`QAOAz`).

Adaptive Operator and Optimisation Schemes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The QuOp_MPI :class:`~quop_mpi.Ansatz` and :class:`~quop_mpi.Unitary` classes are configured via :term:`QuOp Functions <QuOp Function>`. These allow the implementation of arbitrarily parameterised operators and adaptive optimisation schemes. QuOp_MPI includes default QuOp Functions that support the interfacing of user-defined serial Python functions with its :ref:`parallelisation scheme for QVA simulation <parallel-QVA>`. Users may also define MPI-compatible custom QuOp Functions with minimal parallel programming experience.

Key Features
------------

Toolkit Module
^^^^^^^^^^^^^^

The :mod:`~quop_mpi.toolkit` module provides convenience functions for constructing quantum operators:

* **Pauli operators**: :func:`~quop_mpi.toolkit.I`, :func:`~quop_mpi.toolkit.X`, :func:`~quop_mpi.toolkit.Y`, :func:`~quop_mpi.toolkit.Z` — single-qubit Pauli matrices acting on specified qubits in an n-qubit system
* **Kronecker products**: :func:`~quop_mpi.toolkit.kron`, :func:`~quop_mpi.toolkit.kron_power` — utilities for constructing tensor products
* **String operators**: :func:`~quop_mpi.toolkit.string` — for building operators from string representations

These are particularly useful for defining cost Hamiltonians in combinatorial optimisation problems (see the :ref:`maxcut example <maxcut>`).

Initial States and Observables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QuOp_MPI provides predefined functions for common initial states and observable configurations:

* **Initial states** (:mod:`~quop_mpi.state`): ``equal`` (uniform superposition), ``basis`` (computational basis state), ``serial`` (user-defined function), ``array`` (from NumPy array), ``position_grid`` (for multivariable problems)
* **Observables** (:mod:`~quop_mpi.observable`): ``serial`` (user-defined function), ``csv`` (from CSV file), ``hdf5`` (from HDF5 file), ``array`` (from NumPy array), ``rand`` (random observables for testing)

See :meth:`~quop_mpi.Ansatz.set_initial_state` and :meth:`~quop_mpi.Ansatz.set_observables`.

Parameter Mapping
^^^^^^^^^^^^^^^^^

The :meth:`~quop_mpi.Ansatz.set_parameter_map` method enables flexible control over variational parameters, allowing:

* Reduction of the number of free parameters (e.g., parameter sharing across ansatz layers)
* Custom mappings from a reduced parameter space to the full variational parameter vector
* Improved optimisation efficiency for problems with inherent symmetries

Optimiser Support
^^^^^^^^^^^^^^^^^

QuOp_MPI supports classical optimisers from both SciPy and NLopt:

* **SciPy**: All methods from ``scipy.optimize.minimize`` (default: L-BFGS-B)
* **NLopt**: Gradient-based and derivative-free methods (requires ``pip install 'quop_mpi[nlopt]'``)

Configure via :meth:`~quop_mpi.Ansatz.set_optimiser`.

Custom Objective Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

While QuOp_MPI defaults to minimising the expectation value of the observables, custom objective functions can be defined via :meth:`~quop_mpi.Ansatz.set_objective`. This enables:

* Alternative figures of merit (e.g., CVaR, Gibbs objective)
* Multi-objective optimisation schemes
* Problem-specific objective formulations

Sampling Simulation
^^^^^^^^^^^^^^^^^^^

The :meth:`~quop_mpi.Ansatz.set_sampling` method enables simulation of quantum measurement, returning sampled basis states and their associated observable values rather than just the expectation value.

Data I/O
^^^^^^^^

QuOp_MPI provides comprehensive data persistence:

* **HDF5 output**: Save final states, observables, and optimisation results via :meth:`~quop_mpi.Ansatz.save` (parallel HDF5 for large-scale simulations)
* **CSV logging**: Record optimisation progress across multiple runs via :meth:`~quop_mpi.Ansatz.set_log`
* **Benchmark data**: Automated saving during :meth:`~quop_mpi.Ansatz.benchmark` runs

Parallelisation Schemes
-----------------------

QuOp_MPI implements several MPI-based parallelisation schemes, which are all mutually compatible. Of these, :ref:`parallel Quantum simulation <parallel-QVA>` is relevant to :term:`QVA` simulation on personal computers, workstations and clusters. While :ref:`parallel gradient computation <parallel-gradient>` and :ref:`Ansatz swarms <parallel-swarm>` support simulation workflows on clusters.

.. _parallel-QVA:

Quantum Simulation
^^^^^^^^^^^^^^^^^^

For an MPI (sub)communicator with :math:`m` processes with integer ID :math:`\text{rank} \in (0,...,m-1)`, the :term:`system state` and :term:`observables` of a simulated :term:`QVA` are partitioned over an MPI (sub)communicator with,

.. math::

    \text{local_i}

elements per rank and a global index offset of,

.. math::

    \text{local_i_offset} = \sum_{i<\text{rank}} \text{local_i}.

These distributed arrays are acted on by instances of the :class:`~quop_mpi.Unitary` class, which provides an interface to efficient Python extensions which compute the action of the QVA unitaries in MPI parallel. 


.. _parallel-gradient:

Gradient Evaluation
^^^^^^^^^^^^^^^^^^^

For :term:`optimisation <optimiser>` methods that make use of gradient information, computation of the :term:`objective function` gradient may be carried out in MPI parallel by duplicating an :class:`~quop_mpi.Ansatz` over multiple MPI subcommunicators (see :meth:`~quop_mpi.Ansatz.set_parallel_jacobian`).

.. _parallel-swarm:

Ansatz Swarms
^^^^^^^^^^^^^

Optimisation of QVA :term:`variational parameters` over a large search domain, or other QVA meta-optimisation tasks can be accelerated through creation of an :class:`~quop_mpi.meta.swarm`, which manages multiple QVA simulation instances.

Parallel Overview
-----------------

The diagram below depicts a :class:`~quop_mpi.meta.swarm` of two :term:`QVA` simulation instances with parallel gradient evaluation (see :meth:`~quop_mpi.Ansatz.set_parallel_jacobian`). Each QVA simulation occurs over three MPI subcommunicators with two of the subcommunicators carrying out computation of the partial derivatives of the :term:`objective function` and the remaining managing :term:`optimisation <optimiser>` of the :term:`variational parameters` and evaluation of the objective function. The six :class:`~quop_mpi.Ansatz` subcommunicators call the :meth:`~quop_mpi.Unitary.propagate` method of :class:`~quop_mpi.Unitary` instances which compute the action of the QVA's :term:`phase-shift <phase-shift unitary>` and :term:`mixing <mixing unitary>` unitaries in MPI parallel.

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                node [fontsize="10"];

                mpi[label="MPI.COMM_WORLD\n12 nodes, 1536 cores", shape=oval]

                swarm[label="Interrelated QVA simulations.\n(swarm)", shape="trapezium"];

                ansatz_0[label="QVA Simulation 1\n(Ansatz)",shape="invhouse"];
                ansatz_N[label="QVA Simulation 2\n(Ansatz)",shape="invhouse"];

                gradient_0_0[label="Gradient\ncomputation.\n(Ansatz)",shape="rectangle"];
                gradient_0_N[label="Gradient\ncomputation.\n(Ansatz)",shape="rectangle"];
                gradient_N_0[label="Gradient\ncomputation.\n(Ansatz)",shape="rectangle"];
                gradient_N_N[label="Gradient\ncomputation.\n(Ansatz)",shape="rectangle"];

                propagation_0[label="Objective function\nminimisation.\n(Ansatz)",shape="rectangle"];
                propagation_N[label="Objective function\nminimisation.\n(Ansatz)",shape="rectangle"];

                propagation_gradient_0_0[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];
                propagation_gradient_0_N[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];
                propagation_gradient_N_0[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];
                propagation_gradient_N_N[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];

                obj_propagation_0[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];
                obj_propagation_N[label="Parallel quantum\nstate propagation\n(Unitary)", shape="rectangle"];

                mpi -> swarm [style=dashed, dir=none];

                swarm-> ansatz_0 [label=768];
                swarm-> ansatz_N [label=768];

                ansatz_0 -> propagation_0 [label=256];
                ansatz_N -> propagation_N [label=256];

                propagation_0 -> obj_propagation_0 [style=dashed, dir=none];
                propagation_N -> obj_propagation_N [style=dashed, dir=none];

                ansatz_0 -> gradient_0_0 [label=256];
                ansatz_0 -> gradient_0_N [label=256];

                ansatz_N -> gradient_N_0 [label=256];
                ansatz_N -> gradient_N_N [label=256];

                gradient_0_0 -> propagation_gradient_0_0 [style="dashed", dir="none"];
                gradient_0_N -> propagation_gradient_0_N [style="dashed", dir="none"];
                gradient_N_0 -> propagation_gradient_N_0 [style="dashed", dir="none"];
                gradient_N_N -> propagation_gradient_N_N [style="dashed", dir="none"];

            }

.. cssclass:: center

*Example parallel structure for QuOp_MPI running on a cluster with 128 cores per node. Solid arrows indicate splitting of an MPI (sub)communicator and dashed lines sharing of an MPI (sub)communicator. Numbered edges indicate MPI subcommunicator size. Relevant QuOp_MPI classes are indicated in parenthesises.*

Support for Clusters with Job-Scheduling
----------------------------------------

For clusters with time-limited job-scheduling, QuOp_MPI supports automated job suspension and resumption for long-running workflows. This allows simulations to be split across multiple job submissions without manual intervention.

.. note::

    Suspend/resume functionality is available for **multi-execution workflows only**:
    
    * :meth:`~quop_mpi.Ansatz.benchmark` — systematic studies across ansatz depths and repeats
    * :meth:`~quop_mpi.meta.swarm.execute_swarm` — parallel execution of multiple QVA instances
    * :meth:`~quop_mpi.meta.swarm.benchmark_swarm` — benchmarking across swarm configurations
    
    A single :meth:`~quop_mpi.Ansatz.execute` call cannot be suspended and resumed, as it represents one atomic optimisation run.

Suspend and Resume
^^^^^^^^^^^^^^^^^^

When a time limit is set, QuOp_MPI monitors execution time and suspends before the limit is reached, saving progress to a suspend file. On the next job submission, execution resumes from where it left off—completed iterations are skipped and only remaining work is performed.

Example usage:

.. code-block:: python

    alg.benchmark(
        ansatz_depths=range(1, 21),
        repeats=10,
        time_limit=3600,        # Suspend after ~1 hour
        suspend_path="my_simulation"
    )

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

The suspend/resume behaviour can be configured via environment variables, which is useful for cluster job scripts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``QUOP_TIME_LIMIT``
     - Total allocated time in seconds. Overrides the ``time_limit`` parameter.
   * - ``QUOP_SUSPEND_PATH``
     - Path for suspend files. Overrides the ``suspend_path`` parameter.
   * - ``QUOP_FORCE_RESUME``
     - If set to ``1``, force resume from suspend file even if source code has changed.

Example SLURM job script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --time=04:00:00
    #SBATCH --nodes=4

    export QUOP_TIME_LIMIT=14000  # ~3.9 hours (leave margin for cleanup)
    export QUOP_SUSPEND_PATH="simulation_checkpoint"

    srun python my_simulation.py

Performance Profiling
^^^^^^^^^^^^^^^^^^^^^

For performance analysis, QuOp_MPI includes a built-in profiler that traces function calls and execution times:

.. code-block:: bash

    QUOP_PROFILE=1 mpiexec -n 4 python my_simulation.py

This creates a ``quop_profile_<timestamp>/`` directory containing per-rank trace files with timing information for all QuOp_MPI function calls.