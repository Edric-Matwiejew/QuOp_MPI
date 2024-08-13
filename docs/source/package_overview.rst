Package Overview
================

QuOp_MPI provides an objected-oriented framework for the design and simulation of :term:`QVAs <QVA>`. It enables researchers with any level of parallel programming experience to design simulation workflows that are efficiently scalable on massively parallel systems.

QVA Simulation
--------------

Predefined Algorithms
^^^^^^^^^^^^^^^^^^^^^

For combinatorial optimisation problems:

* :class:`~quop_mpi.algorithm.combinatorial.qaoa` (see :ref:`QAOA` and the :ref:`maxcut with QAOA <maxcut>`)
* :class:`~quop_mpi.algorithm.combinatorial.qwoa` (see :ref:`QWOA` and the :ref:`portfolio rebalancing with QWOA <portfolio>` example)

For the optimisation of continuous multivariable functions.

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

For clusters with time-limited job-scheduling, the :meth:`~quop_mpi.Ansatz.benchmark`, :meth:`~quop_mpi.meta.swarm.execute_swarm` and :meth:`~quop_mpi.meta.swarm.benchmark_swarm` methods support automated job-suspension and resumption.