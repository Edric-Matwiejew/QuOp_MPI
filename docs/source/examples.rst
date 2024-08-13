Examples
========

The following examples detail the simulation of :term:`QVAs <QVA>` for unconstrained and constrained combinatorial optimisation problems.

The examples are size so they may be easily ran on most personal computers. The examples must be ran using the ``mpiexec`` or ``mpirun`` launchers. For example, to run the QAOA maxcut example located in *examples/maxcut* on a system with 4 CPU cores:

.. code-block:: bash

    mpiexec -N 4 python3 maxcut.py

.. _maxcut:

Maxcut
------

The max-cut problem seeks to partition the vertices of a graph such that
a maximum number of neighbouring nodes are assigned to two disjoint sets
:cite:p:`farhi_quantum_2014`. A quantum encoding of the
max-cut problem is a bijective mapping of the vertices of a graph
:math:`G` to :math:`n` qubits, with the set membership indicated by the
corresponding qubit state. For example, a two vertex graph with vertices
:math:`{\left\{0,1\right\}}` has a solution space that is completely
represented by an equal superposition over a two-qubit system: 
:math:`{\left\{{\left\{0,1\right\}}\right\}} \rightarrow {\lvert 00\rangle}`, 
:math:`{\left\{{\left\{0\right\}},{\left\{1\right\}}\right\}} \rightarrow {\lvert 01\rangle}`, 
:math:`{\left\{{\left\{0\right\}},{\left\{1\right\}}\right\}} \rightarrow {\lvert 10\rangle}` 
and 
:math:`{\left\{{\left\{0,1\right\}}\right\}} \rightarrow {\lvert 11\rangle}`.

The cost function is then implemented as

.. math::
   :label: maxcut-cost

       {C(s)} = \sum_{E(i,j)\in G} \frac{1}{2}\left( \mathbb{I} + Z_iZ_j\right),

where :math:`Z_i` is a Pauli Z gate acting on the :math:`i^\text{th}`
qubit, :math:`E(i,j)` is an edge in :math:`G` connecting vertex
:math:`i` to vertex :math:`j`, and :math:`Z_iZ_j` has eigenvalue
:math:`1` if qubits :math:`i` and :math:`j` are in the same state or
:math:`-1` otherwise.

.. _QAOA-maxcut:

QAOA (Serial Quality Function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* *examples/maxcut/maxcut.py*

Here the QAOA is applied to the max-cut problem
for the graph shown below. The predefined
``Ansatz`` subclass :class:`~quop_mpi.algorithm.combinatorial.qaoa` forms the basis of the simulation.

.. figure:: _static/maxcut_graph.png
   :name: maxcut-graph
   :scale: 25%
   :align: center

   Graph for the example maxcut problem. The green and purple vertices indicate one of two optimal vertex partitioning.

To generate the graph we use the external package ``networkx``.
And define the cost function as a Python function using the ``I`` and
``Z`` functions from :mod:`~quop_mpi.toolkit`, we are able to directly
implement :math:numref:`maxcut-cost`.

.. literalinclude:: ../../examples/maxcut/maxcut.py
    :lines: 1-18

A :class:`~quop_mpi.algorithm.combinatorial.qaoa` instance is instantiated.
and the :math:`\text{diag}(\hat{Q})` (the solution qualities) is defined via the
:meth:`~quop_mpi.Ansatz.set_qualities` method. For this, we pass the :meth:`~quop_mpi.observable.serial`
:term:`Observables Function` along with a dictionary containing ``maxcut qualities`` and its
arguments. The ``serial`` function assists with memory-efficient
simulation, by calling the ``maxcut_qualities`` at the root MPI process and distributing its output over
Ansatz subcommunicator. The ansatz depth (:math:`D=2`) is defined via the :meth:`~quop_mpi.Ansatz.set_depth`.

.. literalinclude:: ../../examples/maxcut/maxcut.py
    :lines: 21-23

Now that the ``qaoa`` instance is fully specified, simulation of the
algorithm (see :ref:`QAOA`) proceeds via
:meth:`~quop_mpi.Ansatz.execute`. By calling ``execute`` without specifying
the initial :term:`variational parameters` we use the default :term:`Parameter Functions <Parameter Function>`, which
generates `variational parameters` from a uniform distribution over
:math:`(0 \pi, 2\pi]`.

.. literalinclude:: ../../examples/maxcut/maxcut.py
    :lines: 25

Finally, the optimiser result is displayed using
:meth:`~quop_mpi.Ansatz.print_result` and the simulation results are saved
to the HDF5 file ‘maxcut.h5’ under the ‘depth 2’ group. 

.. literalinclude:: ../../examples/maxcut/maxcut.py
    :lines: 27-28

The figure below illustrates the initial and final probability distributions with respect
of the unique values of :math:`q_i` (see :term:`observables`). After application of the QAOA to the
initial superposition, probability density is concentrated at
high-quality solutions with the optimal solution
having the highest probability of measurement.

.. _maxcut_qaoa_initial_and_final:

.. list-table::

   * - .. figure:: _static/maxcut_starting_probabilities.png
     - .. figure:: _static/maxcut_qaoa_probabilities.png

.. cssclass:: center
    
*The starting probability of the maxcut problem solution qualities (left) and the equivalent probability distribution after execution of the QAOA.*

.. _QAOA-parallel-quality:

QAOA (Parallel Observables Function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* *examples/maxcut/maxcut_parallel_qualities.py*

Here we consider a variation on the above :ref:`QAOA example <QAOA-maxcut>`, whereby the cost function given in :math:numref:`maxcut-cost` is computed via a user-defined :term:`Operator Function`. As, previously we require :class:`~quop_mpi.algorithms.combinatorial.qaoa` and ``networkx``. We also import ``NumPy``. 

.. literalinclude:: ../../examples/maxcut/maxcut_parallel_qualities.py
    :lines: 1-3

The graph is generated using ``networkx`` and the :term:`system size` computed from the number of graph nodes.

.. literalinclude:: ../../examples/maxcut/maxcut_parallel_qualities.py
    :lines: 5-8

An :term:`Operator Function` is a :term:`QuOp Function` that returns the matrix :term:`operator` of a :term:`unitary`. The :ref:`QAOA` :term:`phase-shift unitary` has a diagonal matrix operator that contains the solution qualities (which also define the QAOA :term:`observables`). An Operator Function for the computation of the solution qualities must return a 1-D real array containing ``local_i`` elements with global index offset ``local_i_offset``. We are able to compute the maxcut solution qualities in parallel as computation of the quality for any specific solution is independent of the global solution space.

.. literalinclude:: ../../examples/maxcut/maxcut_parallel_qualities.py
    :lines: 11-24

The ``parallel_maxcut_qualities`` :term:`Operator Function` is passed to :meth:`~quop_mpi.algorithm.combinatorial.qaoa.set_qualities`. As, its ``local_i`` and ``local_i_offset`` arguments are attributes of the :class:`~quop_mpi.Ansatz` class, they will be passed to the function automatically. The ``G`` argument is specified as an additional positional argument in a corresponding :term:`FunctionDict`.

.. literalinclude:: ../../examples/maxcut/maxcut_parallel_qualities.py
    :lines: 27-32

Ex-QAOA
^^^^^^^

* *examples/maxcut_extended/maxcut_extended.py*

Here we address the maxcut problem :ref:`defined above<QAOA-maxcut>` using the :ref:`Ex-QAOA`. We will do this by implementing a :term:`QVA` using the :class:`~quop_mpi.Ansatz` base-class, the :mod:`~quop_mpi.propagator.diagonal` ``propagator`` to simulate the action of the Ex-QAOA :term:`phase-shift unitary` and the :class:`~quop_mpi.propagator.sparse` ``propagator`` to simulate the action of the Ex-QAOA :term:`mixing unitary`. The :meth:`~quop_mpi.observable.serial` :term:`Observables Function` will be used to interface a serial Python function for computation of the maxcut solution qualities (see :math:numref:`maxcut-cost`) with QuOp_MPI. Finally, the :meth:`~quop_mpi.toolkit.Z` Pauli operator function from :mod:`~quop_mpi.toolkit` will be used to efficiently implement the parameterised Ex-QAOA phase-shift unitary matrix :term:`operator`.

.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 1-7

The problem graph is generated using ``networkx`` and the :term:`system size` computed from the number of graph vertices.

.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 9-13

The ``maxcut_terms`` functions implements the Ex-QAOA phase-shift unitary by returning a ``list`` of 1-D arrays that correspond to the non-identity Pauli terms in the problem cost function (see :math:numref:`qaoa-cost)`). Each of these terms will associated with a phase-shift unitary with independently parameterised :term:`operator parameters <operator parameter>`. 


.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 14-22

Due to these extra degrees of freedom, the the Ex-QAOA phase-shift unitary does not contain the :term:`QVA` :term:`observables`. As such, the observables are independently computed via the ``max_qualities`` function, which sums the output of ``maxcut_terms`` returning the solution qualities as defined in :math:numref:`maxcut-cost`.

.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 24-25

The phase-shift unitary ``UQ`` is implemented via an instance of the ``diagonal`` :class:`~quop_mpi.propagator.diagonal.unitary` class. The ``unitary_n_params`` specifies the number of :term:`unitary parameters <unitary parameter>` associated with the unitary.

.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 27-33

The :term:`ansatz unitary` is specified by passing ``[UQ, UW]`` to :meth:`~quop_mpi.Ansatz.set_unitaries`. Simulation then proceeds with the default :class:`quop_mpi.Ansatz `:term:`initial state` (an equal superposition).

.. literalinclude:: ../../examples/maxcut_extended/maxcut_extended.py
    :lines: 35-41

.. figure:: _static/maxcut_extended_qaoa_probabilities.png
    :name: maxcut-extended-final-state
    :scale: 50%
    :align: center

    Probability distribution of the maxcut problem solution qualities after execution of the Ex-QAOA.

.. _portfolio:

Portfolio Rebalancing
---------------------

To explore the case of constrained optimisation using the :ref:`QWOA` and the
:ref:`QAOAz` we will consider the problem of portfolio re-balancing. For each
asset in a portfolio of size :math:`M`, an investor must choose one of
the following positions:

#. Short position: buying and selling an asset with the expectation that
   it will drop in value.

#. Long position: buying and holding the asset with the expectation that
   it will rise in value.

#. No position: taking neither the long or short position.

A quantum encoding of the possible solutions uses two qubits per asset.

#. :math:`{\lvert 01\rangle} \rightarrow \text{short position}`

#. :math:`{\lvert 10\rangle} \rightarrow \text{long position}`

#. :math:`{\lvert 00\rangle}` or
   :math:`{\lvert 11\rangle} \rightarrow \text{no position}`

The discrete mean-variance Markowitz model provides a means of
evaluating the quality associated with a given combination of positions.
It can be expressed through minimisation of the cost function,

.. math:: {C(s)} = \omega \sum_{i,j = 1}^{M} \sigma_{ij}Z_iZ_j - (1 - \omega) \sum_{i = 1}^{M} r_iZ_i,

subject to the constraint,

.. math:: {\chi_{\text{asset}}(s)} = \sum_{i = 1}^{M} z_i.

In this formulation, the Pauli-Z gates :math:`Z_i` encode a particular
portfolio where, for each asset, eigenvalue
:math:`z_i \in {\left\{1,-1,0\right\}}` represents a choice of long,
short or no position. Associated with each asset is the expected return
:math:`r_i` and covariance :math:`\sigma_{ij}` between assets :math:`i`
and :math:`j`; which are calculated using historical data. The risk
parameter, :math:`\omega`, weights consideration of :math:`r_i` and
:math:`\sigma_{ij}` such that as :math:`\omega \rightarrow 0` the
optimal portfolio is one providing maximum returns. In contrast, as
:math:`\omega \rightarrow 1`, the optimal portfolio is the one that
minimises risk. The constraint :math:`{\chi_{\text{asset}}(s)}` works to
maintain the relative net position with respect to a pre-existing
portfolio :cite:p:`slate_quantum_2021`.

In the following examples, we demonstrate the application of the QWOA
and QAOAz to a small ‘portfolio’ consisting of four assets taken from
the ASX 100, under the constraint :math:`{\chi_{\text{asset}}(s)} = 2`.


QAOAz
^^^^^

* *examples/portfolio_rebalancing/qaoaz_portfolio.py*
* *examples/portfolio_rebalancing/qaoaz_qualities.py*

A :ref:`QAOAz` approach to the portfolio optimisation problem uses two parity
mixers that act on the short and long qubits, respectively, such that
the :math:`{\mathcal{S}}` is partitioned into subgraphs of the same
:math:`{\chi_{\text{asset}}(s)}` value. For this example, we are
considering four assets so the two parity mixers act on separate
subspaces of :math:`\mathcal{H}` as shown below:

.. figure:: _static/portfolio_parity.png
    :name: qaoaz-subspaces
    :scale: 25%
    :align: center

Where :math:`{\lvert l\rangle}` denotes a ‘long’ qubit,
:math:`{\lvert s\rangle}` denotes a ‘short’ qubit, and the numbering
indicates the global index of each qubit.

To constrain probability amplitude to :math:`{\mathcal{S}^\prime}`,
:math:`{{\lvert\psi_0\rangle}_\text{QAOAz}}` is prepared as

.. math:: 
    :name: qaoaz-portfolio-state
    
    {{\lvert\psi_0\rangle}_\text{QAOAz}} = {\lvert 01\rangle}^{\otimes A}\left( \frac{1}{\sqrt{2}}({\lvert 00\rangle} + {\lvert 11\rangle})^{2N-A} \right),

where :math:`A` is the desired value of
:math:`{\chi_{\text{asset}}(s)}`. This creates a (non-equal)
superposition of states across all qubit subgraphs with a net parity of
:math:`A`.

Here we implement the QAOAz using the :class:`~quop_mpi.Ansatz` base-class, the :mod:`~quop_mpi.propagator.diagonal` ``propagator`` module, the :mod:`~quop_mpi.propagator.sparse` ``propagator`` module and the :mod:`quop_mpi.state` module. The :meth:`~quop_mpi.toolkit.string`, :meth:`~quop_mpi.toolkit.X`, :mod:`~quop_mpi.toolkit.Y`, :meth:`~quop_mpi.toolkit.kron` and :meth:`~quop_mpi.toolkit.kron_power` functions from :mod:`~quop_mpi.toolkit` are imported to assist with the definition of functions for the QAOAz parity :term:`mixing unitary` and :term:`initial state` (see :math:numref:`qaoaz-portfolio-state`). The ``qaoaz_portfolio`` function imported from ``qaoaz_qualities`` (*examples/portfolio_rebalancing/qaoaz_qualities.py*) is an :term:`Operator Function` that computes the solution qualities based on adjusted close price historical data obtained from Yahoo Finance.

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 1-6

The ``parity_ring``, ``parity_mixer`` and ``mixer`` functions generate the :ref:`QAOAz` :term:`mixing unitary` :term:`operator` as a CSR sparse matrix.

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 9-39

Next, the ``parity_state`` function implements :math:numref:`qaoaz-portfolio-state`,

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 42-46

and the :term:`system size` is calculated from the number of qubits required to represent 4 assets.

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 49-50

The :term:`phase-shift <phase-shift unitary>` and :term:`mixing <mixing unitary>` unitaries are defined via instances of the :mod:`~quop_mpi.propagator.diagonal` and :mod:`~quop_mpi.propagator.sparse` ``unitary`` classes.


.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 52-57

The :term:`ansatz unitary` is defined by passing ``[UQ, UW]`` to :meth:`~quop_mpi.Ansatz.set_unitaries` and the :term:`observables` defined by passing the index of ``UQ`` to :meth:`~quop_mpi.Ansatz.set_observables`. 

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 59-62

To assist with the analysis of :term:`QVA` performance, QuOp supports the recording of important simulation metrics in a log file. The :meth:`~quop_mpi.Ansatz.set_log` method specifies the recording of the :term:`system size`, :term:`ansatz depth`, optimised :term:`objective function` value, norm of the :term:`final state`, in-program simulation time, MPI communicator size, number of :term:`objective function` evaluates and the success status of the classical :term:`optimiser` to :literal:`qaoaz_portfolio.csv` for each simulation instance.

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 64

We are often interested in evaluating the performance of a :term:`QVA` as a function of the :term:`ansatz depth`. For this, we use :meth:`~quop_mpi.Ansatz.benchmark`, which will carry out a sequence of calls to :meth:`~quop_mpi.Ansatz.execute` over the ansatz depth range of 1 to 5, with 3 repeats per ansatz depth.

.. literalinclude:: ../../examples/portfolio_rebalancing/qaoaz_portfolio.py
    :lines: 65-67

QWOA
^^^^

* *examples/portfolio_rebalancing/qwoa_portfolio.py*
* *examples/portfolio_rebalancing/qwoa_qualities.py*

Here we address the portfolio optimisation problem using the :ref:`QWOA`, which carries out a quantum search over the subspace of valid solutions. For this we will use the predefined :class:`~quop_mpi.algorithm.combinatorial.qwoa` algorithm and the :meth:`~quop_mpi.propagator.diagonal.operator.csv` :term:`Operator Function`.

.. literalinclude:: ../../examples/portfolio_rebalancing/qwoa_portfolio.py
    :lines: 1

The :term:`system state` is set equal to the number of valid solutions (``31``),

.. literalinclude:: ../../examples/portfolio_rebalancing/qwoa_portfolio.py
    :lines: 3-4

and the :term:`observables` and :term:`phase-shift unitary` matrix :term:`operator` specified by :meth:`~quop_mpi.algorithm.combinatorial.qaoa.set_qualities`. The ``'args'`` and ``'kwargs'`` items in the corresponding :term:`FunctionDict` are passed to the pandas `read_csv` function. The solution quality values are retrieved from ``qwoa_qualities.csv``, which have been precomputed using *examples/portfolio_rebalancing/qwoa_qualities.py*.

.. literalinclude:: ../../examples/portfolio_rebalancing/qwoa_portfolio.py
    :lines: 6-12

Finally, the logging of simulation data is specified and the algorithm simulated from :term:`ansatz depths <ansatz depth>` 1 to 6 with 3 repeats per ansatz depth using :meth:`~quop_mpi.Ansatz.benchmark`.

.. literalinclude:: ../../examples/portfolio_rebalancing/qwoa_portfolio.py
    :lines: 14-21

.. figure:: _static/portfolio_rebalancing.png
    :name: qaoaz-vs-qwoa
    :scale: 50%
    :align: center

    Final objective function value achieved by the QAOAz and the QWOA for a portfolio optimisation problem with 4 assets from ansatz depths 1 to 5. each point depicts the average of three repeats.