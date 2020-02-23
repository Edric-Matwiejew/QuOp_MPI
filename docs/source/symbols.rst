Definitions and Numerical Methods
=================================

Mathematical Framework
----------------------

The following assumes that the reader is Familiar with QAOA algorithms and aims to simply detail notation used throughout this documentation and some of the mathematical particulars of the simulation processes. This sections draws from "Efficient quantum walks over exponentially large sets of combinatorial objects for optimisation" by Sam Marsh and Jingbo Wang, but the notation has been generalized to apply to both QAOA and its walk-assisted variant.

In the QAOA framework the system evolves as:

.. math::

    | \vec{\gamma},\vec{t}\rangle=U_W(t_p)U_Q(\gamma_p)...U_W(t_1)U_Q(\gamma_1) | s\rangle


#. :math:`|s\rangle = \sum_{i=0}^{N - 1}|s_i\rangle` is the initial state where :math:`N = 2^n` with :math:`n` being number of qubits. Each :math:`|s_i\rangle` is mapped to a solution of the optimization problem.

#. :math:`U_Q(\gamma)` applies a phase shift to each :math:`|s_i\rangle` proportional to :math:`\gamma` and the solution quality :math:`q_i \in \mathbb{R}`.


#. :math:`U_W(t)` couples the :math:`|s_i\rangle` to form an topology over which the state probabilities are mixed. In QuOp_MPI this is conceptualized as a *continuous-times qunatum walk*.

#. The set of :math:`2p` parameters :math:`\vec{\gamma} = (\gamma_1,...\gamma_p)` and :math:`\vec{t} = (t_1,...,t_p)` are varied by a classical optimizer in order to maximize the average measured solutions and quality. A higher choice of :math:`p` leads to better solutions at the cost of a deeper circuit.

* :math:`U_Q(\gamma) = \exp(-i\gamma Q)` where :math:`Q \in \mathbb{R}^{N \times N}` is a diagonal operator defined by :math:`\vec{q}=\{q_i\}` such that :math:`\langle \vec{\gamma}, \vec{t} | Q | \vec{\gamma}, \vec{t} \rangle = \sum_{i=0}^{N - 1} p_i q_i` where :math:`p_i` is the probability associated with state :math:`| s_i \rangle`.

* :math:`U_W(t) = \exp(-itW)`, where :math:`W \in \mathbb{R}^{N \times N}` is a Hermitian adjacency matrix describing a graph topology such that :math:`W_{ij} > 0` if and only if there is an edge between states (or *verticies*) :math:`| s_i \rangle` and :math:`| s_j \rangle`. Elsewhere in literature :math:`W` is also referred to as the *mixing operator*.

QAOA-like algorithms seek to minimise the :meth:`~quop_mpi.MPI.objective` function,

.. math::

    f(\vec{\gamma}, \vec{t}) = \frac{q_{max} - \langle \vec{\gamma}, \vec{t} | Q | \vec{\gamma}, \vec{t} \rangle}{q_{max}},

where :math:`q_{max}` corresponds to the highest quality solution. This is achieved by passing :math:`f(\vec{\gamma}, \vec{t})` to a classical optimizer. QuOp_MPI uses the SciPy basinhopping technique, with L-BFGS-B as the underlying optimization algorithm with bounds :math:`\gamma_i, t_i > 0`. A lower :math:`f(\vec{\gamma}, \vec{t})` corresponds to a greater probability or measuring a high quality solution.


QAOA vs QWOA
------------

QuOp_MPI consists of two simulation classes, :class:`~quop_mpi.MPI.qaoa` and :class:`~quop_mpi.MPI.qwoa`, which are both subclasses of the :class:`~quop_mpi.MPI.system` class. The differentiating factor between the two is in how :math:`U_W(t)` is implemented.

* QAOA: :math:`W` is defined as an arbitrary Hermitian adjacency matrix, most generally that of the hypercube graph. :math:`U_W | s \rangle` is thus computed through a high precision approximation of the action of the matrix exponential via Taylor series approximation (as detailed here) and :math:`W` is stored in memory as a Compressed Sparse Row (CSR) matrix.

* QWOA: :math:`W` is defined as a *circulant* Hermitian adjacency matrix. :math:`U_W | s \rangle` is calculated using a fast Fourier transform (FFT) and the analytical solution for the eigenvalues of a circulant matrix. FFT capabilities are provided by the Fastest Fourier Transform in the West.

As FFT provides the most computationally efficient method in the vast majority of cases. If :math:`W` is circulant, the :class:`~quop_mpi.MPI.qwoa` class should be used.

Success Metrics
---------------

When comparing different QAOA-like algorithm a lower :math:`f(\vec{\gamma}, \vec{t})` is indicative of better performance. However this does not provide information about the minimum :math:`q_i` we are likely to measure. For this reason QuOp_MPI introduces the following success metric,

.. math::

    \text{success} = \left\{ \begin{array}{ c c}
        \text{True}, & P_{\text{success}} > P_{\text{target}}  \\
        \text{False}, & \text{otherwise}
    \end{array} \right.

where :math:`P_{\text{success}} = \sum_{\tilde{q}_i > \tilde{q}_\text{cutoff}} p_i` with :math:`\tilde{q}_\text{cutoff}` being the minimum acceptable :math:`\tilde{q}_i = \frac{q_i}{q_\text{max}}`. :math:`P_\text{target}` defines the minimum desired value of :math:`P_\text{success}`. The package defaults are :math:`\tilde{q}_\text{cutoff} = 0.9` and :math:`P_\text{target} = 2/3`. These can be re-defined via :meth:`~quop_mpi.MPI.system.state_success`.

.. note::

    The optimization result, accessed by :meth:`~quop_mpi.MPI.system.print_result` and saved to the .h5 file via :meth:`~quop_mpi.MPI.system.save`, also has a `success` output. This refers to the the convergence of the optimization process.
