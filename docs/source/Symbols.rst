Theory and Definitions
======================

The following assumes that the reader is familar with QAOA algorithms and aims to simply detail notation used throughout this documenation and some of the mathematical particulars of the simulation processes. This sections draws from "Efficient quantum walks over exponentially large sets of combinatorial objects for optimisation" by Samuel Mrach and Jingbo Wang, but the notation has been generalized to apply to both QAOA and its walk-assisted variant.

In the QAOA framework the system evolves as:

.. math::

    |\vec{\gamma},\vec{t}\rangle=U_W(t_p)U_Q(\gamma_p)...U_W(t_1)U_Q(\gamma_1)|s\rangle

#. :math:`|s\rangle = \sum_{i=0}^{2^N-1}|s_i\rangle` is the initial state where :math:`N` is the number of qubits. Each :math:`|s_i\rangle` is mapped to a solution of the optimization problem.

#. :math:`U_Q(\gamma)` applies a phase shift to each :math:`|s_i\rangle` proportional to :math:`\gamma` and the solution quality :math:`q_i \in \mathbb{R}`.

#. :math:`U_W(t)` performs 
