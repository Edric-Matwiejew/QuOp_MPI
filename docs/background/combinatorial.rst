Combinatorial Optimisation
--------------------------

Quantum Variational Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a quantum system of size :math:`N = 2^n`, where integer :math:`n` is
a number of qubits with basis states
:math:`{\left\{{\lvert 0\rangle} = \begin{pmatrix}   0 \\ 1 \end{pmatrix},{\lvert 1\rangle} = \begin{pmatrix}   1 \\ 0 \end{pmatrix}\right\}}`,
QuOp_MPI defines a generalised QVA as

.. math::
   :label: eq:variational_generic

   {{\lvert\bm{\theta}\rangle}}=\left( \prod_{i = 1}^{p}{\hat{U}_{}}(\theta_i) \right){{\lvert\psi_0\rangle}}

where :math:`{{\lvert\psi_0\rangle}} \in \mathbb{C}^{N}` is an
initial quantum state with basis states
:math:`{\left\{{\lvert i\rangle}\right\}}_{i=0,...,N-1}`,
:math:`{\hat{U}_{}} \in \mathbb{C}^{N \times N}` is the
ansatz unitary , integer :math:`p \geq 0` specifies the number of
applications of :math:`{\hat{U}_{}}` to
:math:`{{\lvert\psi_0\rangle}}` (the ‘depth’) and
:math:`\bm{\theta}= \{ \theta_i \in \mathbb{R} \}` is an ordered set of
classically tunable values that parameterise
:math:`{\hat{U}_{}}`. The ansatz unitary
:math:`{\hat{U}_{}}` and initial quantum state
:math:`{{\lvert\psi_0\rangle}}` together define a specific QVA.

A Quantum Variational Algorithm is executed by repeatedly preparing
:math:`{{\lvert\bm{\theta}\rangle}}` and measuring the
expectation value

.. math::
   :label: eq:objective_generic

   f(\bm{\theta}) = {\langle\bm{\theta}\rvert} \hat{Q}{\lvert\bm{\theta}\rangle}_\text,

where :math:`\hat{Q}\in \mathbb{R}^{N \times N}` is a diagonal matrix
operator with entries :math:`\text{diag}(\hat{Q}) = q_i` that specify
the ‘quality’ associated with quantum state :math:`{\lvert i\rangle}`.
The variational parameters :math:`\bm{\theta}` are updated using a
classical optimiser with the objective being minimisation of
:math:`f`.

The ansatz operator :math:`{\hat{U}_{}}` specifies a sequence of
alternating unitaries. This can include phase-shifts

.. math::
   :label: eq:phase_shift_generic

   {\hat{U}_{\text{phase}}}(\gamma) = \exp(-\text{i} \gamma \hat{O}),

where
:math:`\hat{O} = \sum_{i=0}^{N-1} o_i {\lvert i\rangle}{\langle i\rvert}`
is a diagonal phase-shift matrix operator,
:math:`\gamma \in \bm{\theta}` and :math:`\hat{U}_\text{phase}`
applies a phase-shift proportional to :math:`o_i`, as well as
mixing-unitaries

.. math::
   :label: eq:mixing_generic

   {\hat{U}_{\text{mix}}}(t) = \exp(-\text{i} t \hat{W}),

where :math:`t \in \bm{\theta}` is non-negative and
:math:`\hat{W} = \sum_{{i,j} = 0}^{n-1}w_{ij}{\lvert j\rangle}{\langle i\rvert}`
is a mixing matrix operator in which non-diagonal entries specify
coupling between states :math:`{\lvert i\rangle}` and
:math:`{\lvert j\rangle}`. Mixing-unitaries :math:`\hat{U}_\text{mix}`
drive the transfer of probability amplitude between quantum states,
during which encoded phase differences contribute to constructive and
destructive interference.

Phase-shift operators :math:`\hat{O}` and mixing operators
:math:`\hat{W}` may also be parameterised by :math:`\bm{\theta}`. As
these operators are time-independent Hamiltonians of the time-evolution
operator, changes to the corresponding :math:`\theta_i` alter the
element-wise magnitudes or structure of the matrix exponent before
preparation of :math:`{{\lvert\bm{\theta}\rangle}}`.

Typically, the ansatz unitary :math:`{\hat{U}_{}}` is applied
:math:`p` times to :math:`{{\lvert\psi_0\rangle}}` with each
repetition parameterised by subset
:math:`\theta \subseteq \bm{\theta}`. Doing so increases the potential
for constructive and destructive inference to concentrate probability
amplitude at high-quality solutions; at the expense of classical
optimisation over a larger parameter space and a deeper quantum circuit.
In practice, a QVA must balance the improved convergence afforded by
increases to :math:`p` against the ability of the quantum hardware to
maintain coherence over a longer sequence of quantum operations.

The following sections introduce four distinct QVAs for
solving constrained and unconstrained COPs. We summarise here the
following notational conventions for a given QVA:

-  :math:`n`: the number of qubits.

-  :math:`{{\lvert\psi_0\rangle}}`: the initial quantum state
   vector.

-  :math:`{\hat{U}_{}}`: a sequence of phase-shift and mixing
   operators.

-  :math:`{\lvert\psi\rangle}`: :math:`{{\lvert\psi_0\rangle}}`
   after :math:`p \geq 0` applications of :math:`{\hat{U}_{}}`.

-  :math:`{{\lvert\bm{\theta}\rangle}}`:
   :math:`{{\lvert\psi_0\rangle}}` after :math:`p \geq 1`
   applications of :math:`{\hat{U}_{}}`.

-  :math:`\bm{\theta}`: classically tunable variables parameterising
   :math:`{\hat{U}_{}}` with starting values
   :math:`\bm{\theta}_0` and optimised values :math:`\bm{\theta}_f`.

-  :math:`f`: the ansatz objective function.

Combinatorial Optimisation with QVAs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combinatorial optimisation problems seek optimal solutions
:math:`{\Bar{s}}` of the form,

.. math:: {\Bar{s}}= \Big\{s\; | \; {C(s)} \in  \min {\left\{{C(s)} \; | \; s\in {\mathcal{S}^\prime}\right\}}\Big\},

where the problem cost-function :math:`{C(s)}` maps a solution
:math:`s` from an ordered set of problem solutions
:math:`\mathcal{S} = {\left\{s_i\right\}}` to :math:`\mathbb{R}`,
:math:`s` is a :math:`k`-permutation of discrete elements from a finite
set :math:`{\bm{\zeta}}` and

.. math:: {\mathcal{S}^\prime}= {\left\{ s\; | \; s\in {\bm{\chi}}\right\}}

is the problem-specific valid solution space where

.. math:: {\bm{\chi}}= \bigcup_i \Big\{s\; | \; \chi_i(s) = a_i\Big\}

denotes any constraints on :math:`{\Bar{s}}` and
:math:`\bm{a} = {\left\{a_i\right\}}` defines the constraints.

Problems of this type are often difficult to solve as
:math:`{\mathcal{S}}` grows factorially with
:math:`{\left|{\bm{\zeta}}\right|}` and, in general, lacks
identifiable structure. For this reason, heuristic and metaheuristic
algorithms are often used to find solutions that satisfy the relaxed
condition of :math:`{C({\Bar{s}})}` being a ‘sufficiently low’ local
minimum.

To apply a quantum variational algorithm to a given combinatorial
optimisation problem, an injective map is defined between
:math:`{\mathcal{S}}` and :math:`\mathcal{H}` with the cost-function
values forming the diagonal of the quality operator
:math:`\text{diag}(\hat{Q})= {C(s_i)}`. For example, a problem with four
solutions, :math:`\mathcal{S} = \{s_0, s_1, s_2, s_3\}`, maps to a
two-qubit system as

.. math::

   \begin{aligned}
           {\lvert 00\rangle} &= {\lvert 0\rangle} \rightarrow {\lvert s_0\rangle} \\
           {\lvert 01\rangle} &= {\lvert 1\rangle} \rightarrow {\lvert s_1\rangle} \\
           {\lvert 10\rangle} &= {\lvert 2\rangle} \rightarrow {\lvert s_2\rangle} \\
           {\lvert 11\rangle} &= {\lvert 3\rangle} \rightarrow {\lvert s_3\rangle},
       \end{aligned}

where
:math:`\text{diag}(\hat{Q})= \Big(C(s_0),C(s_1),C(s_2),C(s_3)\Big)`.

For a combinatorial optimisation problem to be efficiently solvable by a
QVA, it must satisfy three conditions:

#. The number of solutions :math:`{\left|{\mathcal{S}}\right|}` must be
   efficiently computable in order to establish a bound on the size of
   the required Hilbert space :math:`\mathcal{H}`.

#. For all solutions :math:`s`, :math:`{C(s)}` must be computable in
   polynomial time .

#. For all solutions :math:`s`, :math:`{C(s)}` must be polynomially
   bounded with respect to :math:`{\left|{\mathcal{S}}\right|}`.

Conditions one and two ensure that the objective function
:math:numref:`eq:objective_generic` is efficiently
computable as classical computation of :math:`{C(s)}` is required to
compute :math:`f` and boundedness in :math:`{C(s)}` ensures that
the number of measurements required to accurately compute
:math:`f` does not grow exponentially with
:math:`{\left|{\mathcal{S}^\prime}\right|}`
:cite:p:`crescenzi_structure_1999`. These conditions constrain
the application of QVAs to polynomially bounded (PB) COPs in the
non-deterministic polynomial-time optimisation problem (NPO) complexity
class (together denoted as NPO-PB)
:cite:p:`crescenzi_structure_1999`.

Unconstrained Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^

For the case of unconstrained optimisation, the valid solution space
:math:`{\mathcal{S}^\prime}` is equivalent to :math:`{\mathcal{S}}`. For
these COPs a quantum encoding of :math:`{C(s)}` is equivalent to the
bijective map :math:`{\mathcal{S}}\rightarrow \mathcal{H}`.

.. _QAOA:

QAOA
~~~~

The Quantum Approximate Optimisation Algorithm is comprised of two
alternating unitaries. Firstly the phase-shift-unitary

.. math::
   :label: eq:phase_shift_qaoa

   {\hat{U}_{\text{Q}}}(\gamma_i) = \exp(-\text{i} \gamma_i \hat{Q})

and, secondly, the mixing operator

.. math::
   :label: eq:mixing_qaoa

   {\hat{U}_{\text{X}}}(t_i) = \exp(- \text{i} t_i \hat{W}_\text{X}),

where :math:`\hat{W}_\text{X} = X^{\otimes N}` and :math:`X` is the
Pauli-\ :math:`X` (or quantum NOT) gate. The mixing operator
:math:`\hat{W}_\text{X}` induces a coupling topology that is equivalent
to an :math:`n`-dimension hypercube graph, as shown in
Fig. :ref:`hypercube-mixer <hypercube>`.

.. figure:: _static/hypercube.png
   :name: hypercube
   :scale: 25%
   :align: center

   Coupling topology of :math:`W_\text{X}` in the QAOA for
   :math:`{\left|{\mathcal{S}}\right|} = 16` (:math:`n` = 4).

The initial state :math:`{{\lvert\psi_0\rangle}_\text{QAOA}}` is
prepared as an equal superposition over :math:`\mathcal{H}`,

.. math::
   :label: eq:qaoa_initial_state

   {\lvert+\rangle} = \frac{1}{\sqrt{n}}\sum_{i = 0}^{n-1}{\lvert i\rangle}.

The final quantum state is then

.. math::
   :label: eq:qaoa

   {{\lvert\bm{\theta}\rangle}_\text{QAOA}} = \left( \prod_{i=1}^{p} {\hat{U}_{\text{X}}}(t_i) {\hat{U}_{\text{Q}}}(\gamma_i)  \right) {\lvert+\rangle},

where :math:`\bm{\theta}= \{\gamma_i, t_i \}` and
:math:`{\left|\bm{\theta}\right|} = 2p`
:cite:p:`farhi_quantum_2014`.

.. _Ex-QAOA:

Extended-QAOA
~~~~~~~~~~~~~

A variation of the QAOA, ‘extended-QAOA’ (ex-QAOA), utilises a sequence
of phase-shift unitaries,

.. math::
   :label: eq:phase_shift_qaoa_ex

   {\hat{U}_{\text{Qext}}}(\gamma_{i}) = \prod_{j=1}^{|\Sigma|} \exp(-\text{i} (\gamma_{i})_{j} \Sigma_{j}),

where :math:`\Sigma_j` are non-identity terms in a Pauli-gate
decomposition of :math:`\hat{Q}` and :math:`{\left|\Sigma\right|}` is
the number of non-identity terms
:cite:p:`guerreschi_practical_2017`. This increases the number
of variational parameters to
:math:`{\left|\bm{\theta}\right|} = (1 + {\left|\Sigma\right|})p` with
the intent of achieving a higher degree of convergence to optimal
solutions at a lower circuit depth.

The final state of ex-QAOA is

.. math::
   :label: eq:qaoa_ex

   {{\lvert\bm{\theta}\rangle}_\text{ex-QAOA}}= \left( \prod_{i=1}^{p}   \hat{U}_\text{X}(t_i) \hat{U}_\text{Qext}(\gamma_{i,:}) \right) {\lvert+\rangle},

where :math:`{\lvert+\rangle}` and :math:`\hat{U}_X` are defined as in
:math:numref:`eq:mixing_qaoa` and :math:numref:`eq:qaoa_initial_state`, 
and :math:`\bm{\theta}= {\left\{\gamma_{ij}, t_i\right\}}`.

Constrained Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^

Constrained optimisation problems seek valid solutions
:math:`{s^\prime}` from a subset of :math:`{\mathcal{S}}` as defined by
constraints :math:`{\bm{\chi}}`. Encoding of the solution constraints
:math:`{\bm{\chi}}` is achieved by restricting the action of the
mixing-unitaries :math:`{\hat{U}_{\text{mix}}}` and initialising
:math:`{{\lvert\psi_0\rangle}}` over a subspace of
:math:`\mathcal{H}`.

.. _QAOAz:

QAOAz
~~~~~

The Quantum Alternating Operator Ansatz was developed to solve problems
for which :math:`{\bm{\chi}}` creates a correspondence between
:math:`{\mathcal{S}^\prime}` and quantum states of equal parity – states
with the same number of :math:`{\lvert 1\rangle}` states. This algorithm
consists of the phase-shift-unitary defined in
:math:numref:`eq:phase_shift_qaoa`, followed by a sequence
of three :math:`{\hat{U}_{\text{mix}}}` with mixing operators

.. math::
   :label: eq:parity_terms

   \begin{aligned}
       & \hat{B}_\text{odd} = \sum_{a \, \text{odd}}^{N-1} X_{a}X_{a+1} + Y_{a}Y_{a+1} \\
       & \hat{B}_{\text{even}} = \sum_{a \, \text{even}}^{N}X_aX_{a+1} + Y_aY_{a+1} \\
       & \hat{B}_\text{last} = 
       \begin{cases}
       X_NX_1 + Y_NY_1, \, \text{odd} \\
       I, \, N \text{even},
       \end{cases}
   \end{aligned}

which together form the parity-conserving mixing operator

.. math::
   :label: eq:qaoaz_mixers

   {\hat{U}_{\text{parity}}}(t) = e^{-\text{i} t \hat{B}_\text{last}} e^{-\text{i} t \hat{B}_\text{even}} e^{-\text{i} t \hat{B}_\text{odd}}

that mixes probability amplitude between subgraphs of equal parity as
illustrated in Fig. :ref:`parity-mixer <parity-mixer>`.

By initialising :math:`{{\lvert\psi_0\rangle}_\text{QAOAz}}` in a
quantum state that satisfies the parity constraint, probability
amplitude is constrained to :math:`{\mathcal{S}^\prime}`.

The state evolution of the QAOAz is

.. math::
   :label: eq:qaoaz

   {{\lvert\bm{\theta}\rangle}_\text{QAOAz}} = \left( \prod_{i=1}^{p}{\hat{U}_{\text{parity}}}(t_i){\hat{U}_{\text{Q}}}(\gamma_i) \right){{\lvert\psi_0\rangle}_\text{QAOAz}},

where :math:`{{\lvert\psi_0\rangle}_\text{QAOAz}}` is an initial state
satisfying the parity constraint
:cite:p:`hadfield_quantum_2019`.

.. figure:: _static/parity.png
      :name: parity-mixer
      :scale: 25%
      :align: center

      Coupling topology of :math:`\hat{W}` for the QAOAz
      (:math:`n = 4`). Note that :math:`\mathcal{H}` is partitioned into
      subgraphs of equal state parity.

.. _QWOA:

QWOA
~~~~

The Quantum Walk-assisted Optimisation Algorithm implements
:math:`{\bm{\chi}}` given the existence of an efficient indexing
algorithm for all :math:`s\in\ {\mathcal{S}^\prime}`. Under this
condition, the QWOA implements an indexing unitary

.. math::
   :label: eq:qwoa_index

   {U^{\dag}_{\#}}{\lvert i\rangle} = 
   \begin{cases}
    {\lvert\text{id}_{{\bm{\chi}}}(i)\rangle}, \; {\lvert i\rangle} \in {\lvert{s^\prime}\rangle} \\
    {\lvert i\rangle}, \; \text{otherwise},
   \end{cases}

where :math:`{U^{\dag}_{\#}}` maps states corresponding to valid
solutions :math:`{\lvert{s^\prime}\rangle}` to indexed states
:math:`{\lvert\text{id}_{{\bm{\chi}}}(i)\rangle}`. By preparing
:math:`{{\lvert\psi_0\rangle}_\text{QWOA}}` as an equal superposition
over :math:`{\lvert\text{id}_{{\bm{\chi}}}(i)\rangle}`

.. math::
   :label: eq:qwoa_initial_state

   {{\lvert\psi_0\rangle}_\text{QWOA}} = \frac{1}{\sqrt{\left|{\mathcal{S}^\prime}\right|}}\sum\limits_{i \in {\mathcal{S}^\prime}} {\lvert i\rangle},

probability amplitude is restricted to the subspace of indexed states.

The indexing unitary :math:`{U^{\dag}_{\#}}` and its conjugate
unindexing unitary :math:`\hat{U}_\#` occur either side of a
mixing-unitary that acts on
:math:`{\lvert\text{id}_{{\bm{\chi}}}(i)\rangle}`:

.. math::
   :label: eq:qwoa_mixer

   {\hat{U}_{\text{index}}}(t) = \hat{U}_\# \exp(-i t \hat{W}_\text{QWOA}) \hat{U}_\#^\dag

Where efficiency in the implementation of :math:`{U^{\dag}_{\#}}`
dictates that :math:`\hat{W}_\text{QWOA}` is circulant. Most commonly,
:math:`\hat{W}_\text{QWOA}` is chosen to be the adjacency matrix of the
complete graph as it produces a maximal and unbiased coupling over
:math:`{\lvert{\mathcal{S}^\prime}\rangle}` (see Fig. 
:ref:`complete-mixer <complete>`).

The state evolution of the QWOA is

.. math::
   :label: eq:qwoa

   {{\lvert\bm{\theta}\rangle}_\text{QWOA}} = \prod_{i = 1}^{p}{\hat{U}_{\text{index}}}(t_i){\hat{U}_{\text{Q}}}(\gamma_i){{\lvert\psi_0\rangle}_\text{QWOA}},

where :math:`\bm{\theta}= {\left\{\gamma_i, t_i\right\}}` and there are
:math:`{\left|\bm{\theta}\right|} = 2p` variational parameters
:cite:p:`marsh_combinatorial_2019, marsh_combinatorial_2020`.

.. figure:: _static/complete.png
   :name: complete
   :align: center
   :scale: 25%

   A complete graph mixer, in QWOA with couples all valid solutions in the solution space.