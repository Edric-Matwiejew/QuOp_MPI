Multivariable Optimisation
--------------------------

The Continuous Multivariable Optimisation Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a continuous function :math:`f: X \rightarrow Y`, where
:math:`X \subset \mathbb{R}^D` and :math:`Y \subset \mathbb{R}`,
continuous-variable optimisation seeks
:math:`{\bm{x}^*} = (x_0, ..., x_{D-1}) \in X` satisfying,

.. math::
   :label: optimisation

   f({\bm{x}^*}) \leq f_\text{min} + \epsilon,

where :math:`f_\text{min}` is the global minimum of :math:`f` and
:math:`\epsilon > 0` defines a region of accepted :math:`\bm{x}` near
:math:`f_\text{min}`.

An encoding of the optimisation problem in a system of qubits consists
of evaluating :math:`f` on a grid of :math:`K = N^D` points. In each
dimension :math:`d`, the coordinate is discretised as
:math:`x_{d,n_d} = x_{d,0} + n_d \Delta x_d`, with minimum value
:math:`x_{d,0}`, grid spacing :math:`\Delta x_d`, and
:math:`n_d = 0, \dots, N-1`. The complete solution space of discretised
coordinates :math:`\bm{x}_k` is then represented using
:math:`\mathcal{O}(\log K)` qubits by states
:math:`\ket{k} \equiv \ket{x_{D-1,n_{D-1}},x_{D-2,n_{D-2}}, \dots, x_{0,n_0}}`,
where :math:`k = 0, \dots, N^D-1` is a vectorised index for the set
:math:`(n_0, \dots, n_{D-2}, n_{D-1}) \in \{ 0, \dots, N-1 \}^D`. For
optimisation over this discrete space, we denote the global minimum as
:math:`{\bm{x}^*} \equiv \text{argmin}_{k} f(\bm{x}_k)`.

QVAs for Optimisation of Continuous Multivariable Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we consider QVAs of the form

.. math::
   :label: variational_generic

   \ket{\bm{t},\bm{\gamma}}=\left( \prod_{i = 1}^{p}\hat{U}(t_i,\gamma_i) \right)\ket{\psi_0},

where the positive integer :math:`p` is a fixed number of ansatz
iterations, :math:`\hat{U}` is the ansatz unitary, :math:`t_i` and
:math:`\gamma_i` are real-valued variational parameters and,

.. math::
   :label: equal_superposition

   \ket{\psi_0} = \frac{1}{\sqrt{K}}\sum_{k = 0}^{K-1}\ket{k},

unless otherwise specified. The ansatz unitary consists of the so-called
alternating phase-shift, :math:`\hat{U}_Q`, and mixing,
:math:`\hat{U}_W`, unitaries,

.. math:: \hat{U}(t,\gamma)=\hat{U}_W(t)\hat{U}_Q(\gamma).

The first of these,

.. math::
   :label: phase-shift

   \hat{U}_Q(\gamma) = \exp(-\text{i} \gamma \hat{Q}),

applies a phase-shift proportional to

.. math:: \hat{Q} = \sum_{k=0}^{K - 1} f_k \ket{k}\bra{k},

where :math:`f_k \equiv f(\bm{x}_k)`. The second unitary,
:math:`\hat{U}_W`, conforms to some structure specific to each
algorithm. Its role is to drive the transfer of probability amplitudes
between the solution states. During the mixing stage, phase differences
encoded by :math:`\hat{U}_Q` result in interference that is manipulated
by varying :math:`(\bm{t}, \bm{\gamma})`.

A QVA then proceeds by repeated preparation of
:math:`\ket{\bm{t}, \bm{\gamma}}`. After each iteration,
:math:`(\bm{t}, \bm{\gamma})` are tuned using a classical optimisation
algorithm to minimise the expectation value

.. math:: \langle Q \rangle = \bra{\bm{t}, \bm{\gamma}} \,\hat{Q}\, \ket{\bm{t}, \bm{\gamma}}.

The intended consequence is an increased probability of measuring
solutions satisfying :math:numref:`optimisation`. The
possible amplification increases with :math:`p` at the expense of a
deeper quantum circuit and larger parameter space for the classical
optimiser.

.. _QAOA-multivariable:

QAOA
^^^^

The Quantum Approximate Optimisation Algorithm (QAOA) defines the mixing unitary as

.. math::
   :label: qaoa_mixer

   \hat{U}_{W\text{-QAOA}}(t) = \exp(-\text{i}t \hat{W}),

which is defined by the mixing operator

.. math::
   :label: adjacency

   \hat{W} = \sum_{k,k^\prime=0}^{K-1} w_{kk^\prime} \ket{k}\bra{k^\prime},

where typically :math:`w_{kk^\prime} \in \{0, 1\}`. This can be
interpreted as implementing a continuous-time quantum walk for time
:math:`t \geq 0` over an undirected graph of :math:`K` vertices with
adjacency matrix :math:`w_{kk^\prime}`, where :math:`w_{kk^\prime} = 1`
if vertices :math:`k` and :math:`k^\prime` are connected and
:math:`k \neq k^\prime` :cite:p:`hadfield_quantum_2019,marsh_quantum_2019`.
For a complete graph :math:`\hat{W}`, one can write

.. math::
   :label: qaoa_c_op

   \hat{U}_{W\text{-QAOA}}(t) = e^{it} \left[ \hat{I} + ( e^{-itK} - 1 ) \frac{1}{K} \sum_{k,k'=0}^{K-1} \ket{k} \bra{k'} \right].

The action of a single iteration of
:math:`\hat{U}_\text{QAOA}(t, \gamma) = \hat{U}_{W\text{-QAOA}}(t)\hat{U}_Q(\gamma)`
then maps the amplitudes of an arbitrary state
:math:`\sum_k \alpha_k \ket{k}` (up to a global phase :math:`e^{it}`) as

.. math::
   :label: qaoa_c_amplitude

   \begin{aligned}
     \alpha_k \hspace{1mm} \mapsto \hspace{1mm} e^{-i \gamma f_k} \alpha_k + ( e^{-i K t} - 1 ) \left( \frac{1}{K} \sum_{k'=0}^{K-1} e^{-i \gamma f_{k'}} \alpha_{k'} \right).
   \end{aligned}

We see that the second term averages the amplitudes over the entire
solution space and is the same for all :math:`k`. Amplification of a
particular coefficient :math:`\alpha_k` then depends on how this local
information compares with the global average. This is a useful property
in the absence of an identified solution space structure, since
:math:`k` is distinguished solely by the locally phase-encoded
:math:`f_k` :cite:p:`slate_quantum_2021,bennett_quantum_2022`.
Notice that the unbiased coupling in :math:numref:`qaoa_c_op`
means that amplitudes at any two points :math:`k`, :math:`k'` with
:math:`f_k \approx f_{k'}` evolve similarly under
:math:`\hat{U}_\text{QAOA}(t, \gamma)`, and will also respond similarly
to variation in :math:`t` and :math:`\gamma`. This is a potential
disadvantage in the context of CMOPs since contours in :math:`f` result
in many degenerate :math:`f_k`. Highly degenerate solutions will greatly
influence the sum in :math:numref:`qaoa_c_amplitude`,
and thus are likely to dominate the optimisation process.

The QAOA was originally defined with the :math:`\hat{W}` structured
according to a hypercube graph, as a hypercube on :math:`M` qubits is
easily implemented as :math:`\sum_ {i=0}^{M-1}\hat{X}^{(i)}`, where
superscript :math:`(i)` denotes action on qubit
:math:`i` :cite:p:`farhi_quantum_2014`. For a hypercube graph
:math:`\hat{W}`, the QAOA mixing unitary can be written as:

.. math::
   :label: qaoa_h_op

   \begin{aligned}
     \hat{U}_{W\text{-QAOA}}(t) \nonumber = \sum_{w=0}^M (\cos t)^{M-w} (-i \sin t)^w \sum_{k=0}^{K-1} \sum_{b \in \mathcal{B}_w} \ket{k} \bra{k \oplus b},
   \end{aligned}

where :math:`\mathcal{B}_w` is the set of bit strings of Hamming weight
:math:`w`, and :math:`k \oplus b` denotes bitwise XOR between the binary
representation of :math:`k` and :math:`b`. As opposed to
:math:numref:`qaoa_c_op`, the hypercube mixer couples points
differently according to their respective Hamming distance. Thus, even
if there are many points with similar :math:`f_k` values, amplitudes at
such points should only respond similarly to variations in :math:`t` and
:math:`\gamma` when averages of phase-shifted amplitudes at a fixed
Hamming distance away are the same. Given a hypercube embedding of the
solution space grid, this is likely to occur primarily when :math:`f`
has particular structural properties, such as rotational symmetry or
periodicity.

In the context of a quantum search over the discretised solution space
of a CMOP, the hypercube has the desirable property of (at least
approximate) preservation of the solution space structure, as grids in
one, two, and three dimensions can be embedded in a
hypercube :cite:p:`ostrouchov_parallel_1987`. Examples of the
grid embedding induced by :math:`\hat{U}_{W\text{-QAOA}}` are shown in
Fig. :ref:`hypercube-mixing-structure <hypercube-mixing-structure>`. Also, a
hypercube graph has a diameter of :math:`M` and :math:`M` disjoint paths
between any two vertices :cite:p:`ostrouchov_parallel_1987`,
so the distance between any two :math:`\bm{x}_k` is exponentially
smaller than :math:`K`.

.. _hypercube-mixing-structure:

.. list-table::

   * - .. figure:: _static/Hypercube_2D.png
     - .. figure:: _static/Hypercube_3D.png

Examples of the coupling produced on :math:`\bm{x}_k` by :math:`\hat{U}_{W\text{-QAOA}}` with a hypercube :math:`\hat{W}` are shown for a solution space of size :math:`K=8` in :math:`D=2` and :math:`D=3`, respectively. The dashed red line indicates the grid embedding, which in this case is approximate for :math:`D=2` and exact for :math:`D=3`.

.. _QOWE:

QOWE
^^^^

The Quantum Optimisation with Wavepacket Evolution (QOWE) algorithm is based on the approach to continuous-variable optimisation described in 
:cite:p:`verdon_quantum_2019` which, using continuous quantum variables, consists of the propagation of an initial
Gaussian wavepacket under a phase-shift followed by the mixing unitary

.. math::
   :label: cts_mixer

   \hat{U}(t) = \prod_{d=0}^{D-1} e^{-i t \hat{p}_d^2},

where :math:`\hat{p}_d` is the momentum operator conjugate to the
continuous coordinate :math:`\hat{x}_d`. This choice is inspired by
considering the quantum simulation of a particle evolving under the
potential :math:`f(\bm{x})`.

Here we examine a discretised form of this algorithm, with the problem
solution space encoded in :math:`\ket{k}`. The state is initialised to a
discretised Gaussian wavepacket,

.. math::
   :label: wave-packet

   \ket{\psi_0}=\frac{1}{\sqrt{A}}\sum_{k=0}^{K-1}\prod_{d=0}^{D-1} e^{-\frac{(x_{k}^{(d)} - \mu_d)^2}{2{\sigma_d}^2}}\ket{k}

where :math:`x_{k}^{(d)}` is the :math:`d^{th}` component of
:math:`\bm{x}_k`, :math:`\mu_d` and :math:`\sigma_d` are the centre and
width of the wavepacket in each dimension, and :math:`A` is a
normalising constant. Discretising the mixing unitary requires a
discrete form of the continuous momentum operator. For our
implementation of QOWE, we construct a discrete analogue of the
continuous-variable relationship
:math:`\hat{p}_d = \mathcal{F}^{-1} \hat{x}_d \mathcal{F}` (in each
dimension), where :math:`\mathcal{F}` is the continuous Fourier
transform. The continuous Fourier transform along a single dimension can
be approximated on the discretised grid as

.. math:: \mathcal{F} \approx F_d := \sum_{n_d = 0}^{N-1} e^{-i x_{d,0} \kappa_{d,n_d}} \ket{n_d} \bra{n_d} \text{DFT},

where :math:`\text{DFT}` is the centred discrete Fourier transform, and
:math:`\kappa_{d,n_d} = \kappa_{d,0} + n_d \Delta \kappa_d` is a
momentum-space grid point, with
:math:`\Delta \kappa_d = \frac{2\pi}{N \Delta x_d}`,
:math:`\kappa_{d,0} = \Delta \kappa_d ( -N + 1 + \lfloor \frac{N-1}{2} \rfloor )`,
and :math:`n_d = 0, \dots, N-1`. The corresponding Fourier transform
over the entire discretised solution space is then
:math:`F := \otimes_{d=0}^{D-1} F_d`, and the mixing unitary is

.. math::
   :label: gaussian_mixer

   \hat{U}_{\vert\kappa_{k}\vert^2}(t) = F^{-1}e^{-\text{i}t \hat{W}_{\kappa}} F

where :math:`\hat{W}_{\kappa}` is the diagonal operator,

.. math::
   :label: momentum_mixer

   \hat{W}_{\kappa}=\sum_{k = 0}^{K-1} \vert\bm{\kappa}_{k}\vert^2 \ket{k}\bra{k},

and where
:math:`\bm{\kappa}_k = (\kappa_{0,n_0}, \dots, \kappa_{D-1,n_{D-1}})` is
a momentum space grid point with a similar indexing to :math:`\bm{x}_k`.

Applying the phase-shift unitary followed by the first Fourier transform
in :math:numref:`gaussian_mixer` and computational basis
measurement is related to Jordan’s algorithm for gradient
computation :cite:p:`jordan_fast_2005`. Here, the gradient
information is used coherently by following the first Fourier transform
by the remaining two unitaries
in :math:numref:`gaussian_mixer`, instead of performing
a measurement.

.. _QMOA:

QMOA
^^^^

The Quantum Multivariable Optimisation Algorithm (QMOA) mixer is taken to be a unitary of separable CTQWs,

.. math::
   :label: nd_walk

   \hat{U}_{W\text{-QMOA}}(\bm{t}) = \prod_{d=0}^{D-1}\exp(-\text{i}t_d \hat{C}_d),

where :math:`\bm{t} = (t_0,...t_{D-1})` with :math:`t_d \geq 0` and
:math:`\hat{C}_d` is the adjacency matrix of an undirected graph (see
:math:numref:`adjacency`) connecting vertices along the
dimension :math:`d`. The discretisation of the QOWE mixer is of a
similar form if the generator of
:math:numref:`gaussian_mixer` is interpreted as a
composite of complete graphs with complex-valued :math:`w_{kk'}`. In
QMOA, we only consider cases where :math:`w_{kk'} \in \{ 0, 1 \}`. With
:math:`\hat{C}_d` as a cycle graph, :math:`\hat{W}` is equivalent to a
finite difference approximation of the Laplacian (i.e., a different
discretisation of :math:numref:`cts_mixer`). However, we
consider more general graphs that do not correspond to different
discretisations of :math:numref:`cts_mixer`, but do separate
into independent quantum walks in each dimension. The case where each
:math:`\hat{C}_d` is a complete graph is depicted in
Fig. :ref:`QMOA-mixing-structure <QMOA-mixing-structure>` for a
two-dimensional :math:`4 \times 4` grid. 

Under the condition that :math:`\hat{C}_d` is circulant, and therefore
diagonalised by the coefficient matrix of the discrete Fourier
transform, :math:numref:`nd_walk` is efficiently realisable as

.. math:: (\text{DFT}^{-1})^{\otimes D} \exp\left(- \rm{i} \hat{\Lambda}(\bm{t}) \right) \text{DFT}^{\otimes D},

where :math:`\text{DFT}` denotes the discrete Fourier transform and,

.. math::
   :label: circulant_mixer_multi

   \hat{\Lambda}(\bm{t}) = \sum_{d = 0}^{D-1}t_{d}\sum_{n_d=0}^{N - 1}\Lambda_{d,n_d}\ket{x_{d,n_d}}\bra{x_{d,n_d}},

is constructed using the closed form solution for the eigenvalues
:math:`\Lambda_{d,n_d}` of :math:`\hat{C}_d`. We note that of the graphs
introduced for the QAOA, the complete graph is circulant, while the
hypercube graph is non-circulant. Altogether, the QMOA ansatz unitary
:math:`\hat{U}_\text{QMOA}` has a
:math:`\mathcal{O}\left(\text{polylog} \, K\right)` gate complexity
resulting from :math:`D` instances of the quantum Fourier transform
:cite:p:`hales_improved_2000`.

For complete graphs :math:`\hat{C}_d`, the mixing unitary can be written
as,

.. math::

   \begin{aligned}
     \hat{U}_{W\text{-QMOA}}(\bm{t}) \nonumber 
     = \bigotimes_{d=0}^{D-1} e^{i t_d} \left[ \hat{I} + ( e^{-i t_d N} - 1 ) \frac{1}{N} \sum_{n_d, n_d' = 0}^{N-1} \ket{n_d} \bra{n_d'} \right].
   \end{aligned}

In each dimension, the operator :math:`\exp(-i t \hat{C}_d)` applies an
unbiased coupling between all points within each line parallel to
coordinate axis :math:`d`. The amplitude of a point then evolves
according to the average amplitude along the corresponding line,
analogous to :math:numref:`qaoa_c_amplitude`. Combining
the walks in each dimension, along with the phase-shift,
:math:`\hat{U}_\text{QMOA}(t, \gamma)=\hat{U}_{W\text{-QMOA}}(t)\hat{U}_Q(\gamma)`
causes the amplitude of a point to evolve according to the locally
phase-encoded :math:`f_k` relative to averages of phase-shifted
amplitudes in the various subspaces containing the point. For example,
in Fig. :ref:`QMOA-mixing-structure <QMOA-mixing-structure>` the coordinate
subspaces of :math:`\ket{k=13}` are the row of points containing
:math:`x_{0,3}` and the column of points containing :math:`x_{1,1}`. By
averaging phase-shifted amplitudes among these subspaces, rather than
simply over the entire solution space as in
:math:numref:`qaoa_c_op`, as well as using different walk
times :math:`t_d` in each dimension, :math:`\hat{U}_{W\text{-QWOA}}` can
break degeneracies resulting from contours in :math:`f` that are
non-parallel to the coordinate axes. More generally, the evolution of
amplitudes at any two :math:`\ket{k}` will respond similarly to
variations in :math:`t_d` and :math:`\gamma` only if there is similarity
in their locally encoded :math:`f_k` and the averages of the
phase-encoded :math:`f_k` in their respective subspaces, which is likely
to occur only when there is a high degree of symmetry in :math:`f`.
Furthermore, as the minima of a continuous :math:`f` are stationary
points, every line passing through (or near) a minimum will contain
multiple :math:`\bm{x}_k` with :math:`f_k` close to the minimum value
(provided the discretisation is sufficiently dense). Consequently, the
separable CTQWs have the potential to mutually re-enforce convergence to
subspaces that contain multiple high-quality solutions.

.. figure:: _static/QMOA_Diagram.png
   :name: QMOA-mixing-structure
   :align: center
   :scale: 25%

   Overview of the QMOA for an arbitrary :math:`f` in :math:`D=2` discretised over a grid of 16 points. The horizontal and vertical outlines denote the hyperplanes of constant coordinates. The bottom two graphs in (a) illustrate the coupling between these hyperplanes in :math:`\hat{U}_{W\text{-QMOA}}` with a complete graph in each dimension. 