..
    cspell:words cartesian

########
quop_mpi
########
	
.. autoclass:: quop_mpi.Ansatz
    :exclude-members: setup, destroy
    :members:

.. autoclass:: quop_mpi.Unitary
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan
    :members:

algorithm
=========

combinatorial
-------------
.. automodule:: quop_mpi.algorithm.combinatorial
    :members:
    :exclude-members: serial, csv, hdf5, array, rand, setup, destroy
    :inherited-members:

multivariable
-------------
.. automodule:: quop_mpi.algorithm.multivariable
    :members:
    :exclude-members: cartesian, cartesian_scaled, setup_cartesian, setup, destroy
    :inherited-members:



observable
==========
.. automodule:: quop_mpi.observable
    :members:

.. automodule:: quop_mpi.observable.rand
    :members:

param
=====

.. automodule:: quop_mpi.param
    :members:

.. automodule:: quop_mpi.param.rand
    :members:

state
=====

.. automodule:: quop_mpi.state
    :members:

meta
====

.. autoclass:: quop_mpi.meta.swarm
    :members:

propagator
==========

.. automodule:: quop_mpi.propagator

circulant
---------
.. automodule:: quop_mpi.propagator.circulant
    :members:
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan, copy_plan, destroy, plan, copy_plan, propagate

operators
^^^^^^^^^
.. automodule:: quop_mpi.propagator.circulant.operator
    :members:

diagonal
--------
.. automodule:: quop_mpi.propagator.diagonal
    :members:
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan, copy_plan, destroy, plan, evolve_group, evolve_single

operators
^^^^^^^^^
.. automodule:: quop_mpi.propagator.diagonal.operator
    :members:

sparse
------
.. automodule:: quop_mpi.propagator.sparse
    :members:
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan, copy_plan, destroy, plan, evolve_group, evolve_single

operators
^^^^^^^^^
.. automodule:: quop_mpi.propagator.sparse.operator
    :members:

composite
---------
.. automodule:: quop_mpi.propagator.composite
    :members:
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan, copy_plan, destroy, plan, evolve_group, evolve_single

operators
^^^^^^^^^
.. automodule:: quop_mpi.propagator.composite.operator
    :members:

momentum
--------
.. automodule:: quop_mpi.propagator.momentum
    :members:
    :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function, parse_plan, copy_plan, destroy, plan, evolve_group, evolve_single

operators
^^^^^^^^^
.. automodule:: quop_mpi.propagator.momentum.operator
    :members:

toolkit
=======
.. automodule:: quop_mpi.toolkit
    :members:
