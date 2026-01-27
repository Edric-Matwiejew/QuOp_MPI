Welcome to QuOp_MPI's documentation!
====================================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :alt: Python 3.11+

.. image:: https://img.shields.io/github/license/Edric-Matwiejew/QuOp_MPI.svg
   :target: https://github.com/Edric-Matwiejew/QuOp_MPI/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/GitHub-Repository-blue?logo=github
   :target: https://github.com/Edric-Matwiejew/QuOp_MPI
   :alt: GitHub Repository

.. image:: https://zenodo.org/badge/233372703.svg
   :target: https://zenodo.org/badge/latestdoi/233372703
   :alt: DOI

**QuOp_MPI** is a Python framework for parallel simulation of quantum variational algorithms using MPI.

Quick Start
-----------

Here's a minimal example using QAOA to solve a MaxCut problem:

.. code-block:: python

   from mpi4py import MPI
   import numpy as np
   import networkx as nx
   from quop_mpi.algorithm.combinatorial import qaoa
   from quop_mpi.toolkit import I, Z

   # Create a graph
   G = nx.complete_graph(4)

   # Define the MaxCut cost function
   n_qubits = G.number_of_nodes()
   
   def maxcut_qualities(system_size, local_i, local_i_offset):
       C = sum(0.5 * (I(n_qubits) - Z(i, n_qubits) @ Z(j, n_qubits)) 
               for i, j in G.edges())
       return -C.diagonal()[local_i_offset:local_i_offset + local_i].real

   # Set up and run QAOA
   alg = qaoa(system_size=2**n_qubits, MPI_COMM=MPI.COMM_WORLD)
   alg.set_qualities(maxcut_qualities)
   alg.execute()
   alg.print_result()

Run with MPI:

.. code-block:: bash

   mpiexec -n 4 python maxcut_example.py

For more examples, see the :doc:`examples` page.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Getting Started <readme_link>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   theoretical_background
   package_overview
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   quop_functions
   developing_for_quop

.. toctree::
   :maxdepth: 1
   :caption: About

   cite
   changelog
   glossary
   bibliography

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
