Portfolio Rebalancing using the QWOA and QAOAz
==============================================

This example makes use of pandas-datareader which requires an internet connection.

QWOA
----

Compute the solution qualities and write them to a CSV file:

> python3 qwoa\_qualities.py

Run the simulation:

> mpiexec -N 2 python3 qwoa\_portfolio.py

QAOAz
-----

Run the simulation:

> mpiexec -N 2 python3 qaoaz\_portfolio.py

portfolio\_plots.py
-------------------

Plot simulation results:

> python3 portfolio\_plots.py
