The Max-Cut Problem (QAOA)
==========================

Find a solution to the max-cut problem using QAOA.

maxcut.py
---------

Max-cut qualities passed via a serial function.

> mpiexec -N 2 python3 maxcut.py

maxcut\_parallel\_qualities.py
------------------------------

Max-cut qualities passed via a custom parallel function.

> mpiexec -N 2 python3 maxcut\_parallel\_qualities.py

maxcut\_plots.py
----------------

Plot the simulation results.

> python3 maxcut\_plots.py
