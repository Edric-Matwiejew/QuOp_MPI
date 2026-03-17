Portfolio Rebalancing using the QWOA and QAOAz
==============================================

Force fallback CSV data (skip Yahoo Finance)
-------------------------------------------

Set `QUOP_PORTFOLIO_USE_SAMPLE_DATA=1` at runtime to force bundled CSV data:

> QUOP_PORTFOLIO_USE_SAMPLE_DATA=1 python3 qwoa_qualities.py
>
> QUOP_PORTFOLIO_USE_SAMPLE_DATA=1 mpiexec -N 2 python3 qaoaz_portfolio.py

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
