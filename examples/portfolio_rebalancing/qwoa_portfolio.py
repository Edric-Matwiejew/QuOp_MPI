from quop_mpi.algorithm.combinatorial import QWOA, csv

system_size = 30
alg = QWOA(system_size)

alg.set_qualities(csv, {"args": ["qwoa_qualities.csv"], "kwargs": {"usecols": [1], "header": None}})

alg.set_log("qwoa_portfolio_log", "qwoa", action="w")
alg.verbose_objective = True
alg.benchmark(range(1, 6), 5, param_persist=True, filename="qwoa_portfolio", save_action="w")
