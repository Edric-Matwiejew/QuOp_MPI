from quop_mpi.algorithm.combinatorial import qwoa, csv

system_size = 31
alg = qwoa(system_size)

alg.set_qualities(
    csv, 
    {
        "args": ["qwoa_qualities.csv"],
        "kwargs": {'usecols':[1], 'header':None}
    }
    )

alg.set_log("qwoa_portfolio_log", "qwoa", action="w")
alg.benchmark(
    range(1, 6),
    3,
    param_persist=True,
    filename="qwoa_portfolio",
    save_action="w"
)
