import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qwoa
from quop_mpi import observable
import pandas as pd

qualities_df = pd.read_csv("qwoa_qualities.csv")
qualities = qualities_df.values[:, 1]

system_size = len(qualities)

alg = qwoa(system_size)

alg.set_qualities(observable.array, {"array": qualities})

alg.set_log("qwoa_portfolio_log", "qwoa", action="w")

alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="qwoa_portfolio", save_action="w"
)
