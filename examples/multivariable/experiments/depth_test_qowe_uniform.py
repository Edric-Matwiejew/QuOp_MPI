import sys

sys.path.append("../../")
import os
from pathlib import Path
from time import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from quop_mpi.algorithm import qowe
from quop_mpi.__lib import fCQAOA
from quop_mpi.state import equal
import test_function
from mpi4py import MPI
from scipy.optimize import Bounds

nn = int(os.getenv("N"))
time_limit = int(os.getenv("TIMELIMITSECONDS"))
repeats = int(os.getenv("REPEATS"))
pmax = int(os.getenv("PMAX"))
maxiter = int(os.getenv("MAXITER"))
d = int(os.getenv("D"))

functions = [test_function.styblinski_tang, test_function.rastrigin]

output = "results/depth_test_qowe_uniform"

for function in functions:

    basename = f"{output}/{function.name} d={d} n={nn}"
    suspend_path = f"suspend_data/depth_test_qowe_uniform/{function.name} d={d} n={nn}"
    
    
    if MPI.COMM_WORLD.Get_rank() == 0:
    
        Path(output).mkdir(parents=True, exist_ok=True)
    
    n = 2 ** nn  # number of grid point per dimension
    
    L = np.diff(function.search_domain(2)[0])[0] / 2
    
    dq = 2 * L / n  # position space grid spacing
    Ns = d * [n]  # shape of d-dimensional grid
    
    # grid spacing in each coordinate
    deltas = np.array(d * [dq], dtype=np.float64)
    # minimum value in each coordinate
    mins = np.array(d * [-(n / 2) * dq], dtype=np.float64)
    
    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]
    
    alg = qowe(Ns, deltas, mins)
    alg.set_qualities(function)
    alg.set_initial_state(equal)
    
    bounds = Bounds(-2 * np.pi, 2 * np.pi)
    alg.set_optimiser(
        "scipy",
        {
            "method": "Nelder-Mead",
            "options": {"maxiter": maxiter, "adaptive": True},
            "bounds": bounds,
        },
        ["fun", "nfev", "success"],
    )
    
    alg.set_log(f"{basename}", f"{function.name}", action="w")
    
    start_time = time()
    
    alg.benchmark(
        range(1, pmax + 1),
        repeats,
        param_persist=True,
        filename=f"{basename}",
        label=f"d={d} n={nn}",
        save_action="w",
        verbose=True,
    )
    
    end_time = time()
    
    time_limit -= end_time - start_time
