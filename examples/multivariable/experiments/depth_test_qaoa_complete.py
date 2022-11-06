import sys

sys.path.append("../../")
import os
from pathlib import Path
from time import time
import numpy as np
from mpi4py import MPI
from quop_mpi.algorithm import qwoa
from quop_mpi.__lib import fCQAOA
import test_function

nn = int(os.getenv("N"))
time_limit = int(os.getenv("TIMELIMITSECONDS"))
repeats = int(os.getenv("REPEATS"))
pmax = int(os.getenv("PMAX"))
maxiter = int(os.getenv("MAXITER"))
d = int(os.getenv("D"))

functions = [test_function.styblinski_tang, test_function.rastrigin]

size = MPI.COMM_WORLD.Get_size()

def params(p):
    return np.random.uniform(low=0, high=2 * np.pi, size=2 * p)

output = "results/depth_test_qaoa_complete"

for function in functions:

    basename = f"{output}/{function.name} d={d} n={nn}"
    suspend_path = f"suspend_data/depth_test_qaoa_complete/{function.name} d={d} n={nn}"
    
    if MPI.COMM_WORLD.Get_rank() == 0:
    
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(suspend_path).mkdir(parents = True, exist_ok = True)
    
    n = 2 ** nn  # number of grid point per dimension
    
    L = np.diff(function.search_domain(2)[0])[0] / 2
    dq = 2 * L / n  # position space grid spacing
    Ns = d * [n]  # shape of d-dimensional grid
    deltasq = np.array(d * [dq], dtype=np.float64)
    minsq = np.array(d * [-(n / 2) * dq], dtype=np.float64)
    
    n = 2 ** nn
    
    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]
    
    
    def cost_function(local_i, local_i_offset, MPI_COMM):
    
        x = fCQAOA.continuous.gen_local_grid(
            n ** d, Ns, strides, deltasq, minsq, local_i_offset, local_i
        )
    
        f = []
        for point in x:
            f.append(function(point))
    
        return np.array(f, dtype=np.float64)
    
    
    alg = qwoa(n ** d)
    
    alg.set_qualities(cost_function)
    
    alg.set_optimiser(
        "scipy",
        {"method": "Nelder-Mead", "options": {"maxiter": maxiter, "adaptive": True}},
        ["fun", "nfev", "success"],
    )
    
    alg.set_log(f"{basename}", f"d={d}, n={n}", action="w")
    
    start_time = time()
    
    alg.benchmark(
        range(1, pmax + 1),
        repeats,
        param_persist=True,
        filename=f"{basename}",
        label=f"d={d} n={nn}",
        save_action="w",
        verbose=True,
        time_limit = time_limit,
        suspend_path = suspend_path, 
    )
    
    end_time = time()
    
    time_limit -= end_time - start_time
