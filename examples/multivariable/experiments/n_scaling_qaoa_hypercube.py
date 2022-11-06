import sys

sys.path.append("../../")
import os
from pathlib import Path
from time import time
import numpy as np
import h5py
from mpi4py import MPI
from quop_mpi.algorithm import qaoa
from quop_mpi.__lib import fCQAOA
import test_function

nn = int(os.getenv("N"))
time_limit = int(os.getenv("TIMELIMITSECONDS"))
repeats = int(os.getenv("REPEATS"))
pmax = int(os.getenv("PMAX"))
maxiter = int(os.getenv("MAXITER"))
d = int(os.getenv("D"))

functions = [
        test_function.rastrigin
        ]

#test_function.styblinski_tang
size = MPI.COMM_WORLD.Get_size()

output = "results/n_scaling_qaoa_hypercube"

for function in functions:

    basename = f"{output}/{function.name} d={d} n={nn}"
    suspend_path = f"suspend_data/n_scaling_qaoa_hypercube"
    
    if MPI.COMM_WORLD.Get_rank() == 0:
    
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(suspend_path).mkdir(parents = True, exist_ok = True)
    
    n = 2 ** nn  # number of grid point per dimension
    
    L = np.diff(function.search_domain(2)[0])[0] / 2
    dq = 2 * L / n  # position space grid spacing
    Ns = d * [n]  # shape of d-dimensional grid
    deltas = np.array(d * [dq], dtype=np.float64)
    mins = np.array(d * [-(n / 2) * dq], dtype=np.float64)
    
    n = 2 ** nn
    
    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]
    
    
    def cost_function(local_i, local_i_offset, MPI_COMM):
    
        x = fCQAOA.continuous.gen_local_grid(   n**d,
                                                Ns,
                                                strides,
                                                deltas,
                                                mins,
                                                local_i_offset,
                                                local_i)
        f = []
        for point in x:
            f.append(function(point))
    
        return np.array(f, dtype = np.float64)
    
    alg = qaoa(n ** d)
    
    alg.set_qualities(cost_function)
    
    alg.set_optimiser(
        "scipy",
        {"method": "Nelder-Mead", "options": {"adaptive": True}},
        ["fun", "nfev", "success"],
    )
    
    alg.set_log(f"{basename}", f"d={d}, n={n}", action="w")
    alg.verbose_objective = True 
    start_time = time()
    alg.set_seed(14) 
    alg.benchmark(
        range(1, pmax + 1),
        repeats,
        param_persist=False,
        filename=f"{basename}",
        label=f"d={d} n={n}",
        save_action="w",
        verbose=True,
        time_limit = time_limit,
        suspend_path = f"{suspend_path}/{function.name} d={d} n={nn}",
    )
    
    end_time = time()
    
    time_limit -= end_time - start_time
