import sys

sys.path.append("../../")
import os
from time import time
from pathlib import Path
from mpi4py import MPI
import numpy as np
import test_function
from quop_mpi.algorithm import qmoa
from quop_mpi.__lib import fCQAOA

nn = int(os.getenv("N"))
time_limit = int(os.getenv("TIMELIMITSECONDS"))
repeats = int(os.getenv("REPEATS"))
pmax = int(os.getenv("PMAX"))
maxiter = int(os.getenv("MAXITER"))
d = int(os.getenv("D"))

functions = [test_function.rastrigin]
    
output = f"results/n_scaling_qmoa_complete"

for function in functions:

    basename = f"{output}/{function.name} d={d} n={nn}"
    suspend_path = "suspend_data/n_scaling_qmoa_complete"
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(suspend_path).mkdir(parents = True, exist_ok = True)
    
    n = 2 ** nn  # number of grid points per dimension
    
    L = np.diff(function.search_domain(2)[0])[0] / 2
    
    dq = 2 * L / n  # position space grid spacing
    Ns = d * [n]  # shape of d-dimensional grid
    deltas = np.array(d * [dq], dtype=np.float64)
    mins = np.array(d * [-(n / 2) * dq], dtype=np.float64)
    
    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]
    
    def cost_function(local_i, local_i_offset, MPI_COMM, function = None):
    
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
    
    alg = qmoa(Ns, deltas, mins)
    alg.UQ.operator_function = cost_function
    alg.UQ.operator_kwargs = {"function": function}
    alg.set_observables(0)
    
    alg.set_optimiser(
        "scipy",
        {"method": "Nelder-Mead", "options": {"maxiter": maxiter, "adaptive": True}},
        ["fun", "nfev", "success"],
    )
    
    alg.set_log(f"{basename}", f"d={d} n={n}", action="w")
    
    alg.verbose_objective = True 
    start_time = time()
    
    alg.benchmark(
        range(1, pmax + 1),
        repeats,
        param_persist=True,
        filename=f"{basename}",
        label=f"d={d} n={n}",
        save_action="w",
        verbose=True,
        time_limit = time_limit,
        suspend_path = f"{suspend_path}/{function.name} d={d} n={nn}",
    )
    
    end_time = time()
    
    time_limit -= end_time - start_time
