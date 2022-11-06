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
funcint = int(os.getenv("FUNCINT"))

if d == 2:
    function = test_function.functions[funcint]
else:
    function = test_function.functions_d3[funcint]
    
for c in [1, 2, 4, 8, 16]:

    output = f"results/circulant_mixers_non_independent_t/d={d} n={nn}/{c}"
    basename = f"{output}/{function.name} d={d} n={nn} c={c}"
    suspend_path = "suspend_data/circulant_mixers"
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(suspend_path).mkdir(parents = True, exist_ok = True)
    
    n = 2 ** nn  # number of grid point per dimension
    
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
    
        f = np.array(f, dtype = np.float64)
    
        f_max = MPI_COMM.allreduce(np.max(f), op = MPI.MAX) 
        f_min = MPI_COMM.allreduce(np.min(f), op = MPI.MIN) 
    
    
        return 100*(f - f_min)/(f_max - f_min)
   
    alg = qmoa(Ns, deltas, mins)
    alg.set_mixer(d * [c])
    alg.UQ.operator_function = cost_function
    alg.UQ.operator_kwargs = {"function": function}
    alg.set_observables(0)
      
    alg.set_independent_t(False) 

    alg.set_optimiser(
        "scipy",
        {"method": "Nelder-Mead", "options": {"maxiter": maxiter, "adaptive": True}},
        ["fun", "nfev", "success"],
    )
    
    alg.set_log(f"{basename}", f"d={d}, n={n}, c={c}", action="w")
    
    start_time = time()

    alg.benchmark(
        range(1, pmax + 1),
        repeats,
        param_persist=True,
        filename=f"{basename}",
        label=f"d={d}, n={n}, c={c}",
        save_action="w",
        verbose=True,
        time_limit = time_limit,
        suspend_path = f"{suspend_path}/{function.name} d={d} n={n} c={c} non independent t"
    )

    end_time = time()

    time_limit -= max([0, end_time - start_time])
