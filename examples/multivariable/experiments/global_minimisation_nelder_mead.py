import sys
sys.path.append("../../")
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize as sp_minimize
import test_function
from mpi4py import MPI
from quop_mpi.__lib import fCQAOA
from mpi4py import MPI

dmax = int(os.getenv("DMAX"))
dmin = int(os.getenv("DMIN"))

repeatmin = int(os.getenv("REPEATMIN"))
repeatmax = int(os.getenv("REPEATMAX"))

functions = [
    test_function.styblinski_tang,
    test_function.rastrigin,
]

output_dir = f"results/global_minimisation_nelder_mead"

for function in functions:

    csv_name = f'{output_dir}/{function.name}.csv'
        
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    size = COMM.Get_size()
    
    if MPI.COMM_WORLD.Get_rank() == 0:
    
        Path(output_dir).mkdir(parents = True, exist_ok = True)
    
        if not os.path.isfile(csv_name):
            with open(csv_name, 'w') as f:
                f.write('function,repeat,dimension,nfev,fun\n')
                f.flush()
    
    for d in range(dmin, dmax + 1):
    
        for repeat in range(repeatmin, repeatmax + 1):
    
            np.random.seed(repeat*size + rank)
    
            classical_evals = 0
            classical_minimum = np.inf
            
            bounds = [function.search_domain(2)[0]] * d
       
            classical_minimum = np.inf
            
            while not np.isclose(
                function.minimum(d)[0], classical_minimum, atol=1e-5
            ):
                result = sp_minimize(
                    function,
                    np.random.uniform(low = bounds[0][0], high = bounds[0][1], size = d),
                    bounds=bounds,
                    method="Nelder-Mead",
                )
                print(result)
                classical_evals += result["nfev"]
                classical_minimum = np.min([classical_minimum, result["fun"]])
            
            nevals = COMM.gather(classical_evals, root = 0)
            funs = COMM.gather(classical_minimum, root = 0)
            
            if COMM.Get_rank() == 0:
                with open(csv_name, 'a') as f:
                    for i, (neval, fun) in enumerate(zip(nevals, funs)):
                        f.write(f'{function.name},{repeat*size + i},{d},{neval},{fun}\n')
                        f.flush()
