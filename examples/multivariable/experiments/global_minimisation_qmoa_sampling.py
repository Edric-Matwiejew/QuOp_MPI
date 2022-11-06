import sys
sys.path.append('../../')
import os
from pathlib import Path
import numpy as np
import h5py
from quop_mpi.algorithm import qmoa
import test_function
from mpi4py import MPI

dmax = int(os.getenv("DMAX"))
dmin = int(os.getenv("DMIN"))

repeatmin = int(os.getenv("REPEATMIN"))
repeatmax = int(os.getenv("REPEATMAX"))

functions = [
    test_function.styblinski_tang,
    test_function.rastrigin,
]

ps = [2, 5]

output_dir = f"results/global_minimisation_qmoa_sampling"

for p, function in zip(ps, functions):

    for repeat in range(repeatmin, repeatmax + 1):
    
        csv_name = f'{output_dir}/{function.name} {repeat}.csv'
    
        if MPI.COMM_WORLD.Get_rank() == 0:
        
            Path(output_dir).mkdir(parents = True, exist_ok = True)
        
            with open(csv_name, 'w') as f:
                f.write('function;repeat;dimension;shots;index;variational_parameters\n')
        
        
        def cost_function(x):
            return function(x)
    
        c_nevals = []
        q_nevals = []
        
        np.random.seed(repeat)
        
        for d in range(2, dmax + 1):
        
            nn = 4
            
            n = 2**nn  # number of grid point per dimension
           
            L = np.diff(function.search_domain(2)[0])[0] / 2
            
            dq = 2 * L / n  # position space grid spacing
            Ns = d * [n]  # shape of d-dimensional grid
            
            # grid spacing in each coordinate
            deltasq = np.array(d * [dq], dtype=np.float64)
            
            # minimum value in each coordinate
            minsq = np.array(d * [-(n / 2) * dq], dtype=np.float64)
            
            alg = qmoa(Ns, deltasq, minsq)
            alg.set_qualities(cost_function)
            alg.set_depth(p)
            alg.set_independent_t(False)  
            alg.set_sampling(30)
            alg.set_optimiser(
                "scipy",
                {"method": "Nelder-Mead", "options": {"maxiter": 1000}},
                ["fun", "nfev", "success"],
            )
            
            np.random.seed(np.random.randint(low = 0, high = 10000))
            alg.execute()
            alg.print_summary()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                with open(csv_name, 'a') as f:
                    for i, (min_index, variational_parameters) in enumerate(zip(alg.sample_minimum_indexes, alg.variational_parameter_history)):
                        params = '['
                        for param in variational_parameters[:-1]:
                            params += f'{param},'
                        params += f'{variational_parameters[-1]}]'
                        f.write(f'{function.name};{repeat};{d};{(i + 1)*30};{min_index};{params}\n')
            
