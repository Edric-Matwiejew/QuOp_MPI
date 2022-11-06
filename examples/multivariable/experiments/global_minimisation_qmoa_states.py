import sys
sys.path.append('../../')
from pathlib import Path
import os
import numpy as np
import h5py
import pandas as pd
from quop_mpi.algorithm import qmoa
from quop_mpi.__lib import fCQAOA
import test_function
from mpi4py import MPI

functions = [
    test_function.styblinski_tang,
    test_function.rastrigin,
]

ps = [2, 5]

dfs = [
        pd.read_csv(
            f"results/global_minimisation_qmoa_and_nelder_mead/{function.name}.csv",
            sep = ";"
            )
        for function in functions
        ]

for function, df, p in zip(functions, dfs, ps):

      for d, variational_parameters in zip(df['dimension'], df['variational_parameters']):
        params = np.array(eval(variational_parameters), dtype = np.float64)
    
        nn = 4
        
        n = 2**nn  # number of grid point per dimension
       
        L = np.diff(function.search_domain(2)[0])[0] / 2
        
        dq = 2 * L / n  # position space grid spacing
        Ns = d * [n]  # shape of d-dimensional grid
        
        # grid spacing in each coordinate
        deltasq = np.array(d * [dq], dtype=np.float64)
        
        # minimum value in each coordinate
        minsq = np.array(d * [-(n / 2) * dq], dtype=np.float64)
     
        def cost_function(x):
            return function(x)

        alg = qmoa(Ns, deltasq, minsq)
        alg.set_independent_t(False)
        alg.set_qualities(cost_function)
        alg.set_depth(p)
        alg.evolve_state(params)
        probs = alg.get_probabilities()
        
        if alg.MPI_COMM.Get_rank() == 0:
  
            output_dir = f"results/global_minimisation_qmoa_states/{function.name}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            #grid = fCQAOA.continuous.gen_local_grid(
            #    alg.system_size,
            #    Ns,
            #    alg.UW.strides,
            #    alg.deltas,
            #    alg.mins,
            #    0,
            #    alg.system_size,
            #)
    
            #np.savez(f"{output_dir}/{d}", probs, grid, observables)
            #           

        alg.save(f'{output_dir}/{d}', 'min_iden')
