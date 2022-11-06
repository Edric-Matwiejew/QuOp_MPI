import sys

sys.path.append('../../')
import os
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from scipy.optimize import minimize as sp_minimize
from quop_mpi.algorithm import qmoa
import test_function
from quop_mpi.__lib import fCQAOA

dmin = int(os.getenv("DMIN"))
dmax = int(os.getenv("DMAX"))

functions = [
    test_function.styblinski_tang,
    test_function.rastrigin,
]

qmoa_dfs = []
for function in functions:
    paths = glob(f"results/global_minimisation_qmoa_sampling/{function.name}*.csv")
    dfs = [pd.read_csv(f"{a}", delimiter = ';') for a in paths]
    qmoa_dfs.append(pd.concat(dfs))

ps = [2, 5]


def grid_point_from_index(index, Ns, strides, deltasq, minsq):

    inds = fCQAOA.continuous.get_index(index + 1, Ns, strides)
    grid_points = (inds - 1) * deltasq
    grid_points += minsq
    return grid_points

for function in functions:

    for p, (function, df) in zip(ps, zip(functions, qmoa_dfs)):
    
        output_dir = f"results/global_minimisation_qmoa_and_nelder_mead"
        csv_name = f"{output_dir}/{function.name}.csv"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(csv_name, "w") as f:
            f.write("function;depth;dimension;mean nfev;std;variational_parameters\n")
            f.flush()
    
        for d in range(2, dmax + 1):
    
            d_df = df.loc[df["dimension"] == d]
    
            n = 16
    
            L = np.diff(function.search_domain(2)[0])[0] / 2
    
            dq = 2 * L / n  # position space grid spacing
            Ns = d * [n]  # shape of d-dimensional grid
            minsq = np.array(d * [-(n / 2) * dq], dtype=np.float64)
            deltasq = np.array(d * [dq], dtype=np.float64)
    
            strides = np.empty(len(Ns), dtype=int)
            strides[-1] = 1
            for i in range(len(Ns) - 2, -1, -1):
                strides[i] = strides[i + 1] * Ns[i]
    
    
            evals = [0]
            repeats = len(d_df['repeat'].unique())
    
            found = True
    
            for repeat in d_df['repeat'].unique():
                repeat_df = d_df.loc[d_df["repeat"] == repeat]
    
                if found:
                    assisted_evals = 0
                    assisted_minimum = np.inf
    
                bounds = [function.search_domain(2)[0]] * d
    
                min_indexes = iter(repeat_df["index"].values)
                min_shots = iter(repeat_df["shots"].values)
                variational_parameters = iter(repeat_df["variational_parameters"].values)
                index = next(min_indexes)
                shots = next(min_shots)
    
                print(function.name, function.minimum(d))
    
                cvals = 0
                while (
                    not np.isclose(function.minimum(d)[0], assisted_minimum, atol=1e-5)
                ) and (index is not None):
                    params = next(variational_parameters, None)
                    x = grid_point_from_index(index, Ns, strides, deltasq, minsq)
                    print(function.name, d, x)
                    result = sp_minimize(function, x, bounds=bounds, method="Nelder-Mead")
                    print(function.name, result, flush = True)
                    cvals += result["nfev"]  
                    assisted_evals = cvals + (p + 1) * shots
                    assisted_minimum = np.min([assisted_minimum, result["fun"]])
                    index = next(min_indexes, None)
                    shots = next(min_shots, None)
    
                if index is not None:
                    evals[-1] += assisted_evals
                    evals.append(assisted_evals)
                else:
                    evals[-1] += assisted_evals
    
            with open(csv_name, "a") as f:
                f.write(f"{function.name};{p};{d};{np.mean(evals)};{np.std(evals)/np.sqrt(repeats)};{params}\n")
                f.flush()
