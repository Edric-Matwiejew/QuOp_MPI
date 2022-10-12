from importlib import import_module
import numpy as np

def magnitude_squared(Ns, minsk, deltask):

    grid = np.empty(max(Ns), dtype = np.float64)
    momentums = np.zeros((max(Ns), len(Ns)), dtype = np.complex128)

    for i, (N, mink, deltak)  in enumerate(zip(Ns, minsk, deltask)):

        grid[0] = mink

        for j in range(1, N):
            grid[i] = (mink + j*deltak)**2

        momentums[:N,i] = grid[:N]

    return momentums


