from importlib import import_module
import numpy as np

def ith(Ns = None, Cs = None):

    circulant_eigenvalues = import_module('quop_mpi.propagator.circulant.operator.eigenvalues')

    eigenvalues = np.zeros((np.max(Ns), len(Ns)), dtype = np.complex128)

    for i, (N, C)  in enumerate(zip(Ns, Cs)):
        eigenvalues[:N,i] = circulant_eigenvalues.graph(N, N, 0, C)

    return eigenvalues

 
