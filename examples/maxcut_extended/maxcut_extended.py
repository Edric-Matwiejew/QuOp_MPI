import sys
sys.path.append('../../')
import mpi4py.MPI
import h5py
from quop_mpi import Ansatz
from quop_mpi.propagator import diagonal, sparse
from quop_mpi.observable import serial
from quop_mpi.param.rand import uniform
from quop_mpi.toolkit import Z
import numpy as np
import networkx as nx

graph = nx.circular_ladder_graph(4)

n_qubits = graph.number_of_nodes()
n_edges = 2 * graph.number_of_edges()

system_size = 2 ** n_qubits

def maxcut_terms(graph, n_qubits):
    terms = []
    for edge in graph.edges:
        term = Z(edge[0], n_qubits) @ Z(edge[1], n_qubits)
        terms.append(0.5 * (1 + term.diagonal()))
    return terms

def maxcut_qualities(terms):
    return np.sum(terms, axis=0)

computed_terms = maxcut_terms(graph, n_qubits)

UQ = diagonal.unitary(
    diagonal.operator.serial,
    operator_kwargs={"function": maxcut_terms, "args": [graph, n_qubits]},
    unitary_n_params=n_edges,
    parameter_function=uniform,
)

UW = sparse.unitary(sparse.operator.hypercube, parameter_function=uniform)

alg = Ansatz(system_size)

alg.set_unitaries([UQ, UW])

alg.verbose_objective = True
alg.set_observables(serial, {"function": maxcut_qualities, "args": [computed_terms]})
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
alg.save("maxcut_extended", "depth 2", "w")
