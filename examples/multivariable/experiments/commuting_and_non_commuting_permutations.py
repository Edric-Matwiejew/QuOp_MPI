import sys
sys.path.append("../../")
import os
from pathlib import Path
import numpy as np
from quop_mpi.algorithm import qmoa
from quop_mpi.algorithm import qwoa
from quop_mpi.__lib import fCQAOA
import test_function
from mpi4py import MPI

repeats = 1
nn = 4

pmax = int(os.getenv("PMAX"))
d = int(os.getenv("DMIN"))

function = test_function.rastrigin

output_dir = f"results/commuting_and_non_commuting_permutations/{function.name} d={d} n={nn}"

if MPI.COMM_WORLD.Get_rank() == 0:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

n = 2 ** nn  # number of grid point per dimension

L = np.diff(function.search_domain(2)[0])[0] / 2

dq = 2 * L / n  # position space grid spacing
Ns = d * [n]  # shape of d-dimensional grid
deltasq = np.array(d * [dq], dtype=np.float64)
minsq = np.array(d * [-(n / 2) * dq], dtype=np.float64)

strides = np.empty(len(Ns), dtype=int)
strides[-1] = 1

for i in range(len(Ns) - 2, -1, -1):
    strides[i] = strides[i + 1] * Ns[i]

permute = False
mix = False

def cost_function(local_i, local_i_offset, MPI_COMM, function = None):

    if MPI_COMM.Get_rank() == 0:
        position_grid = fCQAOA.continuous.gen_local_grid(
            n ** d, Ns, strides, deltasq, minsq, 0, n ** d
        )

        f = []
        for point in position_grid:
            f.append(function(point))

        f = np.array(f, dtype=np.float64)

        if permute:
            f = np.reshape(f, Ns)
            f = f[np.random.permutation(f.shape[0]), :]
            f = f[:, np.random.permutation(f.shape[1])]
            f = f.flatten()

        if mix:
            np.random.shuffle(f)
    else:

        f = None

    return MPI_COMM.bcast(f, root=0)[local_i_offset : local_i_offset + local_i]


alg = qmoa(Ns, deltasq, minsq)
alg.UQ.operator_function = cost_function
alg.UQ.operator_kwargs = {"function": function}
alg.set_observables(0)

alg.set_optimiser(
    "scipy",
    {"method": "Nelder-Mead", "options": {"maxiter": 100000, "adaptive": True}},
    ["fun", "nfev", "success"],
)

alg.set_log(f"{output_dir}/baseline", f"{function.name}", action="w")

alg.benchmark(
    range(1, pmax + 1),
    repeats,
    param_persist=True,
    filename=f"{output_dir}/baseline",
    label=f"{function.name}",
    save_action="w",
    verbose=True,
)


permute = True

alg = qmoa(Ns, deltasq, minsq)
alg.UQ.operator_function = cost_function
alg.UQ.operator_kwargs = {"function": function}
alg.set_observables(0)

alg.set_log(f"{output_dir}/commuting", f"{function.name}", action="w")

alg.benchmark(
    range(1, pmax + 1),
    repeats,
    param_persist=True,
    filename=f"{output_dir}/commuting",
    label=f"{function.name}",
    save_action="w",
    verbose=True,
)

permuted = False
mix = True

alg = qmoa(Ns, deltasq, minsq)
alg.UQ.operator_function = cost_function
alg.UQ.operator_kwargs = {"function": function}
alg.set_observables(0)

alg.set_log(f"{output_dir}/non_commuting", f"{function.name}", action="w")

alg.benchmark(
    range(1, pmax + 1),
    repeats,
    param_persist=True,
    filename=f"{output_dir}/non_commuting",
    label=f"{function.name}",
    save_action="w",
    verbose=True,
)

permuted = False
mix = False

alg = qwoa(n ** d)

alg.set_qualities(cost_function, kwargs = {"function":function})

alg.set_optimiser(
    "scipy",
    {"method": "Nelder-Mead", "options": {"maxiter": 100000, "adaptive": True}},
    ["fun", "nfev", "success"],
)

alg.set_log(f"{output_dir}/qaoa", f"{function.name}", action="w")

alg.benchmark(
    range(1, pmax + 1),
    repeats,
    param_persist=True,
    filename=f"{output_dir}/qaoa",
    label="qaoa",
    save_action="w",
    verbose=True,
)
