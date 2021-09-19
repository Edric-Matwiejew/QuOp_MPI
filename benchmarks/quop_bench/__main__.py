import sys

sys.path.insert(0, "../")
sys.path.insert(0, ".")
import pickle
import quop_bench.benchmark as benchmark
from quop_mpi.algorithm import qwoa, qaoa
from quop_mpi.observable.rand import uniform
from quop_bench.benchmark import optimisers as benchmark_optimisers


bench_type = str(sys.argv[1])

if bench_type == "evolution":

    exec("import " + sys.argv[4] + " as test_module")

    simulation_time = float(sys.argv[2])
    output_filepath = str(sys.argv[3])

    benchmark.evolution(simulation_time, output_filepath, test_module.function)

if bench_type == "execute_depth":

    exec("import " + sys.argv[7] + " as test_module")

    simulation_time = float(sys.argv[2])
    qubits = int(sys.argv[3])
    output_filepath = str(sys.argv[4])
    bench_log_name = str(sys.argv[5])
    quop_log_name = str(sys.argv[6])

    benchmark.execute_depth(
        simulation_time,
        qubits,
        output_filepath,
        bench_log_name,
        quop_log_name,
        test_module.function,
    )

if bench_type == "execute":

    exec("import " + sys.argv[7] + " as test_module")

    depth = float(sys.argv[2])
    qubits = int(sys.argv[3])
    output_filepath = str(sys.argv[4])
    bench_log_name = str(sys.argv[5])
    quop_log_name = str(sys.argv[6])

    benchmark.execute(
        depth,
        qubits,
        output_filepath,
        bench_log_name,
        quop_log_name,
        test_module.function,
    )

if bench_type == "optimisers":

    basename = str(sys.argv[2])
    alg_name = str(sys.argv[3])

    max_evaluations = 5000

    min_depth = 2
    max_depth = 2

    min_qubits = 3
    max_qubits = 3

    alg_names = [alg_name]
    if alg_name == "qwoa":
        algs = [qwoa]
    elif alg_name == "qaoa":
        algs = [qaoa]

    qualities = uniform

    backends = ["scipy", "nlopt"]

    # unconstrained optimisation methods

    scipy_method_names = ["BFGS", "CG", "Nelder-Mead", "trust-constr", "Powell", "TNC"]

    open_file = open(basename + "/other/scipy_names", "wb")
    pickle.dump(scipy_method_names, open_file)
    open_file.close()

    scipy_method_options = [
        {"method": "BFGS", "jac": True},
        {"method": "CG", "jac": True},
        {"method": "Nelder-Mead", "options": {"maxfev": max_evaluations}},
        {"method": "trust-constr", "jac": True, "hess": "2-point"},
        {"method": "Powell"},
        {"method": "TNC", "jac": True},
    ]

    nlopt_method_names = [
        "LD_LBFGS",
        "LN_BOBYQA",
        "LN_PRAXIS",
        "LN_NELDERMEAD",
        "LN_SBPLX",
        "LD_MMA",
        "LD_CCSAQ",
    ]

    open_file = open(basename + "/other/nlopt_names", "wb")
    pickle.dump(nlopt_method_names, open_file)
    open_file.close()

    nlopt_method_options = [
        {
            "method": "LD_LBFGS",
            "jac": True,
            "ftol_abs": 1e-12,
            "maxeval": max_evaluations,
        },
        {"method": "LN_BOBYQA", "ftol_abs": 1e-12, "maxeval": max_evaluations},
        {"method": "LN_PRAXIS", "ftol_abs": 1e-12, "maxeval": max_evaluations},
        {"method": "LN_NELDERMEAD", "ftol_abs": 1e-12, "maxeval": max_evaluations},
        {"method": "LN_SBPLX", "ftol_abs": 1e-12, "maxeval": max_evaluations},
        {"method": "LD_MMA", "ftol_abs": 1e-12, "maxeval": max_evaluations},
        {"method": "LD_CCSAQ", "ftol_abs": 1e-12, "maxeval": max_evaluations},
    ]

    method_names = [scipy_method_names, nlopt_method_names]

    options = [scipy_method_options, nlopt_method_options]

    log_filename = basename + "/csv"
    objective_history_filename = basename + "/npy"

    benchmark.optimisers(
        min_qubits,
        max_qubits,
        min_depth,
        max_depth,
        algs,
        alg_names,
        qualities,
        backends,
        method_names,
        options,
        log_filename,
        objective_history_filename,
    )
