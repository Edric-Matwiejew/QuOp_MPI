import sys
sys.path.insert(0,'../')
import quop_bench.benchmark as benchmark

bench_type = str(sys.argv[1])

if bench_type == "evolution":

    exec('import ' + sys.argv[4] + ' as test_module')

    simulation_time = float(sys.argv[2])
    output_filepath = str(sys.argv[3])

    benchmark.evolution(
            simulation_time,
            output_filepath,
            test_module.function
            )

if bench_type == "execute":

    exec('import ' + sys.argv[5] + ' as test_module')

    simulation_time = float(sys.argv[2])
    qubits = int(sys.argv[3])
    output_filepath = str(sys.argv[4])

    benchmark.execute(
            simulation_time,
            qubits,
            output_filepath,
            test_module.function)
