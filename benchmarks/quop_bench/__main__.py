import sys
sys.path.insert(0,'../')
exec('import ' + sys.argv[4] + ' as test_module')
import quop_bench.benchmark as benchmark

bench_type = str(sys.argv[1])
simulation_time = float(sys.argv[2])
output_filepath = str(sys.argv[3])

if bench_type == "evolution":

    benchmark.evolution(
            simulation_time,
            output_filepath,
            test_module.function
            )
