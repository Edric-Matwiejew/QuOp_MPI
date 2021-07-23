import quop_bench.benchmark as benchmark

bench_type = sys.argv[1]
simulation_time = float(sys.argv[2])
output_filepath = sys.argv[3]
test_module_name = import_module(sys.argv[4])

if bench_type == "evolution":

    test_module = import_module(test_module)

    evolution(
            simulation_time,
            output_filepath,
            test_module.function
            )
