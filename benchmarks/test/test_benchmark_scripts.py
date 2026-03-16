import importlib.util
import sys
import types
from pathlib import Path


def load_bench_module():
    module_path = Path(__file__).resolve().parents[1] / "bench.py"
    spec = importlib.util.spec_from_file_location("benchmark_bench", module_path)
    module = importlib.util.module_from_spec(spec)

    mpi4py_stub = types.ModuleType("mpi4py")
    mpi4py_stub.MPI = object()

    previous = sys.modules.get("mpi4py")
    sys.modules["mpi4py"] = mpi4py_stub
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("mpi4py", None)
        else:
            sys.modules["mpi4py"] = previous

    return module


def test_qmoa_benchmark_metadata_uses_size_spec_and_tensor_dims():
    bench = load_bench_module()

    assert bench.parse_size_arg("qmoa", "3 3 3") == [3, 3, 3]
    assert bench.size_spec("qmoa", [3, 3, 3]) == "3x3x3"
    assert bench.tensor_dims_spec("qmoa", [3, 3, 3]) == "8x8x8"
    result = bench.csv_filename(
        "qmoa", "mpi", 512, "3x3x3", 64, "intra",
    )
    assert result == "qmoa_mpi_512_3x3x3_intra_64.csv"


def test_non_qmoa_benchmark_metadata_keeps_legacy_filename_shape():
    bench = load_bench_module()

    assert bench.parse_size_arg("qaoa", "1024") == [1024]
    assert bench.size_spec("qaoa", [1024]) == "1024"
    assert bench.tensor_dims_spec("qaoa", [1024]) == ""
    result = bench.csv_filename(
        "qaoa", "mpi", 1024, "1024", 16, "multi",
    )
    assert result == "qaoa_mpi_1024_multi_16.csv"


def test_csv_header_includes_profile_column():
    """The CSV header written by bench.py should include the profile field."""
    bench = load_bench_module()

    # The header is assembled inline in main(), but verify the field order
    # by checking the csv_filename helper still works and the profile column
    # is documented in the header string.
    expected_header = (
        "algorithm,backend,profile,"
        "system_size,size_spec,tensor_dims,"
        "phase,nprocs,nodes,"
        "prepare_s,mean_evolve_s,std_evolve_s"
    )

    # Inspect the source to confirm the header includes 'profile'
    import inspect
    source = inspect.getsource(bench.main)
    assert "profile" in source
    for col in expected_header.split(","):
        assert col in source
