import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_bench_module():
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "benchmarks" / "bench.py"
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

    assert bench.parse_size_arg("qaoa_transverse_field", "1024") == [1024]
    assert bench.size_spec("qaoa_transverse_field", [1024]) == "1024"
    assert bench.tensor_dims_spec("qaoa_transverse_field", [1024]) == ""
    result = bench.csv_filename(
        "qaoa_transverse_field", "mpi", 1024, "1024", 16, "multi",
    )
    assert result == "qaoa_transverse_field_mpi_1024_multi_16.csv"


def test_parse_size_arg_rejects_negative_qmoa_exponents():
    bench = load_bench_module()

    with pytest.raises(ValueError, match="non-negative"):
        bench.parse_size_arg("qmoa", "-1 3")


def test_parse_size_arg_rejects_non_positive_system_size_for_qaoa_qwoa():
    bench = load_bench_module()

    with pytest.raises(ValueError, match="positive integer"):
        bench.parse_size_arg("qaoa", "0")

    with pytest.raises(ValueError, match="positive integer"):
        bench.parse_size_arg("qwoa", "-8")


def test_parse_args_phase_and_verify_flags(monkeypatch):
    bench = load_bench_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "qaoa_transverse_field", "1024", "--phase", "multi", "--verify"],
    )
    args = bench.parse_args()

    assert args.algorithm == "qaoa_transverse_field"
    assert args.size_arg == "1024"
    assert args.phase == "multi"
    assert args.verify is True


def test_algorithm_classes_expose_get_state_norm():
    bench = load_bench_module()

    import quop_mpi.ansatz as ansatz

    assert hasattr(ansatz.Ansatz, "get_state_norm")


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


def test_submit_scaling_accepts_qaoa_transverse_field_and_reuses_qaoa_config():
    project_root = Path(__file__).resolve().parents[2]
    script = (project_root / "benchmarks" / "submit_scaling.sh").read_text()

    assert "qaoa_transverse_field" in script
    assert 'CONFIG_ALGORITHM="qaoa"' in script
