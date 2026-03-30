"""Tests for save() method.

This module tests the HDF5 save functionality including:
- Basic file creation and structure
- Config naming and duplicate handling
- Data integrity (final_state, observables, variational_parameters)
- Append vs overwrite modes
"""

import os
import tempfile

import numpy as np
import pytest

from tests.conftest import TestOracle


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps save tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the save-test oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.fixture
def save_small_system_size(mpi_sizing):
    """Small save-test size for quick distributed round-trip checks."""
    return mpi_sizing.power_of_two(base=16, min_per_rank=1)


@pytest.fixture
def save_medium_system_size(mpi_sizing):
    """Moderate save-test size that keeps multiple ranks active."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1, min_per_node=16)


@pytest.fixture
def save_partition_system_size(mpi_sizing):
    """Non-power-of-two size for partition-boundary coverage."""
    return mpi_sizing.multiple(base=24, per_rank=3, per_node=12)


@pytest.fixture
def save_uneven_system_size(mpi_sizing):
    """Prime size that preserves the uneven-partition contract."""
    return mpi_sizing.prime(base=17, min_per_rank=1, min_per_node=8)


@pytest.fixture
def save_large_system_size(mpi_sizing):
    """Larger save-test size for stress-style coverage."""
    return mpi_sizing.power_of_two(base=256, min_per_rank=8, min_per_node=64)


@pytest.fixture
def temp_h5_file(mpi_comm):
    """Create a temporary HDF5 file path for testing."""
    # Only rank 0 creates the temp file path
    if mpi_comm.Get_rank() == 0:
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        os.unlink(path)  # Remove, let save() create it
        # Remove .h5 extension as save() adds it
        path = path[:-3]
    else:
        path = None

    path = mpi_comm.bcast(path, root=0)

    yield path

    # Cleanup
    mpi_comm.barrier()
    if mpi_comm.Get_rank() == 0:
        full_path = path + ".h5"
        if os.path.exists(full_path):
            os.unlink(full_path)


@pytest.mark.mpi
class TestSaveBasic:
    """Basic tests for save() file creation."""

    def test_save_creates_h5_file(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify save() creates an HDF5 file."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, "test_config")

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            assert os.path.exists(
                temp_h5_file + ".h5"
            ), f"HDF5 file not created at {temp_h5_file}.h5"

        alg.destroy()

    def test_save_creates_config_group(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify save() creates a config group in the HDF5 file."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "my_simulation"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert config_name in f, f"Config group '{config_name}' not found in file"

        alg.destroy()

    def test_save_adds_h5_extension(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify save() adds .h5 extension if not present."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        # Pass filename without extension
        alg.save(temp_h5_file, "test_config")

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            # File should have .h5 extension
            assert os.path.exists(temp_h5_file + ".h5")

        alg.destroy()


@pytest.mark.mpi
class TestSaveDatasets:
    """Tests for save() dataset contents."""

    def test_save_contains_final_state(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains final_state dataset."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "state_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "final_state" in f[config_name], "final_state dataset not found"

                final_state = np.array(f[config_name]["final_state"]).view(np.complex128)
                assert final_state.shape[0] == oracle.system_size

        alg.destroy()

    def test_save_contains_observables(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains observables dataset."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "obs_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "observables" in f[config_name], "observables dataset not found"

                observables = np.array(f[config_name]["observables"]).view(np.float64)
                assert observables.shape[0] == oracle.system_size

        alg.destroy()

    def test_save_contains_initial_state(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains initial_state dataset."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "init_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "initial_state" in f[config_name], "initial_state dataset not found"

                initial_state = np.array(f[config_name]["initial_state"]).view(np.complex128)
                assert initial_state.shape[0] == oracle.system_size

        alg.destroy()

    def test_save_contains_variational_params(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains initial_phases (variational parameters)."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "params_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                # Verify the initial_phases dataset exists
                assert "initial_phases" in f[config_name], "initial_phases dataset not found"

        alg.destroy()

    def test_save_contains_minimize_result(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains minimize_result attribute."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "result_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert (
                    "minimize_result" in f[config_name].attrs
                ), "minimize_result attribute not found"

                result_str = f[config_name].attrs["minimize_result"]
                assert len(result_str) > 0

        alg.destroy()


@pytest.mark.mpi
class TestSaveDuplicateHandling:
    """Tests for save() duplicate config name handling."""

    def test_save_duplicate_config_adds_underscore(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify duplicate config names get underscore suffix."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "duplicate_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        # Save twice with same config name
        alg.save(temp_h5_file, config_name)
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert config_name in f, "Original config not found"
                assert f"{config_name}_" in f, "Duplicate config with underscore not found"

        alg.destroy()


@pytest.mark.mpi
class TestSaveModes:
    """Tests for save() append vs overwrite modes."""

    def test_save_append_mode(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify append mode preserves existing configs."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        # Save first config
        alg.save(temp_h5_file, "config_1", action="a")

        # Save second config with append
        alg.save(temp_h5_file, "config_2", action="a")

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "config_1" in f, "First config not preserved"
                assert "config_2" in f, "Second config not added"

        alg.destroy()

    def test_save_overwrite_mode(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify overwrite mode replaces file contents."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        # Save first config
        alg.save(temp_h5_file, "old_config", action="a")

        # Save with overwrite mode
        alg.save(temp_h5_file, "new_config", action="w")

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "old_config" not in f, "Old config should be removed"
                assert "new_config" in f, "New config should exist"

        alg.destroy()


@pytest.mark.mpi
class TestSaveDataIntegrity:
    """Tests for save() data integrity."""

    def test_saved_state_matches_computed(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved final_state matches computed state."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "integrity_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        # Get the expectation before saving
        expectation_before = alg.get_expectation_value()

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                final_state = np.array(f[config_name]["final_state"]).view(np.complex128)
                observables = np.array(f[config_name]["observables"]).view(np.float64)

                # Compute expectation from saved data
                probabilities = np.abs(final_state) ** 2
                saved_expectation = np.dot(probabilities, observables)

                assert np.isclose(
                    expectation_before, saved_expectation, rtol=1e-10
                ), f"Expectation mismatch: computed={expectation_before}, saved={saved_expectation}"

        alg.destroy()

    def test_saved_observables_are_valid(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved observables are valid (non-empty, correct size, finite)."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        config_name = "qualities_test"

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.execute(params)

        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_obs = np.array(f[config_name]["observables"]).view(np.float64)

                # Verify the observables are valid
                assert (
                    saved_obs.shape[0] == oracle.system_size
                ), f"Expected {oracle.system_size} observables, got {saved_obs.shape[0]}"
                assert np.all(np.isfinite(saved_obs)), "Observables contain non-finite values"

        alg.destroy()


# =============================================================================
# Tests for Parallel I/O Data Integrity
# =============================================================================


@pytest.mark.mpi
class TestParallelIODataIntegrity:
    """Tests verifying data integrity when saved in parallel across MPI ranks."""

    def test_distributed_state_assembled_correctly(
        self, mpi_comm, temp_h5_file, save_small_system_size
    ):
        """Verify that distributed state pieces are correctly assembled in HDF5."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_small_system_size

        def qualities(local_i, local_i_offset):
            # Each rank has unique observable values for identification
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64) * 2.0

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Use non-trivial parameters for non-uniform state
        params = np.array([0.3, 0.5])
        alg.evolve_state(params)

        # Get the full state via proper API
        full_state_gathered = alg.get_final_state()

        # Save
        config_name = "dist_state_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_state = np.array(f[config_name]["final_state"]).view(np.complex128)

                # Verify saved state matches gathered state
                np.testing.assert_allclose(
                    saved_state,
                    full_state_gathered,
                    rtol=1e-14,
                    err_msg="Saved state does not match gathered distributed state",
                )

        alg.destroy()

    def test_distributed_observables_assembled_correctly(
        self, mpi_comm, temp_h5_file, save_medium_system_size
    ):
        """Verify that distributed observables are correctly assembled in HDF5."""
        import h5py

        from quop_mpi._utils._mpi import gather_array
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_medium_system_size

        def qualities(local_i, local_i_offset):
            # Unique pattern: offset squared to identify rank contributions
            return np.array(
                [(local_i_offset + i) ** 2 * 0.1 for i in range(local_i)],
                dtype=np.float64,
            )

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        params = np.array([0.0, 0.0])
        alg.execute(params)

        config_name = "dist_obs_test"
        full_obs_gathered = None

        if alg.subcomms.in_subcomm():
            # Gather observables after setup (observables are set after execute)
            local_obs = alg.local_observables[: alg.local_i]
            partition_table = alg.partition_table

            full_obs_gathered = gather_array(local_obs, partition_table, alg.MPI_COMM)

            alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_obs = np.array(f[config_name]["observables"]).view(np.float64)

                # Verify saved observables match gathered values
                np.testing.assert_allclose(
                    saved_obs,
                    full_obs_gathered,
                    rtol=1e-14,
                    err_msg="Saved observables do not match gathered distributed observables",
                )

        alg.destroy()

    def test_partition_contributions_are_contiguous(
        self, mpi_comm, temp_h5_file, save_partition_system_size
    ):
        """Verify each rank's partition appears at correct offset in saved file."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_partition_system_size

        # Use rank-identifiable values
        # Encode as rank * system_size + global_index so modular decode works
        # for any system_size (the old hardcoded 1000 broke when system_size > 1000).
        stride = system_size

        def qualities(local_i, local_i_offset):
            rank = mpi_comm.Get_rank()
            return np.array(
                [rank * stride + local_i_offset + i for i in range(local_i)],
                dtype=np.float64,
            )

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Store local info for verification

        params = np.array([0.0, 0.0])
        alg.execute(params)

        config_name = "partition_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_obs = np.array(f[config_name]["observables"]).view(np.float64)

                # Each element encodes: rank * system_size + global_index
                for i in range(system_size):
                    val = saved_obs[i]
                    encoded_index = int(val) % stride

                    assert (
                        encoded_index == i
                    ), f"Index mismatch at position {i}: expected {i}, got {encoded_index}"

        alg.destroy()

    def test_complex_data_real_imag_preserved(
        self, mpi_comm, temp_h5_file, save_small_system_size
    ):
        """Verify complex state data preserves both real and imaginary parts."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Use parameters that produce complex amplitudes
        params = np.array([np.pi / 4, np.pi / 3])
        alg.evolve_state(params)

        # Store expected state using proper API
        full_expected = alg.get_final_state()

        config_name = "complex_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_state = np.array(f[config_name]["final_state"]).view(np.complex128)

                # Verify real parts match
                np.testing.assert_allclose(
                    saved_state.real,
                    full_expected.real,
                    rtol=1e-14,
                    err_msg="Real parts of state do not match",
                )

                # Verify imaginary parts match
                np.testing.assert_allclose(
                    saved_state.imag,
                    full_expected.imag,
                    rtol=1e-14,
                    err_msg="Imaginary parts of state do not match",
                )

        alg.destroy()

    def test_uneven_partition_sizes(self, mpi_comm, temp_h5_file, save_uneven_system_size):
        """Verify correct saving with uneven partition sizes across ranks."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        # Use a prime size so partitioning stays uneven as the MPI layout scales.
        system_size = save_uneven_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Verify uneven partitioning exists (note: local_i is 0 before setup)
        params = np.array([0.2, 0.3])
        alg.execute(params)

        if alg.subcomms.in_subcomm():
            alg.MPI_COMM.allgather(alg.local_i)

        config_name = "uneven_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_obs = np.array(f[config_name]["observables"]).view(np.float64)
                saved_state = np.array(f[config_name]["final_state"]).view(np.complex128)

                # Verify correct sizes
                assert (
                    saved_obs.shape[0] == system_size
                ), f"Observables size {saved_obs.shape[0]} != system_size {system_size}"
                assert (
                    saved_state.shape[0] == system_size
                ), f"State size {saved_state.shape[0]} != system_size {system_size}"

                # Verify observables are sequential
                expected_obs = np.arange(system_size, dtype=np.float64)
                np.testing.assert_allclose(saved_obs, expected_obs, rtol=1e-14)

        alg.destroy()

    def test_large_system_parallel_save(self, mpi_comm, temp_h5_file, save_large_system_size):
        """Test parallel save with larger system size for stress testing."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_large_system_size

        def qualities(local_i, local_i_offset):
            return np.sin(np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64))

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        params = np.array([0.1, 0.2])
        alg.execute(params)

        config_name = "large_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_state = np.array(f[config_name]["final_state"]).view(np.complex128)
                saved_obs = np.array(f[config_name]["observables"]).view(np.float64)

                # Verify sizes
                assert saved_state.shape[0] == system_size
                assert saved_obs.shape[0] == system_size

                # Verify normalization
                probs = np.abs(saved_state) ** 2
                assert np.isclose(
                    np.sum(probs), 1.0, rtol=1e-10
                ), f"State not normalized: sum(|psi|^2) = {np.sum(probs)}"

                # Verify observables match expected
                expected_obs = np.sin(np.arange(system_size, dtype=np.float64))
                np.testing.assert_allclose(saved_obs, expected_obs, rtol=1e-10)

        alg.destroy()

    def test_initial_state_parallel_save(
        self, mpi_comm, temp_h5_file, save_medium_system_size
    ):
        """Verify initial_state is correctly saved in parallel."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_medium_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        params = np.array([0.0, 0.0])
        alg.execute(params)

        config_name = "init_state_test"
        alg.save(temp_h5_file, config_name)

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                saved_init = np.array(f[config_name]["initial_state"]).view(np.complex128)

                # Initial state should be uniform superposition
                expected_amplitude = 1.0 / np.sqrt(system_size)
                expected_init = np.full(system_size, expected_amplitude, dtype=np.complex128)

                np.testing.assert_allclose(saved_init, expected_init, rtol=1e-10)

        alg.destroy()

    def test_multiple_configs_independent(
        self, mpi_comm, temp_h5_file, save_small_system_size
    ):
        """Verify multiple saved configs don't interfere with each other."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = save_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Save with different parameters
        params1 = np.array([0.1, 0.1])
        alg.evolve_state(params1)
        alg.save(temp_h5_file, "config_1", action="a")

        params2 = np.array([0.5, 0.5])
        alg.evolve_state(params2)
        alg.save(temp_h5_file, "config_2", action="a")

        params3 = np.array([1.0, 1.0])
        alg.evolve_state(params3)
        alg.save(temp_h5_file, "config_3", action="a")

        mpi_comm.barrier()

        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                state1 = np.array(f["config_1"]["final_state"]).view(np.complex128)
                state2 = np.array(f["config_2"]["final_state"]).view(np.complex128)
                state3 = np.array(f["config_3"]["final_state"]).view(np.complex128)

                # States should be different
                assert not np.allclose(state1, state2), "config_1 and config_2 should differ"
                assert not np.allclose(state2, state3), "config_2 and config_3 should differ"
                assert not np.allclose(state1, state3), "config_1 and config_3 should differ"

                # Each should be normalized
                for state, name in [
                    (state1, "config_1"),
                    (state2, "config_2"),
                    (state3, "config_3"),
                ]:
                    probs = np.abs(state) ** 2
                    assert np.isclose(np.sum(probs), 1.0, rtol=1e-10), f"{name} not normalized"

        alg.destroy()
