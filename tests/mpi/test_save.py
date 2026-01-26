"""Tests for save() method.

This module tests the HDF5 save functionality including:
- Basic file creation and structure
- Config naming and duplicate handling
- Data integrity (final_state, observables, variational_parameters)
- Append vs overwrite modes
"""

import pytest
import numpy as np
import os
import tempfile
from mpi4py import MPI


@pytest.fixture
def temp_h5_file(mpi_comm):
    """Create a temporary HDF5 file path for testing."""
    # Only rank 0 creates the temp file path
    if mpi_comm.Get_rank() == 0:
        fd, path = tempfile.mkstemp(suffix='.h5')
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
        from quop_mpi.algorithm.combinatorial import qwoa
        
        oracle = simple_oracle
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, "test_config")
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            assert os.path.exists(temp_h5_file + ".h5"), \
                f"HDF5 file not created at {temp_h5_file}.h5"
        
        del alg

    def test_save_creates_config_group(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify save() creates a config group in the HDF5 file."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "my_simulation"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert config_name in f, \
                    f"Config group '{config_name}' not found in file"
        
        del alg

    def test_save_adds_h5_extension(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify save() adds .h5 extension if not present."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        oracle = simple_oracle
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
        
        del alg


@pytest.mark.mpi
class TestSaveDatasets:
    """Tests for save() dataset contents."""

    def test_save_contains_final_state(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains final_state dataset."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "state_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "final_state" in f[config_name], \
                    "final_state dataset not found"
                
                final_state = np.array(f[config_name]["final_state"]).view(np.complex128)
                assert final_state.shape[0] == oracle.system_size
        
        del alg

    def test_save_contains_observables(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains observables dataset."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "obs_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "observables" in f[config_name], \
                    "observables dataset not found"
                
                observables = np.array(f[config_name]["observables"]).view(np.float64)
                assert observables.shape[0] == oracle.system_size
        
        del alg

    def test_save_contains_initial_state(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains initial_state dataset."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "init_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "initial_state" in f[config_name], \
                    "initial_state dataset not found"
                
                initial_state = np.array(f[config_name]["initial_state"]).view(np.complex128)
                assert initial_state.shape[0] == oracle.system_size
        
        del alg

    def test_save_contains_variational_params(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains initial_phases (variational parameters)."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "params_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                # Verify the initial_phases dataset exists
                assert "initial_phases" in f[config_name], \
                    "initial_phases dataset not found"
        
        del alg

    def test_save_contains_minimize_result(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved file contains minimize_result attribute."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "result_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        
        params = oracle.optimal_params(depth=1)
        alg.execute(params)
        
        alg.save(temp_h5_file, config_name)
        
        mpi_comm.barrier()
        
        if mpi_comm.Get_rank() == 0:
            with h5py.File(temp_h5_file + ".h5", "r") as f:
                assert "minimize_result" in f[config_name].attrs, \
                    "minimize_result attribute not found"
                
                result_str = f[config_name].attrs["minimize_result"]
                assert len(result_str) > 0
        
        del alg


@pytest.mark.mpi
class TestSaveDuplicateHandling:
    """Tests for save() duplicate config name handling."""

    def test_save_duplicate_config_adds_underscore(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify duplicate config names get underscore suffix."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "duplicate_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
        
        del alg


@pytest.mark.mpi
class TestSaveModes:
    """Tests for save() append vs overwrite modes."""

    def test_save_append_mode(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify append mode preserves existing configs."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
        
        del alg

    def test_save_overwrite_mode(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify overwrite mode replaces file contents."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
        
        del alg


@pytest.mark.mpi
class TestSaveDataIntegrity:
    """Tests for save() data integrity."""

    def test_saved_state_matches_computed(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved final_state matches computed state."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "integrity_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
                
                assert np.isclose(expectation_before, saved_expectation, rtol=1e-10), \
                    f"Expectation mismatch: computed={expectation_before}, saved={saved_expectation}"
        
        del alg

    def test_saved_observables_are_valid(self, mpi_comm, simple_oracle, temp_h5_file):
        """Verify saved observables are valid (non-empty, correct size, finite)."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import h5py
        
        oracle = simple_oracle
        config_name = "qualities_test"
        
        alg = qwoa(oracle.system_size, mpi_comm)
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
                assert saved_obs.shape[0] == oracle.system_size, \
                    f"Expected {oracle.system_size} observables, got {saved_obs.shape[0]}"
                assert np.all(np.isfinite(saved_obs)), "Observables contain non-finite values"
        
        del alg
