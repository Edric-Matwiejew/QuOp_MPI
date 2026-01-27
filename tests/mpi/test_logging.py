"""Tests for set_log() method.

This module tests the logging functionality including:
- Log file creation
- Log file content after execute()
- Append vs overwrite modes
- Log file format
"""

import pytest
import numpy as np
import tempfile
import os
import csv
from mpi4py import MPI


@pytest.mark.mpi
class TestSetLogBasic:
    """Basic tests for set_log() configuration."""

    def test_set_log_creates_file(self, mpi_comm, simple_oracle):
        """Verify set_log creates a log file after execute()."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log")

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "test_run")
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)
            alg.execute(params)

            # Barrier to ensure all ranks have finished
            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                expected_file = log_path + ".csv"
                assert os.path.exists(
                    expected_file
                ), f"Log file not created: {expected_file}"

            del alg

    def test_set_log_adds_csv_extension(self, mpi_comm, simple_oracle):
        """Verify .csv extension is added automatically."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "my_log")  # No .csv

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "label")
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)
            alg.execute(params)

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                # Should have .csv extension
                assert os.path.exists(log_path + ".csv")
                # Should NOT create file without extension
                assert (
                    not os.path.exists(log_path)
                    or os.path.isdir(log_path)
                    or log_path.endswith(".csv")
                )

            del alg

    def test_set_log_with_csv_extension(self, mpi_comm, simple_oracle):
        """Verify explicit .csv extension works."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "my_log.csv")

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "label")
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)
            alg.execute(params)

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                assert os.path.exists(log_path)

            del alg


@pytest.mark.mpi
class TestSetLogContent:
    """Tests for log file content."""

    def test_log_contains_label(self, mpi_comm, simple_oracle):
        """Verify log file contains the specified label."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")
            label = "my_test_label"

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, label)
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)
            alg.execute(params)

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    content = f.read()
                assert label in content, f"Label '{label}' not found in log file"

            del alg

    def test_log_contains_objective_value(self, mpi_comm, simple_oracle):
        """Verify log file contains the final objective value."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "test")
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)
            alg.execute(params)

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                # Should have at least header and one data row
                assert len(rows) >= 2, "Log file should have header and data"

                # Check that objective value is present (as float)
                data_row = rows[-1]
                # One of the columns should be the objective value
                has_float = any(self._is_float(x) for x in data_row)
                assert has_float, "Log should contain numeric objective value"

            del alg

    @staticmethod
    def _is_float(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def test_log_contains_depth(self, mpi_comm, simple_oracle):
        """Verify log file contains the ansatz depth."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle
        depth = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "test")
            alg.set_depth(depth)

            params = oracle.optimal_params(depth=depth)
            alg.execute(params)

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    content = f.read()
                # Depth value should appear somewhere in the log
                assert str(depth) in content, f"Depth {depth} not found in log file"

            del alg


@pytest.mark.mpi
class TestSetLogModes:
    """Tests for log file append/overwrite modes."""

    def test_log_append_mode(self, mpi_comm, simple_oracle):
        """Verify append mode adds to existing log file."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")

            # First run
            alg1 = qwoa(oracle.system_size, mpi_comm)
            alg1.set_qualities(oracle.qualities_function())
            alg1.set_log(log_path, "run1", action="a")
            alg1.set_depth(1)
            alg1.execute(oracle.optimal_params(depth=1))
            del alg1

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    lines_after_first = len(f.readlines())
            else:
                lines_after_first = None
            lines_after_first = mpi_comm.bcast(lines_after_first, root=0)

            # Second run - should append
            alg2 = qwoa(oracle.system_size, mpi_comm)
            alg2.set_qualities(oracle.qualities_function())
            alg2.set_log(log_path, "run2", action="a")
            alg2.set_depth(1)
            alg2.execute(oracle.optimal_params(depth=1))
            del alg2

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    lines_after_second = len(f.readlines())

                # Should have more lines after second run
                assert (
                    lines_after_second > lines_after_first
                ), f"Append mode should add lines: {lines_after_first} -> {lines_after_second}"

    def test_log_overwrite_mode(self, mpi_comm, simple_oracle):
        """Verify overwrite mode replaces log file content."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")

            # First run
            alg1 = qwoa(oracle.system_size, mpi_comm)
            alg1.set_qualities(oracle.qualities_function())
            alg1.set_log(log_path, "first_run", action="w")
            alg1.set_depth(1)
            alg1.execute(oracle.optimal_params(depth=1))
            del alg1

            mpi_comm.barrier()

            # Second run - should overwrite
            alg2 = qwoa(oracle.system_size, mpi_comm)
            alg2.set_qualities(oracle.qualities_function())
            alg2.set_log(log_path, "second_run", action="w")
            alg2.set_depth(1)
            alg2.execute(oracle.optimal_params(depth=1))
            del alg2

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    content = f.read()

                # Should only contain second run label
                assert "second_run" in content, "Overwrite should have second label"
                assert "first_run" not in content, "Overwrite should remove first label"


@pytest.mark.mpi
class TestSetLogWithMultipleExecutions:
    """Tests for logging across multiple execute() calls."""

    def test_multiple_executes_multiple_log_entries(self, mpi_comm, simple_oracle):
        """Verify multiple execute() calls create multiple log entries."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")

            alg = qwoa(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_log(log_path, "multi_run", action="a")
            alg.set_depth(1)

            params = oracle.optimal_params(depth=1)

            # Multiple executions
            n_runs = 3
            for i in range(n_runs):
                alg.execute(params * (0.8 + i * 0.1))

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                with open(log_path, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                # Should have header + n_runs data rows
                # (or just n_runs rows if header is per-run)
                assert (
                    len(rows) >= n_runs
                ), f"Expected at least {n_runs} rows, got {len(rows)}"

            del alg
