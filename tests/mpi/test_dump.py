"""
T7.6 - T7.8: Smoke tests for the diagnostic dump utility.

Verifies that ``dump_comm_info`` is triggered by the ``QUOP_DUMP_COMM_INFO``
environment variable and writes the expected file(s).

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_dump.py -v --with-mpi --backend mpi
"""

import glob
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

# -- Helpers ----------------------------------------------------------


@pytest.fixture
def dump_system_size(small_system_size):
    """Small representative system size for dump smoke tests."""
    return small_system_size


@pytest.fixture
def mpi_shared_work_dir(mpi_comm, request):
    """Create one shared work directory per test without using pytest tmp fixtures."""
    if mpi_comm.Get_rank() == 0:
        root = Path.cwd() / ".quop_pytest_tmp" / "mpi_dump"
        root.mkdir(parents=True, exist_ok=True)
        node_name = request.node.name.replace(os.sep, "_")
        work_dir = root / f"{node_name}_{uuid.uuid4().hex}"
    else:
        work_dir = None

    work_dir = Path(mpi_comm.bcast(str(work_dir) if work_dir is not None else None, root=0))

    os.makedirs(work_dir, exist_ok=True)
    mpi_comm.Barrier()

    yield work_dir

    mpi_comm.Barrier()
    if mpi_comm.Get_rank() == 0:
        shutil.rmtree(work_dir, ignore_errors=True)
    mpi_comm.Barrier()


def _run_qaoa(comm, system_size, depth=1):
    """Run a tiny QAOA through setup + execute."""
    from quop_mpi.algorithm.combinatorial import QAOA

    def qualities(local_i, local_i_offset):
        obs = np.ones(local_i, dtype=np.float64)
        if local_i_offset == 0:
            obs[0] = 0.0
        return obs

    alg = QAOA(system_size, comm)
    alg.set_qualities(qualities)
    alg.set_depth(depth)
    alg.execute()
    alg.destroy()


# -- T7.6: QUOP_DUMP_COMM_INFO=1 -> dump in CWD ----------------------


@pytest.mark.mpi
class TestDumpEnabled:
    def test_dump_creates_files_in_cwd(
        self, mpi_comm, mpi_shared_work_dir, monkeypatch, dump_system_size
    ):
        """T7.6 -- With QUOP_DUMP_COMM_INFO=1, dump files appear in CWD."""
        dump_dir = str(mpi_shared_work_dir)

        monkeypatch.setenv("QUOP_DUMP_COMM_INFO", "1")
        monkeypatch.chdir(dump_dir)

        _run_qaoa(mpi_comm, dump_system_size)

        mpi_comm.Barrier()

        if mpi_comm.Get_rank() == 0:
            files = glob.glob(os.path.join(dump_dir, "quop_comm_info_*.txt"))
            # Expect at least 2 files: init + locked
            assert len(files) >= 2, f"Expected >= 2 dump files, found {len(files)}: {files}"
            # Check that each file contains the expected header
            for fpath in files:
                content = open(fpath).read()
                assert "quop_mpi_layout_t dump" in content
                assert "system_size" in content


# -- T7.7: QUOP_DUMP_COMM_INFO=<dir> -> dump in that directory --------


@pytest.mark.mpi
class TestDumpToDirectory:
    def test_dump_to_custom_directory(
        self, mpi_comm, mpi_shared_work_dir, monkeypatch, dump_system_size
    ):
        """T7.7 -- With QUOP_DUMP_COMM_INFO=<dir>, files go to that dir."""
        dump_dir = str(mpi_shared_work_dir / "quop_ci_dump")
        os.makedirs(dump_dir, exist_ok=True)
        mpi_comm.Barrier()

        monkeypatch.setenv("QUOP_DUMP_COMM_INFO", dump_dir)

        _run_qaoa(mpi_comm, dump_system_size)

        mpi_comm.Barrier()

        if mpi_comm.Get_rank() == 0:
            assert os.path.isdir(dump_dir)
            files = glob.glob(os.path.join(dump_dir, "quop_comm_info_*.txt"))
            assert len(files) >= 2, f"Expected >= 2 dump files in {dump_dir}, found {len(files)}"
            # Verify "locked" phase file contains partition table
            locked = [f for f in files if "locked" in os.path.basename(f)]
            assert len(locked) >= 1
            content = open(locked[0]).read()
            assert "Partition table" in content


# -- T7.8: Unset QUOP_DUMP_COMM_INFO -> no dump files -----------------


@pytest.mark.mpi
class TestDumpDisabled:
    def test_no_dump_when_unset(
        self, mpi_comm, mpi_shared_work_dir, monkeypatch, dump_system_size
    ):
        """T7.8 -- With QUOP_DUMP_COMM_INFO unset, no dump files are created."""
        work_dir = str(mpi_shared_work_dir)

        monkeypatch.delenv("QUOP_DUMP_COMM_INFO", raising=False)
        monkeypatch.chdir(work_dir)

        _run_qaoa(mpi_comm, dump_system_size)

        mpi_comm.Barrier()

        if mpi_comm.Get_rank() == 0:
            files = glob.glob(os.path.join(work_dir, "quop_comm_info_*.txt"))
            assert len(files) == 0, f"Expected 0 dump files, but found {len(files)}: {files}"
