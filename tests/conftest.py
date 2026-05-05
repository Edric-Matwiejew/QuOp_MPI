"""
Shared pytest fixtures for QuOp_MPI tests.

MPI tests should be run with:
    mpiexec -n <nprocs> python -m pytest tests/ --with-mpi

Use --backend to select the backend:
    mpiexec -n <nprocs> python -m pytest tests/ --with-mpi --backend mpi
    mpiexec -n <nprocs> python -m pytest tests/ --with-mpi --backend wavefront
"""

import math
import os
import signal
import sys
import tempfile
import time
import traceback
from ctypes import CDLL
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

# Set OMP_NUM_THREADS=1 to prevent OpenMP thread contention with MPI
# This must be set before any OpenMP-enabled libraries are loaded
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    _LIBC = CDLL(None)
except OSError:
    _LIBC = None


# Default timeout (seconds) for per-test MPI deadlock detection.
# Override with ``--mpi-timeout`` on the pytest command line.
_MPI_TEST_TIMEOUT: int = 120


def _flush_process_output() -> None:
    """Best-effort flush of Python and C stdio buffers for the current rank."""
    try:
        sys.stdout.flush()
    except Exception:
        pass

    try:
        sys.stderr.flush()
    except Exception:
        pass

    if _LIBC is not None:
        try:
            _LIBC.fflush(None)
        except Exception:
            pass


def _timed_barrier(comm, timeout_seconds: int = 30, label: str = "") -> None:
    """Non-blocking barrier with timeout to prevent deadlocks.

    Polls ``Ibarrier`` in a loop.  If *timeout_seconds* elapses before
    all ranks arrive, the current rank dumps its Python thread stacks to
    stderr and calls ``MPI.COMM_WORLD.Abort(1)`` so the entire job
    terminates instead of hanging forever.
    """
    req = comm.Ibarrier()
    deadline = time.monotonic() + timeout_seconds
    while True:
        # ``Request.Test`` returned ``(bool, status)`` in mpi4py < 4.0
        # but returns a plain ``bool`` from 4.0 onwards.  Accept either.
        result = req.Test()
        done = result[0] if isinstance(result, tuple) else result
        if done:
            break
        if time.monotonic() > deadline:
            rank = comm.Get_rank()
            _flush_process_output()
            lines = [
                f"\n{'='*60}",
                f"BARRIER TIMEOUT on rank {rank} after {timeout_seconds}s",
            ]
            if label:
                lines.append(f"  context: {label}")
            lines.append("Stack traces of all threads:")
            for thread_id, frame in sys._current_frames().items():
                lines.append(f"\n--- Thread {thread_id} ---")
                lines.extend(traceback.format_stack(frame))
            lines.append(f"{'='*60}\n")
            print("\n".join(lines), file=sys.stderr, flush=True)
            _flush_process_output()
            MPI.COMM_WORLD.Abort(1)
        time.sleep(0.1)

def _system_tmp_is_shared() -> bool:
    """Return True if the system temp directory is writable AND on a shared filesystem.

    Parallel HDF5 / MPI-IO requires files to be on a shared (e.g. Lustre)
    filesystem.  On Cray compute nodes ``/tmp`` is typically a node-local
    RAM-backed tmpfs -- writable, but invisible to MPI-IO and other ranks.
    We therefore check not only writability but also whether the temp dir
    lives on the same device as CWD (which is normally on scratch/Lustre).
    """
    tmp_root = tempfile.gettempdir()
    try:
        fd, probe_path = tempfile.mkstemp(prefix="quop_tmp_probe_", dir=tmp_root)
    except OSError:
        return False

    try:
        os.close(fd)
        os.unlink(probe_path)
    except OSError:
        return False

    # If /tmp is on a different device from CWD it is probably node-local.
    try:
        tmp_dev = os.stat(tmp_root).st_dev
        cwd_dev = os.stat(".").st_dev
        if tmp_dev != cwd_dev:
            return False
    except OSError:
        return False

    return True


def _configure_temp_fallback_if_needed(config):
    """
    Configure tempfile/pytest temp roots to a hidden CWD folder when system
    temp is not writable or not on a shared filesystem (needed for MPI-IO).
    """
    if _system_tmp_is_shared():
        return

    cwd = Path.cwd()
    fallback_root = cwd / ".quop_pytest_tmp"
    shared_root = fallback_root / "shared"
    pytest_base = shared_root / "pytest_basetemp"

    shared_root.mkdir(parents=True, exist_ok=True)

    fallback_tmp = str(shared_root.resolve())
    os.environ["TMPDIR"] = fallback_tmp
    os.environ["TEMP"] = fallback_tmp
    os.environ["TMP"] = fallback_tmp
    tempfile.tempdir = fallback_tmp


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add --backend and --mpi-timeout options to pytest."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["mpi", "wavefront"],
        help="Set the QuOp backend: mpi or wavefront",
    )
    parser.addoption(
        "--mpi-timeout",
        action="store",
        type=int,
        default=_MPI_TEST_TIMEOUT,
        help=(
            "Per-test timeout in seconds for MPI deadlock detection. "
            "When a test exceeds this limit the rank dumps its stack "
            "traces and calls MPI_Abort. Default: %(default)s"
        ),
    )


def pytest_configure(config):
    """Register custom markers and set backend environment variable."""
    _configure_temp_fallback_if_needed(config)

    # Set backend environment variable before any quop_mpi imports
    backend = config.getoption("--backend")
    if backend:
        os.environ["QUOP_BACKEND"] = backend

    config.addinivalue_line(
        "markers",
        "requires_nprocs(n): skip test unless at least n MPI processes are available",
    )


# Directories that should never be collected under MPI.
_SERIAL_ONLY_DIRS = ["environments", "benchmarks", "examples"]


def pytest_ignore_collect(collection_path, config):
    """Exclude serial-only test directories when running under MPI."""
    if config.getoption("--with-mpi", default=False):
        p = Path(collection_path)
        tests_dir = Path(__file__).parent
        for dirname in _SERIAL_ONLY_DIRS:
            if p == tests_dir / dirname or (tests_dir / dirname) in p.parents:
                return True
    return False


def pytest_report_header(config):
    """Add backend information to pytest header."""
    from quop_mpi import config as quop_config

    return [
        f"QuOp backend: {quop_config.backend}",
        f"MPI size: {MPI.COMM_WORLD.Get_size()}",
    ]


def pytest_runtest_setup(item):
    """Skip tests that require more MPI processes than available."""
    for marker in item.iter_markers(name="requires_nprocs"):
        required_nprocs = marker.args[0]
        actual_nprocs = MPI.COMM_WORLD.Get_size()
        if actual_nprocs < required_nprocs:
            pytest.skip(
                f"Test requires {required_nprocs} MPI processes, but only {actual_nprocs} available"
            )


# =============================================================================
# Grover's Algorithm Parameter Calculator
# =============================================================================


@dataclass
class GroverResult:
    """Result of Grover parameter calculation."""

    k_opt: int  # Optimal number of iterations
    theta: float  # Rotation angle per iteration
    success_prob: float  # Probability of measuring a marked state


@dataclass(frozen=True)
class MpiTopology:
    """Cluster topology details that are useful when sizing MPI tests."""

    world_size: int
    world_rank: int
    node_size: int
    node_rank: int
    node_count: int
    n_physical_gpus: int
    ranks_per_gpu: int
    gpu_slots_per_node: int
    total_gpu_slots: int

    @property
    def has_gpus(self) -> bool:
        """True when the backend reports at least one physical GPU."""
        return self.n_physical_gpus > 0


@dataclass(frozen=True)
class MpiSystemSizer:
    """Intent-driven helpers for choosing topology-aware system sizes."""

    topology: MpiTopology

    def power_of_two(
        self,
        *,
        base: int,
        min_per_rank: int = 0,
        min_per_node: int = 0,
        min_per_gpu: int = 0,
    ) -> int:
        """Return a power-of-two size that scales with the active topology."""
        target = max(1, int(base))
        if min_per_rank:
            target = max(target, self.topology.world_size * int(min_per_rank))
        if min_per_node:
            target = max(target, self.topology.node_count * int(min_per_node))
        if min_per_gpu and self.topology.total_gpu_slots > 0:
            target = max(target, self.topology.total_gpu_slots * int(min_per_gpu))
        return _next_power_of_two(target)

    def multiple(
        self,
        *,
        base: int = 0,
        per_rank: int = 0,
        per_node: int = 0,
        per_gpu: int = 0,
        remainder: int = 0,
    ) -> int:
        """Return a size that preserves a remainder-based partitioning intent."""
        target = max(1, int(base))
        if per_rank:
            target = max(target, self.topology.world_size * int(per_rank))
        if per_node:
            target = max(target, self.topology.node_count * int(per_node))
        if per_gpu and self.topology.total_gpu_slots > 0:
            target = max(target, self.topology.total_gpu_slots * int(per_gpu))
        return target + int(remainder)

    def prime(
        self,
        *,
        base: int,
        min_per_rank: int = 0,
        min_per_node: int = 0,
        min_per_gpu: int = 0,
    ) -> int:
        """Return a prime size at or above the requested topology-scaled floor."""
        target = max(2, int(base))
        if min_per_rank:
            target = max(target, self.topology.world_size * int(min_per_rank))
        if min_per_node:
            target = max(target, self.topology.node_count * int(min_per_node))
        if min_per_gpu and self.topology.total_gpu_slots > 0:
            target = max(target, self.topology.total_gpu_slots * int(min_per_gpu))
        return _next_prime(target)

    def below_world_power_of_two(self, *, minimum: int = 2) -> int:
        """Return the largest power of two below COMM_WORLD size."""
        minimum = max(1, int(minimum))
        world_size = self.topology.world_size
        if world_size <= minimum:
            return minimum
        value = 1 << (world_size.bit_length() - 1)
        if value >= world_size:
            value //= 2
        return max(value, minimum)

    def world_fraction(
        self,
        numerator: int,
        denominator: int,
        *,
        minimum: int = 1,
    ) -> int:
        """Return ceil(world_size * numerator / denominator)."""
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        scaled = math.ceil(self.topology.world_size * int(numerator) / int(denominator))
        return max(int(minimum), scaled)


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two that is at least *value*."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _is_prime(value: int) -> bool:
    """Return True when *value* is prime."""
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0:
        return False
    limit = int(math.isqrt(value))
    for factor in range(3, limit + 1, 2):
        if value % factor == 0:
            return False
    return True


def _next_prime(value: int) -> int:
    """Return the smallest prime that is at least *value*."""
    candidate = max(2, int(value))
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def grover_params(n_marked: int, system_size: int) -> GroverResult:
    """
    Calculate optimal Grover algorithm parameters.

    For M marked states out of N total states:
    - theta = arcsin(sqrt(M/N))
    - k_opt = floor(pi / (4 * theta))
    - success_prob = sin^2((2*k_opt + 1) * theta)
    """
    if n_marked <= 0 or system_size <= 0:
        return GroverResult(k_opt=0, theta=0.0, success_prob=0.0)

    if n_marked >= system_size:
        return GroverResult(k_opt=0, theta=np.pi / 2, success_prob=1.0)

    theta = np.arcsin(np.sqrt(n_marked / system_size))
    k_opt = max(int(np.floor(np.pi / (4 * theta))), 1)
    success_prob = np.sin((2 * k_opt + 1) * theta) ** 2

    return GroverResult(k_opt=k_opt, theta=theta, success_prob=success_prob)


# =============================================================================
# MPI Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _mpi_barrier_teardown(request):
    """Barrier on COMM_WORLD after every test to prevent cascade desync.

    When FFTW excludes ranks from a SUBCOMM, some tests leave unmatched
    collectives in flight.  A world barrier at teardown guarantees all
    ranks are synchronised before the next test begins. Flush each rank's
    Python and C stdio buffers first so the barrier marker better reflects
    the output that preceded it.

    Uses a non-blocking barrier with timeout so that if a test fails on
    one rank (leaving other ranks stuck in a collective), the job aborts
    with full stack traces instead of hanging forever.
    """
    timeout = request.config.getoption("--mpi-timeout", default=_MPI_TEST_TIMEOUT)
    yield
    if MPI.Is_initialized() and not MPI.Is_finalized():
        _flush_process_output()
        _timed_barrier(MPI.COMM_WORLD, timeout, label=f"teardown of {request.node.nodeid}")
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[BARRIER] after {request.node.nodeid}", file=sys.stderr, flush=True)
        _flush_process_output()
        _timed_barrier(MPI.COMM_WORLD, timeout, label=f"teardown-flush of {request.node.nodeid}")


def _mpi_test_timeout_handler(signum, frame):
    """Signal handler that fires when a test exceeds --mpi-timeout."""
    rank = MPI.COMM_WORLD.Get_rank() if MPI.Is_initialized() else "?"
    _flush_process_output()
    lines = [
        f"\n{'='*60}",
        f"MPI TEST TIMEOUT on rank {rank} (SIGALRM)",
        "Stack traces of all threads:",
    ]
    for thread_id, frame in sys._current_frames().items():
        lines.append(f"\n--- Thread {thread_id} ---")
        lines.extend(traceback.format_stack(frame))
    lines.append(f"{'='*60}\n")
    print("\n".join(lines), file=sys.stderr, flush=True)
    _flush_process_output()
    if MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.COMM_WORLD.Abort(1)
    sys.exit(1)


@pytest.fixture(autouse=True)
def _mpi_test_timeout(request):
    """Per-test alarm that fires if a test body deadlocks.

    Sets SIGALRM before the test and cancels it afterwards.  On timeout
    the handler prints stack traces for every thread on this rank, then
    calls ``MPI_Abort`` so the entire job terminates with useful output.
    Only active on Unix (SIGALRM is not available on Windows).
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    timeout = request.config.getoption("--mpi-timeout", default=_MPI_TEST_TIMEOUT)
    old_handler = signal.signal(signal.SIGALRM, _mpi_test_timeout_handler)
    signal.alarm(timeout)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.fixture(scope="session")
def mpi_comm():
    """MPI COMM_WORLD fixture."""
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def mpi_topology(mpi_comm):
    """Discover the world/node/GPU topology visible to the MPI test job."""
    node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        node_size = node_comm.Get_size()
        node_rank = node_comm.Get_rank()
        node_roots = mpi_comm.allreduce(1 if node_rank == 0 else 0, op=MPI.SUM)

        topo_info = {
            "n_physical_gpus": 0,
            "ranks_per_gpu": 1,
            "node_size": node_size,
        }
        try:
            from quop_mpi._utils._comm_size import QuopMpiLayout

            layout = QuopMpiLayout.create_workers(1, mpi_comm)
            try:
                topo_info.update(layout.get_topology_info())
            finally:
                layout.destroy()
        except Exception:
            # Tests should still be able to size by ranks/nodes even when
            # GPU topology discovery is unavailable in the current backend.
            pass

        gpu_slots_per_node = topo_info["n_physical_gpus"] * max(topo_info["ranks_per_gpu"], 1)
        total_gpu_slots = gpu_slots_per_node * node_roots if gpu_slots_per_node > 0 else 0

        return MpiTopology(
            world_size=mpi_comm.Get_size(),
            world_rank=mpi_comm.Get_rank(),
            node_size=node_size,
            node_rank=node_rank,
            node_count=node_roots,
            n_physical_gpus=int(topo_info["n_physical_gpus"]),
            ranks_per_gpu=int(topo_info["ranks_per_gpu"]),
            gpu_slots_per_node=int(gpu_slots_per_node),
            total_gpu_slots=int(total_gpu_slots),
        )
    finally:
        node_comm.Free()


@pytest.fixture(scope="session")
def mpi_sizing(mpi_topology):
    """Provide reusable, intent-driven system-size calculations."""
    return MpiSystemSizer(mpi_topology)


@pytest.fixture(scope="session")
def mpi_rank(mpi_comm):
    """Current MPI rank."""
    return mpi_comm.Get_rank()


@pytest.fixture(scope="session")
def mpi_size(mpi_comm):
    """Total number of MPI processes."""
    return mpi_comm.Get_size()


@pytest.fixture(scope="session")
def is_root(mpi_rank):
    """True only on rank 0."""
    return mpi_rank == 0


# =============================================================================
# System Size Fixtures
# =============================================================================


@pytest.fixture
def small_system_size(mpi_sizing):
    """Small system size for quick tests with enough room for active ranks."""
    return mpi_sizing.power_of_two(base=16, min_per_rank=1)


@pytest.fixture
def medium_system_size(mpi_sizing):
    """Medium system size that scales to keep multi-rank tests active."""
    return mpi_sizing.power_of_two(base=64, min_per_rank=1, min_per_node=16)


# =============================================================================
# Test Oracle Utilities
# =============================================================================
# These utilities create test problems with analytically known solutions,
# based on Grover's search algorithm structure. This allows verification
# of state evolution and optimization against theoretical predictions.


class _GroverOracle:
    """
    A test oracle with known optimal parameters and expected outcomes.

    Uses Grover's search structure: observables are 0 for marked states,
    1 for unmarked. This provides:
    - Known optimal parameters (gamma=pi, t=pi/N for complete graph mixing)
    - Predictable probability concentration on marked states
    - Analytically computable success probabilities

    Supports both QAOA and QWOA by providing appropriate mixer operators.

    Note: Named with underscore prefix to prevent pytest collection warnings.
    Use the `GroverOracle` or `TestOracle` alias for imports.
    """

    # Prevent pytest from trying to collect this as a test class
    __test__ = False

    def __init__(self, system_size: int, n_marked: int, seed: int = 42):
        """
        Create a test oracle.

        Parameters
        ----------
        system_size : int
            Total number of basis states (N = 2^n_qubits)
        n_marked : int
            Number of "solution" states (M)
        seed : int
            Random seed for reproducible marked state selection
        """
        self.system_size = system_size
        self.n_marked = n_marked
        self.seed = seed

        # Generate marked states reproducibly
        rng = np.random.default_rng(seed)
        self.marked_states = set(rng.choice(system_size, size=n_marked, replace=False))

        # Compute optimal Grover parameters
        self.grover_result = grover_params(n_marked, system_size)
        self.optimal_iterations = max(self.grover_result.k_opt, 1)

        # Optimal walk time for complete graph
        self.optimal_walk_time = np.pi / system_size

    def qualities_function(self):
        """
        Return a qualities function for use with qaoa/qwoa.set_qualities().

        Marked states have quality 0, unmarked have quality 1.
        The optimizer will try to minimize expectation value.
        """
        marked = self.marked_states

        def _qualities(local_i, local_i_offset):
            obs = np.ones(local_i, dtype=np.float64)
            for idx in marked:
                if local_i_offset <= idx < local_i_offset + local_i:
                    obs[idx - local_i_offset] = 0.0
            return obs

        return _qualities

    def optimal_params(self, depth: int) -> np.ndarray:
        """
        Return optimal variational parameters for given depth.

        For Grover: (gamma=pi, t=walk_time) repeated for each layer.
        """
        return np.array([np.pi, self.optimal_walk_time] * depth, dtype=np.float64)

    def theoretical_success_probability(self, n_iterations: int) -> float:
        """
        Theoretical probability on marked states after n iterations.

        P = sin^2((2k+1)*theta) where theta = arcsin(sqrt(M/N))
        """
        theta = math.asin(math.sqrt(self.n_marked / self.system_size))
        return math.sin((2 * n_iterations + 1) * theta) ** 2

    def uniform_expectation(self) -> float:
        """
        Expectation value for uniform superposition (no evolution).

        E = (N-M)/N since marked states have observable 0.
        """
        return (self.system_size - self.n_marked) / self.system_size

    def compute_marked_probability(self, full_probs: np.ndarray) -> float:
        """Compute total probability on marked states."""
        return sum(full_probs[i] for i in self.marked_states)

    @staticmethod
    def complete_graph_sparse_operator(system_size: int):
        """
        Return a sparse operator function for QAOA that generates a complete graph.

        This makes QAOA's mixing unitary equivalent to QWOA's complete graph circulant,
        enabling the same Grover-like behavior for testing both algorithms.

        The complete graph adjacency matrix A has:
        - A[i,j] = 1 for all i != j
        - A[i,i] = 0

        This is used with sparse.operator.serial to configure QAOA's mixer.

        Parameters
        ----------
        system_size : int
            Number of basis states (N)

        Returns
        -------
        callable
            A function compatible with sparse.operator.serial that returns
            a list containing the complete graph adjacency as CSR matrix.
        """
        from scipy.sparse import csr_matrix

        # Build complete graph: all-ones matrix minus identity
        # Efficient construction using COO-like arrays
        rows = []
        cols = []
        for i in range(system_size):
            for j in range(system_size):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        data = np.ones(len(rows), dtype=np.float64)
        complete_graph = csr_matrix((data, (rows, cols)), shape=(system_size, system_size))

        def _operator():
            """Return complete graph as list of CSR matrices for sparse.operator.serial."""
            return [complete_graph]

        return _operator


# Alias for backward compatibility (avoids pytest collection warning with "Test" prefix)
GroverOracle = _GroverOracle
TestOracle = _GroverOracle


@pytest.fixture
def simple_oracle():
    """A simple test oracle: N=64, M=4 marked states."""
    return _GroverOracle(system_size=64, n_marked=4, seed=42)


@pytest.fixture
def single_solution_oracle():
    """A test oracle with unique solution: N=64, M=1."""
    return _GroverOracle(system_size=64, n_marked=1, seed=123)


# =============================================================================
# Helper Functions
# =============================================================================


def mpi_barrier(comm):
    """Synchronize all MPI ranks."""
    comm.Barrier()


def assert_on_root(condition, message, comm):
    """Assert a condition, but only report from root to avoid duplicate output."""
    result = comm.gather(condition, root=0)
    if comm.Get_rank() == 0:
        assert all(result), message


def collect_to_root(value, comm):
    """Gather values from all ranks to root."""
    return comm.gather(value, root=0)


def gather_state_probabilities(alg, comm):
    """
    Gather quantum state probabilities from all ranks to root.

    Uses the algorithm's get_probabilities() method which properly
    handles the distributed state in context.state.

    Parameters
    ----------
    alg : Ansatz
        The algorithm instance after evolve_state() or execute()
    comm : MPI.Intracomm
        MPI communicator (for API compatibility, not used internally)

    Returns
    -------
    ndarray or None
        Full probability array on rank 0, None on other ranks.
    """
    return alg.get_probabilities()


# =============================================================================
# Algorithm Factory Functions
# =============================================================================


@contextmanager
def patch_qaoa_mixer(complete_graph_operator_func):
    """
    Context manager to temporarily replace QAOA's hypercube mixer with a complete graph.

    This allows testing the actual qaoa class with Grover-like behavior by
    monkey-patching the sparse.operator.hypercube function during setup.

    Parameters
    ----------
    complete_graph_operator_func : callable
        A function with the same signature as sparse.operator.hypercube
        that returns a complete graph CSR partition.

    Usage
    -----
    >>> complete_op = make_complete_graph_operator(system_size)
    >>> alg = QAOA(system_size, comm)
    >>> alg.set_qualities(oracle.qualities_function())
    >>> alg.set_depth(1)
    >>> with patch_qaoa_mixer(complete_op):
    ...     alg.setup()
    >>> # alg now uses complete graph mixer
    """
    from quop_mpi.propagator.sparse import operator as sparse_operator

    # Save original
    original_hypercube = sparse_operator.hypercube

    # Patch
    sparse_operator.hypercube = complete_graph_operator_func

    try:
        yield
    finally:
        # Restore
        sparse_operator.hypercube = original_hypercube


def make_complete_graph_operator(system_size: int):
    """
    Create a complete graph operator function compatible with sparse.operator.hypercube.

    The returned function has the same signature as hypercube() and can be
    used to replace it via patch_qaoa_mixer.

    Parameters
    ----------
    system_size : int
        Number of basis states

    Returns
    -------
    callable
        Function compatible with sparse.operator.hypercube signature
    """
    from scipy.sparse import csr_matrix

    # Pre-build the complete graph adjacency matrix
    rows = []
    cols = []
    for i in range(system_size):
        for j in range(system_size):
            if i != j:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.float64)
    complete_graph = csr_matrix((data, (rows, cols)), shape=(system_size, system_size))

    def complete_graph_operator(partition_table, MPI_COMM, *args, **kwargs):  # noqa: N803
        """
        Complete graph operator with same signature as sparse.operator.hypercube.

        Returns CSR partition for a complete graph (all-to-all connectivity).
        """
        from quop_mpi._utils._mpi import __scatter_sparse

        rank = MPI_COMM.Get_rank()

        if rank == 0:
            row_starts = [(complete_graph.tocsr()).indptr + 1]
            col_indexes = [(complete_graph.tocsr()).indices + 1]
            values = [(complete_graph.tocsr()).data]
        else:
            row_starts = None
            col_indexes = None
            values = None

        return __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM)

    return complete_graph_operator


def create_qaoa_complete_graph(system_size: int, comm, oracle: TestOracle = None):
    """
    Create a QAOA instance configured with a complete graph mixer.

    This makes QAOA equivalent to QWOA for testing purposes - both
    will implement Grover-like search on a complete graph.

    Parameters
    ----------
    system_size : int
        Number of basis states
    comm : MPI.Intracomm
        MPI communicator
    oracle : TestOracle, optional
        If provided, qualities are set from the oracle

    Returns
    -------
    qaoa
        QAOA instance with complete graph mixer instead of hypercube
    """
    from quop_mpi import Ansatz
    from quop_mpi.propagator import diagonal, sparse

    # Create base ansatz (not QAOA subclass, to avoid hardcoded hypercube)
    alg = Ansatz(system_size, comm)

    # Get the complete graph operator function
    complete_op = TestOracle.complete_graph_sparse_operator(system_size)

    # Set up unitaries: phase separator (diagonal) + mixer (complete graph sparse)
    phase_unitary = diagonal.Unitary(
        diagonal.operator.observables,
    )

    # Use operator_dict with 'kwargs' key to pass function to serial
    mixer_unitary = sparse.Unitary(
        sparse.operator.serial,
        operator_dict={"kwargs": {"function": complete_op}},
    )

    alg.set_unitaries([phase_unitary, mixer_unitary])

    if oracle is not None:
        alg.set_observables(oracle.qualities_function())

    return alg


def create_qwoa_complete_graph(system_size: int, comm, oracle: TestOracle = None):
    """
    Create a QWOA instance configured with a complete graph mixer.

    This is the standard QWOA for Grover-like search.

    Parameters
    ----------
    system_size : int
        Number of basis states
    comm : MPI.Intracomm
        MPI communicator
    oracle : TestOracle, optional
        If provided, qualities are set from the oracle

    Returns
    -------
    qwoa
        QWOA instance with complete graph circulant mixer
    """
    from quop_mpi.algorithm.combinatorial import QWOA

    alg = QWOA(system_size, comm)

    if oracle is not None:
        alg.set_qualities(oracle.qualities_function())

    return alg
