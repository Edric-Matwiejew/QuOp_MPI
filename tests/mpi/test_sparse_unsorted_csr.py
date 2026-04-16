import numpy as np
import pytest
from scipy.sparse import csr_matrix


def _build_cycle_graph_csr(system_size):
    """Build a cycle graph adjacency matrix as a scipy CSR matrix.

    Each node i is connected to (i-1) % N and (i+1) % N.  The resulting
    CSR matrix has sorted column indices by construction.
    """
    rows = []
    cols = []
    for i in range(system_size):
        rows.append(i)
        cols.append((i - 1) % system_size)
        rows.append(i)
        cols.append((i + 1) % system_size)

    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(system_size, system_size))


def _unshuffle_csr(matrix):
    """Return a copy of a CSR matrix with columns reversed within each row.

    This deliberately breaks the sorted-column invariant while keeping
    the logical matrix identical.
    """
    m = matrix.copy()
    for i in range(m.shape[0]):
        lo = m.indptr[i]
        hi = m.indptr[i + 1]
        m.indices[lo:hi] = m.indices[lo:hi][::-1]
        m.data[lo:hi] = m.data[lo:hi][::-1]
    return m


def _make_sorted_operator(csr):
    """Operator function returning sorted CSR (reference)."""
    def _op():
        return [csr]
    return _op


def _make_unsorted_operator(csr):
    """Operator function returning unsorted CSR."""
    unsorted = _unshuffle_csr(csr)
    def _op():
        return [unsorted]
    return _op


def _create_ansatz(system_size, mpi_comm, operator_fn):
    """Create an Ansatz with a diagonal phase + custom sparse mixer."""
    from quop_mpi import Ansatz
    from quop_mpi.propagator import diagonal, sparse

    alg = Ansatz(system_size, mpi_comm)

    phase = diagonal.Unitary(diagonal.operator.observables)
    mixer = sparse.Unitary(
        sparse.operator.serial,
        operator_dict={"kwargs": {"function": operator_fn}},
    )

    alg.set_unitaries([phase, mixer])

    def qualities(local_i, local_i_offset):
        return np.sin(
            np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        )

    alg.set_observables(qualities)
    alg.set_depth(1)
    return alg


@pytest.fixture
def cycle_system_size(mpi_sizing):
    """Power-of-two size for the cycle graph tests."""
    return mpi_sizing.power_of_two(base=16, min_per_rank=2)


@pytest.mark.mpi
class TestSparseUnsortedCSR:
    """Regression: unsorted CSR columns must not produce NaN."""

    def test_unsorted_csr_result_is_finite(self, mpi_comm, cycle_system_size):
        """T1: evolve_state with unsorted columns must not produce NaN."""
        csr = _build_cycle_graph_csr(cycle_system_size)
        op = _make_unsorted_operator(csr)
        alg = _create_ansatz(cycle_system_size, mpi_comm, op)

        params = np.array([0.3, 0.7])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert np.all(np.isfinite(probs)), (
                "Sparse propagator produced NaN/Inf with unsorted CSR columns"
            )
        alg.destroy()

    def test_unsorted_csr_preserves_normalization(self, mpi_comm, cycle_system_size):
        """T2: probability must sum to 1 after evolution with unsorted CSR."""
        csr = _build_cycle_graph_csr(cycle_system_size)
        op = _make_unsorted_operator(csr)
        alg = _create_ansatz(cycle_system_size, mpi_comm, op)

        params = np.array([0.3, 0.7])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = float(np.sum(probs, dtype=np.float64))
            np.testing.assert_allclose(
                total,
                1.0,
                atol=1e-8,
                err_msg=f"Normalization violated: total probability = {total}",
            )
        alg.destroy()

    def test_unsorted_matches_sorted(self, mpi_comm, cycle_system_size):
        """T3: unsorted and sorted CSR must produce identical probabilities."""
        csr = _build_cycle_graph_csr(cycle_system_size)

        sorted_op = _make_sorted_operator(csr)
        alg_sorted = _create_ansatz(cycle_system_size, mpi_comm, sorted_op)
        params = np.array([0.3, 0.7])
        alg_sorted.evolve_state(params)
        probs_sorted = alg_sorted.get_probabilities()

        unsorted_op = _make_unsorted_operator(csr)
        alg_unsorted = _create_ansatz(cycle_system_size, mpi_comm, unsorted_op)
        alg_unsorted.evolve_state(params)
        probs_unsorted = alg_unsorted.get_probabilities()

        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(
                probs_unsorted,
                probs_sorted,
                atol=1e-12,
                err_msg="Unsorted CSR produced different results from sorted CSR",
            )

        alg_sorted.destroy()
        alg_unsorted.destroy()
