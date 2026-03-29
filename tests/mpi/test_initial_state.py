"""
Tests for Ansatz initial state functionality: set_initial_state() and default behavior.

The initial state defines the quantum state before any unitary evolution.
Default is an equal superposition over all basis states.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_initial_state.py -v --with-mpi
"""

import numpy as np
import pytest
from mpi4py import MPI

from tests.conftest import TestOracle


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps initial-state tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


def _single_eval_optimiser(fun, x0, **_kwargs):
    """Evaluate exactly one parameter vector so execute() becomes deterministic."""
    x = np.asarray(x0, dtype=np.float64)
    return {
        "x": x.copy(),
        "fun": float(fun(x)),
        "nfev": 1,
        "nit": 0,
        "success": True,
    }


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the initial-state oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.mark.mpi
class TestDefaultInitialState:
    """Tests for default initial state behavior (equal superposition)."""

    def test_default_initial_state_is_equal_superposition(self, mpi_comm, simple_oracle):
        """Verify default initial state is uniform superposition."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        # After __pre, ansatz_initial_state should be set
        if alg.subcomms.in_subcomm():
            assert alg.ansatz_initial_state is not None

            # Each element should have amplitude 1/sqrt(N)
            expected_amplitude = 1.0 / np.sqrt(simple_oracle.system_size)

            # Check local portion
            for val in alg.ansatz_initial_state:
                assert np.isclose(np.abs(val), expected_amplitude, atol=1e-10)

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()

    def test_default_initial_state_is_normalized(self, mpi_comm, simple_oracle):
        """Verify default initial state has unit norm."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        if alg.subcomms.in_subcomm():
            # Compute local norm squared
            local_norm_sq = np.sum(np.abs(alg.ansatz_initial_state) ** 2)

            # Reduce across all ranks in subcomm
            total_norm_sq = alg.subcomms.SUBCOMM.allreduce(local_norm_sq, op=MPI.SUM)

            assert np.isclose(
                total_norm_sq, 1.0, atol=1e-10
            ), f"Initial state norm should be 1.0, got {np.sqrt(total_norm_sq)}"

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()

    def test_default_uses_equal_function(self, mpi_comm, simple_oracle):
        """Verify default initial state uses quop_mpi.state.equal function."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        if alg.subcomms.in_subcomm():
            # After __pre, the initial_state_function should be set
            # Compare with manually calling equal()
            expected = equal(simple_oracle.system_size, alg.local_i)

            assert np.allclose(alg.ansatz_initial_state, expected)

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()


@pytest.mark.mpi
class TestSetInitialStateBasic:
    """Basic tests for set_initial_state() method."""

    def test_set_initial_state_sets_dirty_flag(self, mpi_comm, simple_oracle):
        """Verify set_initial_state() sets the INITIAL_STATE dirty flag."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.ansatz import _Dirty
        from quop_mpi.state import equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Explicitly set initial state
        alg.set_initial_state(equal)

        assert alg._dirty & _Dirty.INITIAL_STATE

        alg.destroy()

    def test_set_initial_state_stores_function(self, mpi_comm, simple_oracle):
        """Verify set_initial_state() stores the function."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.set_initial_state(equal)

        # The function should be stored (before parsing)
        assert alg.initial_state_function is equal

        alg.destroy()


@pytest.mark.mpi
class TestEqualInitialState:
    """Tests for the equal (uniform) initial state function."""

    def test_equal_state_with_qaoa(self, mpi_comm, simple_oracle):
        """Verify equal initial state works with QAOA."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(equal)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_equal_state_with_qwoa(self, mpi_comm, simple_oracle):
        """Verify equal initial state works with QWOA."""
        from quop_mpi.algorithm.combinatorial import QWOA
        from quop_mpi.state import equal

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(equal)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestBasisInitialState:
    """Tests for the basis initial state function."""

    def test_basis_state_single_state(self, mpi_comm, simple_oracle):
        """Verify basis state with single basis state generates valid state."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Start in basis state |0>
        alg.set_initial_state(
            basis, initial_state_dict={"args": [], "kwargs": {"basis_states": [0]}}
        )

        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        if alg.subcomms.in_subcomm():
            # Initial state should be set
            assert alg.ansatz_initial_state is not None

            # Verify state has correct size
            assert len(alg.ansatz_initial_state) == alg.local_i

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()

    def test_basis_state_multiple_states(self, mpi_comm, simple_oracle):
        """Verify basis state with multiple basis states is set correctly."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Start in equal superposition of |0> and |1>
        basis_states = [0, 1]
        alg.set_initial_state(
            basis,
            initial_state_dict={"args": [], "kwargs": {"basis_states": basis_states}},
        )

        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        if alg.subcomms.in_subcomm():
            # Verify state was generated
            assert alg.ansatz_initial_state is not None
            assert len(alg.ansatz_initial_state) == alg.local_i

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()

    def test_basis_state_executes_successfully(self, mpi_comm, simple_oracle):
        """Verify basis initial state allows successful execution."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(
            basis, initial_state_dict={"args": [], "kwargs": {"basis_states": [0]}}
        )

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestCustomInitialState:
    """Tests for custom initial state functions."""

    def test_custom_function_called(self, mpi_comm, simple_oracle):
        """Verify custom initial state function is called."""
        from quop_mpi.algorithm.combinatorial import QAOA

        call_count = [0]

        def custom_initial_state(system_size, local_i):
            call_count[0] += 1
            # Return uniform superposition
            return np.full(local_i, 1.0 / np.sqrt(system_size), dtype=np.complex128)

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(custom_initial_state)

        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        # Custom function should have been called (on ranks in subcomm)
        total_calls = mpi_comm.reduce(call_count[0], op=MPI.SUM, root=0)

        if mpi_comm.Get_rank() == 0:
            assert total_calls > 0

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()

    def test_custom_localized_state(self, mpi_comm, simple_oracle):
        """Test custom function that creates a localized initial state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        def localized_state(system_size, local_i, local_i_offset):
            """Create state localized at index 0."""
            state = np.zeros(local_i, dtype=np.complex128)
            if local_i_offset == 0 and local_i > 0:
                state[0] = 1.0
            return state

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(localized_state)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_custom_state_with_parameters(self, mpi_comm, simple_oracle):
        """Test custom function with additional parameters via FunctionDict."""
        from quop_mpi.algorithm.combinatorial import QAOA

        def parameterized_state(system_size, local_i, amplitude_scale=1.0):
            """Create state with scaled amplitude."""
            amp = amplitude_scale / np.sqrt(system_size)
            return np.full(local_i, amp, dtype=np.complex128)

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Pass amplitude_scale via initial_state_dict
        alg.set_initial_state(
            parameterized_state,
            initial_state_dict={"args": [], "kwargs": {"amplitude_scale": 1.0}},
        )

        alg.setup()
        alg._Ansatz__pre()  # __pre() generates initial state

        if alg.subcomms.in_subcomm():
            expected_amp = 1.0 / np.sqrt(simple_oracle.system_size)
            assert np.allclose(np.abs(alg.ansatz_initial_state), expected_amp)

        alg._Ansatz__post()  # Finalize execution phase
        alg.destroy()


@pytest.mark.mpi
class TestInitialStateEvolution:
    """Tests for how initial state affects algorithm evolution."""

    def test_zero_params_preserves_equal_state(self, mpi_comm, simple_oracle):
        """Verify zero parameters keep state close to equal superposition."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(equal)

        # Evolve with zero parameters
        zero_params = np.zeros(2)  # gamma=0, beta=0
        alg.evolve_state(zero_params)

        if alg.subcomms.in_subcomm():
            # With zero params, state should be unchanged (still uniform)
            expected_amplitude = 1.0 / np.sqrt(simple_oracle.system_size)

            # Check that amplitudes are still uniform
            for val in alg.context.state[: alg.local_i]:
                assert np.isclose(np.abs(val), expected_amplitude, atol=1e-5)

        alg.destroy()

    def test_initial_state_affects_expectation(self, mpi_comm, simple_oracle):
        """Verify different initial states give different expectation values."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis, equal

        marked_state = min(simple_oracle.marked_states)
        params = np.zeros(2, dtype=np.float64)

        # Test with equal superposition
        alg_equal = QAOA(simple_oracle.system_size, mpi_comm)
        alg_equal.set_qualities(simple_oracle.qualities_function())
        alg_equal.set_depth(1)
        alg_equal.set_initial_state(equal)

        alg_equal.evolve_state(params)
        exp_equal = alg_equal.get_expectation_value()

        alg_equal.destroy()

        # Test with a marked basis state so the expected value is known exactly.
        alg_basis = QAOA(simple_oracle.system_size, mpi_comm)
        alg_basis.set_qualities(simple_oracle.qualities_function())
        alg_basis.set_depth(1)
        alg_basis.set_initial_state(
            basis,
            initial_state_dict={"args": [], "kwargs": {"basis_states": [marked_state]}},
        )

        alg_basis.evolve_state(params)
        exp_basis = alg_basis.get_expectation_value()

        if alg_basis.subcomms.in_rootcomm():
            assert np.isclose(exp_equal, simple_oracle.uniform_expectation(), atol=1e-10)
            assert np.isclose(exp_basis, 0.0, atol=1e-10)
            assert not np.isclose(exp_equal, exp_basis, atol=1e-10)

        alg_basis.destroy()


@pytest.mark.mpi
class TestInitialStateWithDifferentAlgorithms:
    """Test initial state works across different algorithm types."""

    def test_qaoa_with_custom_initial_state(self, mpi_comm, simple_oracle):
        """Verify QAOA works with custom initial state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        def half_amplitude_state(system_size, local_i):
            """Half of normal amplitude (not normalized)."""
            return np.full(local_i, 0.5 / np.sqrt(system_size), dtype=np.complex128)

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(half_amplitude_state)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_qwoa_with_basis_initial_state(self, mpi_comm, simple_oracle):
        """Verify QWOA works with basis initial state."""
        from quop_mpi.algorithm.combinatorial import QWOA
        from quop_mpi.state import basis

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(
            basis,
            initial_state_dict={"args": [], "kwargs": {"basis_states": [0, 1, 2]}},
        )

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestInitialStateEdgeCases:
    """Tests for edge cases in initial state handling."""

    def test_reinitialize_initial_state(self, mpi_comm, simple_oracle):
        """Test setting initial state multiple times."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis, equal

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # First set to basis
        alg.set_initial_state(
            basis, initial_state_dict={"args": [], "kwargs": {"basis_states": [0]}}
        )

        # Then change to equal
        alg.set_initial_state(equal)

        # Should use equal (the last one set)
        assert alg.initial_state_function is equal

        alg.destroy()

    def test_initial_state_after_execute(self, mpi_comm, simple_oracle):
        """Test that initial state can be changed and re-executed."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis, equal

        marked_state = min(simple_oracle.marked_states)
        params = np.zeros(2, dtype=np.float64)

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_initial_state(equal)
        alg.set_optimiser(_single_eval_optimiser, {}, ["fun", "x", "nfev", "success"])

        alg.execute(params)

        if alg.subcomms.in_rootcomm():
            result1 = alg.result["fun"]

        # Change initial state
        alg.set_initial_state(
            basis,
            initial_state_dict={"args": [], "kwargs": {"basis_states": [marked_state]}},
        )

        # Re-execute
        alg.execute(params)

        if alg.subcomms.in_rootcomm():
            result2 = alg.result["fun"]
            assert np.isclose(result1, simple_oracle.uniform_expectation(), atol=1e-10)
            assert np.isclose(result2, 0.0, atol=1e-10)
            assert not np.isclose(result1, result2, atol=1e-10)
            assert alg.result["nfev"] == 1

        alg.destroy()

    def test_initial_state_with_deeper_circuit(self, mpi_comm, simple_oracle):
        """Test custom initial state with deeper circuit."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.state import basis

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(3)  # Deeper circuit
        alg.set_initial_state(
            basis, initial_state_dict={"args": [], "kwargs": {"basis_states": [0]}}
        )

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            # Should have 6 parameters for depth=3 (2 per layer)
            assert len(alg.result["x"]) == 6

        alg.destroy()
