"""
Tests for the MPI scope decorator contract.

Verifies that:
1. Every public method on Ansatz (and its mixins) is decorated with @scope.
2. The ``_returns`` metadata is valid on all decorated methods.
3. Excluded ranks receive ``None`` from subcomm-scoped methods.
4. ``returns="all"`` methods give the same value on all subcomm ranks.
5. ``returns="root"`` methods give non-None only on subcomm rank 0.

Run with:
    mpiexec -n 4 python -m pytest tests/mpi/test_scope.py -v --with-mpi --backend mpi
"""

import inspect

import numpy as np
import pytest

from tests.conftest import TestOracle

from quop_mpi._scope import get_returns, get_scope


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps scope tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the shared-return tests while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Methods inherited from Bindable that live outside the scope decorator
# contract (Bindable is shared with Unitary, which has no subcomms).
_BINDABLE_METHODS = {"get_bindable_attributes", "print_bindable_attributes"}

# Names that are public by convention but are not user-facing API -- they are
# internal hooks called only by the framework itself.
_INTERNAL_HOOKS = set()


def _public_methods(cls):
    """Yield ``(name, method)`` for every public, non-dunder instance method
    on *cls* (including inherited ones), excluding Bindable methods and
    properties.
    """
    for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in _BINDABLE_METHODS:
            continue
        # Skip properties (they appear as data descriptors on the class)
        if isinstance(inspect.getattr_static(cls, name, None), property):
            continue
        yield name, obj


# ---------------------------------------------------------------------------
# 1. Contract: every public method must carry a _scope attribute
# ---------------------------------------------------------------------------


@pytest.mark.mpi
class TestScopeContract:
    """Every public Ansatz method must be decorated with @scope."""

    def test_all_public_methods_have_scope(self, mpi_comm):
        """Meta-test: scan Ansatz class and assert _scope is present."""
        from quop_mpi.ansatz import Ansatz

        missing = []
        for name, method in _public_methods(Ansatz):
            if not hasattr(method, "_scope"):
                missing.append(name)

        assert missing == [], f"Public methods missing @scope decorator: {missing}"

    def test_scope_values_are_valid(self, mpi_comm):
        """Every decorated method must use a recognised scope name."""
        from quop_mpi.ansatz import Ansatz

        valid_scopes = {"world", "subcomm", "jaccomm"}

        for name, method in _public_methods(Ansatz):
            scope_val = get_scope(method)
            assert scope_val in valid_scopes, f"{name} has invalid scope {scope_val!r}"

    def test_returns_values_are_valid(self, mpi_comm):
        """Every decorated method must have a valid _returns classifier."""
        from quop_mpi.ansatz import Ansatz

        valid_returns = {"none", "all", "root"}

        for name, method in _public_methods(Ansatz):
            returns_val = get_returns(method)
            assert returns_val in valid_returns, f"{name} has invalid returns {returns_val!r}"


# ---------------------------------------------------------------------------
# 2. Metadata correctness: verify _returns matches known expectations
# ---------------------------------------------------------------------------

# Ground-truth table of expected (scope, returns) for every public method.
_EXPECTED = {
    # Ansatz setters -- world / none
    "set_parameter_map": ("world", "none"),
    "set_unitaries": ("world", "none"),
    "set_observables": ("world", "none"),
    "set_optimiser": ("world", "none"),
    "set_depth": ("world", "none"),
    "set_initial_state": ("world", "none"),
    "set_seed": ("world", "none"),
    "set_objective": ("world", "none"),
    # Ansatz print helpers -- world / none
    "print_all_bindable_attributes": ("world", "none"),
    "print_result": ("world", "none"),
    "print_optimiser_result": ("world", "none"),
    # Ansatz lifecycle -- world / none
    "setup": ("world", "none"),
    "prepare": ("world", "none"),
    "destroy": ("world", "none"),
    "evolve_state": ("world", "none"),
    "execute": ("world", "none"),
    # Ansatz evaluation -- subcomm
    "get_expectation_value": ("subcomm", "all"),
    "get_state_norm": ("subcomm", "all"),
    "evaluate": ("subcomm", "all"),
    "gen_initial_params": ("subcomm", "all"),
    "objective": ("subcomm", "root"),
    "get_final_state": ("subcomm", "root"),
    "get_probabilities": ("subcomm", "root"),
    # Mixin: Sampling
    "set_sampling": ("world", "none"),
    "unset_sampling": ("world", "none"),
    # Mixin: Logging
    "set_log": ("world", "none"),
    "save": ("subcomm", "none"),
    # Mixin: Benchmark
    "benchmark": ("world", "none"),
    # Mixin: Jacobian
    "set_parallel_jacobian": ("world", "none"),
}


@pytest.mark.mpi
class TestScopeMetadata:
    """Verify the scope/returns metadata matches the ground-truth table."""

    def test_metadata_matches_expected(self, mpi_comm):
        """Each method's (scope, returns) must match our expectation table."""
        from quop_mpi.ansatz import Ansatz

        errors = []
        for name, method in _public_methods(Ansatz):
            if name not in _EXPECTED:
                errors.append(f"{name}: not in expectation table")
                continue
            expected_scope, expected_returns = _EXPECTED[name]
            actual_scope = get_scope(method)
            actual_returns = get_returns(method)
            if actual_scope != expected_scope:
                errors.append(f"{name}: scope expected {expected_scope!r}, got {actual_scope!r}")
            if actual_returns != expected_returns:
                errors.append(
                    f"{name}: returns expected {expected_returns!r}, got {actual_returns!r}"
                )

        assert errors == [], "\n".join(errors)

    def test_expectation_table_is_complete(self, mpi_comm):
        """The expectation table must cover every public method (no extras, no gaps)."""
        from quop_mpi.ansatz import Ansatz

        actual_names = {name for name, _ in _public_methods(Ansatz)}

        missing_from_table = actual_names - set(_EXPECTED)
        extra_in_table = set(_EXPECTED) - actual_names

        errors = []
        if missing_from_table:
            errors.append(f"Methods not in expectation table: {missing_from_table}")
        if extra_in_table:
            errors.append(f"Extra entries in expectation table: {extra_in_table}")

        assert errors == [], "\n".join(errors)


# ---------------------------------------------------------------------------
# 3. Excluded-rank contract: subcomm methods return None on excluded ranks
# ---------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.requires_nprocs(4)
class TestExcludedRankReturnsNone:
    """With system_size=2 and 4 ranks, at least one rank is excluded.
    Subcomm-scoped methods must return None on that rank."""

    def _make_alg(self, comm):
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle = TestOracle(system_size=2, n_marked=1)
        alg = QAOA(2, comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        return alg, oracle

    def test_evaluate_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.prepare()
        params = np.array([1.0, 0.5])
        result = alg.evaluate(params)
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()

    def test_get_expectation_value_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.evolve_state(np.array([1.0, 0.5]))
        result = alg.get_expectation_value()
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()

    def test_get_final_state_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.evolve_state(np.array([1.0, 0.5]))
        result = alg.get_final_state()
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()

    def test_get_probabilities_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.evolve_state(np.array([1.0, 0.5]))
        result = alg.get_probabilities()
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()

    def test_gen_initial_params_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.prepare()
        result = alg.gen_initial_params()
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()

    def test_objective_returns_none_on_excluded(self, mpi_comm):
        alg, oracle = self._make_alg(mpi_comm)
        alg.prepare()
        result = alg.objective(np.array([1.0, 0.5]))
        if not alg.subcomms.in_subcomm():
            assert result is None
        alg.destroy()


# ---------------------------------------------------------------------------
# 4. returns="all" -- all subcomm ranks get the same value
# ---------------------------------------------------------------------------


@pytest.mark.mpi
class TestReturnsAll:
    """Methods with returns='all' must give the same value on every subcomm rank."""

    def test_evaluate_same_on_all_subcomm_ranks(self, mpi_comm, simple_oracle):
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)
        result = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            assert result is not None
            # Gather to subcomm root and compare
            all_results = alg.subcomms.SUBCOMM.gather(result, root=0)
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                for r in all_results:
                    assert np.isclose(
                        r, all_results[0]
                    ), f"evaluate() returned different values across ranks: {all_results}"
        alg.destroy()

    def test_get_expectation_value_same_on_all_subcomm_ranks(self, mpi_comm, simple_oracle):
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        result = alg.get_expectation_value()

        if alg.subcomms.in_subcomm():
            assert result is not None
            all_results = alg.subcomms.SUBCOMM.gather(result, root=0)
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                for r in all_results:
                    assert np.isclose(
                        r, all_results[0]
                    ), f"get_expectation_value() differs across ranks: {all_results}"
        alg.destroy()

    def test_gen_initial_params_same_on_all_subcomm_ranks(self, mpi_comm, simple_oracle):
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        result = alg.gen_initial_params()

        if alg.subcomms.in_subcomm():
            assert result is not None
            all_results = alg.subcomms.SUBCOMM.gather(result.tolist(), root=0)
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                for r in all_results:
                    assert np.allclose(
                        r, all_results[0]
                    ), "gen_initial_params() differs across ranks"
        alg.destroy()


# ---------------------------------------------------------------------------
# 5. returns="root" -- only subcomm rank 0 gets non-None
# ---------------------------------------------------------------------------


@pytest.mark.mpi
class TestReturnsRoot:
    """Methods with returns='root' must give non-None only on subcomm rank 0."""

    def test_get_final_state_only_on_root(self, mpi_comm, simple_oracle):
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        result = alg.get_final_state()

        if alg.subcomms.in_subcomm():
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                assert result is not None
                assert isinstance(result, np.ndarray)
                assert len(result) == simple_oracle.system_size
            else:
                assert result is None
        else:
            assert result is None

        alg.destroy()

    def test_get_probabilities_only_on_root(self, mpi_comm, simple_oracle):
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        result = alg.get_probabilities()

        if alg.subcomms.in_subcomm():
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                assert result is not None
                assert isinstance(result, np.ndarray)
                assert len(result) == simple_oracle.system_size
                # Probabilities should sum to ~1
                assert np.isclose(np.sum(result), 1.0, atol=1e-8)
            else:
                assert result is None
        else:
            assert result is None

        alg.destroy()
