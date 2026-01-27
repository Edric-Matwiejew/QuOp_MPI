"""
Unit tests for the quop_mpi.__utils.__nlopt_wrap module.

This module tests the NLopt wrapper which provides a scipy-like interface
to the NLopt optimization library.
"""
import pytest
import numpy as np
from scipy.optimize import rosen, rosen_der

# Skip all tests in this module if nlopt is not available
nlopt = pytest.importorskip("nlopt")


class TestNloptEnumLookup:
    """Tests for get_nlopt_enum function."""

    def test_get_nlopt_enum_uppercase(self):
        """Look up algorithm by uppercase name."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('LN_NELDERMEAD')
        assert result == nlopt.LN_NELDERMEAD

    def test_get_nlopt_enum_lowercase(self):
        """Look up algorithm by lowercase name."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('ln_neldermead')
        assert result == nlopt.LN_NELDERMEAD

    def test_get_nlopt_enum_mixed_case(self):
        """Look up algorithm by mixed case name."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('ln_NelderMead')
        assert result == nlopt.LN_NELDERMEAD

    def test_get_nlopt_enum_none_returns_default(self):
        """None returns default algorithm (LN_BOBYQA)."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum(None)
        assert result == nlopt.LN_BOBYQA

    def test_get_nlopt_enum_unknown_returns_default(self):
        """Unknown algorithm returns default with warning."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        with pytest.warns(RuntimeWarning, match="could not be found"):
            result = get_nlopt_enum('foobar')
        
        assert result == nlopt.LN_BOBYQA

    def test_get_nlopt_enum_bobyqa(self):
        """Look up LN_BOBYQA."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('LN_BOBYQA')
        assert result == nlopt.LN_BOBYQA

    def test_get_nlopt_enum_cobyla(self):
        """Look up LN_COBYLA."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('LN_COBYLA')
        assert result == nlopt.LN_COBYLA

    def test_get_nlopt_enum_lbfgs(self):
        """Look up LD_LBFGS."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_enum
        
        result = get_nlopt_enum('LD_LBFGS')
        assert result == nlopt.LD_LBFGS


class TestNormalizeBound:
    """Tests for normalize_bound function."""

    def test_normalize_bound_both_finite(self):
        """Both bounds finite - no change."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bound
        
        result = normalize_bound((2.6, 7.2))
        assert result == (2.6, 7.2)

    def test_normalize_bound_lower_none(self):
        """Lower bound None becomes -inf."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bound
        
        result = normalize_bound((None, 7.2))
        assert result == (-float('inf'), 7.2)

    def test_normalize_bound_upper_none(self):
        """Upper bound None becomes +inf."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bound
        
        result = normalize_bound((2.6, None))
        assert result == (2.6, float('inf'))

    def test_normalize_bound_both_none(self):
        """Both bounds None become -inf, +inf."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bound
        
        result = normalize_bound((None, None))
        assert result == (-float('inf'), float('inf'))

    def test_normalize_bound_idempotent(self):
        """Operation is idempotent."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bound
        
        result = normalize_bound((-float("inf"), float("inf")))
        assert result == (-float('inf'), float('inf'))


class TestNormalizeBounds:
    """Tests for normalize_bounds function."""

    def test_normalize_bounds_multiple(self):
        """Normalize multiple bounds."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bounds
        
        bounds = [(2.6, 7.2), (None, 2), (3.14, None), (None, None)]
        result = list(normalize_bounds(bounds))
        
        expected = [(2.6, 7.2), (-float('inf'), 2), (3.14, float('inf')), 
                    (-float('inf'), float('inf'))]
        assert result == expected

    def test_normalize_bounds_empty(self):
        """Empty bounds list."""
        from quop_mpi.__utils.__nlopt_wrap import normalize_bounds
        
        result = list(normalize_bounds([]))
        assert result == []


class TestNloptMessage:
    """Tests for get_nlopt_message function."""

    def test_get_nlopt_message_success(self):
        """Message for SUCCESS."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_message
        
        result = get_nlopt_message(nlopt.SUCCESS)
        assert result == 'Success'

    def test_get_nlopt_message_ftol_reached(self):
        """Message for FTOL_REACHED."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_message
        
        result = get_nlopt_message(nlopt.FTOL_REACHED)
        assert 'ftol' in result.lower()

    def test_get_nlopt_message_xtol_reached(self):
        """Message for XTOL_REACHED."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_message
        
        result = get_nlopt_message(nlopt.XTOL_REACHED)
        assert 'xtol' in result.lower()

    def test_get_nlopt_message_maxeval_reached(self):
        """Message for MAXEVAL_REACHED."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_message
        
        result = get_nlopt_message(nlopt.MAXEVAL_REACHED)
        assert 'maxeval' in result.lower()

    def test_get_nlopt_message_invalid_args(self):
        """Message for INVALID_ARGS."""
        from quop_mpi.__utils.__nlopt_wrap import get_nlopt_message
        
        result = get_nlopt_message(nlopt.INVALID_ARGS)
        assert 'invalid' in result.lower()


class TestMakeNloptFun:
    """Tests for make_nlopt_fun function."""

    def test_make_nlopt_fun_no_gradient(self):
        """Create function without gradient."""
        from quop_mpi.__utils.__nlopt_wrap import make_nlopt_fun
        
        def simple_fun(x):
            return np.sum(x**2)
        
        nlopt_fun = make_nlopt_fun(simple_fun, jac=False)
        
        x = np.array([1.0, 2.0, 3.0])
        grad = np.array([])
        
        result = nlopt_fun(x, grad)
        expected = 14.0  # 1 + 4 + 9
        
        assert result == expected

    def test_make_nlopt_fun_with_callable_gradient(self):
        """Create function with callable gradient."""
        from quop_mpi.__utils.__nlopt_wrap import make_nlopt_fun
        
        def simple_fun(x):
            return np.sum(x**2)
        
        def simple_grad(x):
            return 2 * x
        
        nlopt_fun = make_nlopt_fun(simple_fun, jac=simple_grad)
        
        x = np.array([1.0, 2.0, 3.0])
        grad = np.zeros(3)
        
        result = nlopt_fun(x, grad)
        
        assert result == 14.0
        np.testing.assert_array_equal(grad, np.array([2.0, 4.0, 6.0]))

    def test_make_nlopt_fun_with_tuple_return(self):
        """Function returns (value, gradient) tuple."""
        from quop_mpi.__utils.__nlopt_wrap import make_nlopt_fun
        
        def fun_with_grad(x):
            return np.sum(x**2), 2 * x
        
        nlopt_fun = make_nlopt_fun(fun_with_grad, jac=True)
        
        x = np.array([1.0, 2.0])
        grad = np.zeros(2)
        
        result = nlopt_fun(x, grad)
        
        assert result == 5.0
        np.testing.assert_array_equal(grad, np.array([2.0, 4.0]))

    def test_make_nlopt_fun_with_args(self):
        """Function with additional arguments."""
        from quop_mpi.__utils.__nlopt_wrap import make_nlopt_fun
        
        def scaled_fun(x, scale):
            return scale * np.sum(x**2)
        
        nlopt_fun = make_nlopt_fun(scaled_fun, jac=False, args=(2.0,))
        
        x = np.array([1.0, 2.0])
        grad = np.array([])
        
        result = nlopt_fun(x, grad)
        
        assert result == 10.0  # 2 * (1 + 4)

    def test_make_nlopt_fun_records_path(self):
        """Function records optimization path."""
        from quop_mpi.__utils.__nlopt_wrap import make_nlopt_fun
        
        def simple_fun(x):
            return np.sum(x**2)
        
        path = []
        nlopt_fun = make_nlopt_fun(simple_fun, jac=False, path=path)
        
        x1 = np.array([1.0, 2.0])
        x2 = np.array([0.5, 1.0])
        grad = np.array([])
        
        nlopt_fun(x1, grad)
        nlopt_fun(x2, grad)
        
        assert len(path) == 2
        np.testing.assert_array_equal(path[0], x1)
        np.testing.assert_array_equal(path[1], x2)


class TestMinimize:
    """Tests for the main minimize function."""

    def test_minimize_rosenbrock_derivative_free(self):
        """Minimize Rosenbrock function with derivative-free method."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        x0 = np.array([0.5, 0.5])
        
        res = minimize(rosen, x0, method='LN_BOBYQA', maxeval=1000)
        
        # Should converge reasonably close to (1, 1)
        # Note: success can be False due to roundoff limits even when converged
        assert res.x is not None
        assert res.fun is not None
        # Check we got close to the minimum
        np.testing.assert_allclose(res.x, [1.0, 1.0], atol=0.1)

    def test_minimize_rosenbrock_with_gradient(self):
        """Minimize Rosenbrock function with gradient-based method."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        
        res = minimize(rosen, x0, method='LD_LBFGS', jac=rosen_der)
        
        assert res.success
        assert res.message == 'Success'
        np.testing.assert_allclose(res.fun, 0, atol=1e-5)
        np.testing.assert_allclose(res.x, np.ones(5), atol=1e-3)

    def test_minimize_with_ftol(self):
        """Minimize with ftol_abs stopping criterion."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        
        res = minimize(rosen, x0, method='LD_LBFGS', jac=rosen_der, ftol_abs=1e-5)
        
        assert res.success
        assert 'ftol' in res.message.lower() or res.message == 'Success'

    def test_minimize_returns_optimize_result(self):
        """Minimize returns OptimizeResult-like object."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        res = minimize(quadratic, x0, method='LN_BOBYQA', maxeval=100)
        
        # Check OptimizeResult attributes
        assert hasattr(res, 'x')
        assert hasattr(res, 'fun')
        assert hasattr(res, 'success')
        assert hasattr(res, 'message')
        assert hasattr(res, 'nfev')

    def test_minimize_with_bounds(self):
        """Minimize with parameter bounds."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum((x - 2)**2)
        
        x0 = np.array([0.0, 0.0])
        bounds = [(0, 1), (0, 1)]  # Constrain to [0, 1]
        
        res = minimize(quadratic, x0, method='LN_BOBYQA', bounds=bounds, maxeval=100)
        
        # Minimum is at x=[2,2], but constrained to [0,1], so should be at [1,1]
        np.testing.assert_allclose(res.x, [1.0, 1.0], atol=0.1)

    def test_minimize_invalid_option_raises(self):
        """Invalid option raises ValueError."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0])
        
        with pytest.raises(ValueError, match="could not be recognized"):
            minimize(quadratic, x0, method='LN_BOBYQA', invalid_option=42)

    def test_minimize_simple_quadratic(self):
        """Minimize simple quadratic function."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return (x[0] - 3)**2 + (x[1] + 2)**2
        
        x0 = np.array([0.0, 0.0])
        res = minimize(quadratic, x0, method='LN_BOBYQA', maxeval=200)
        
        # Minimum at (3, -2)
        np.testing.assert_allclose(res.x, [3.0, -2.0], atol=0.1)
        np.testing.assert_allclose(res.fun, 0.0, atol=0.01)

    def test_minimize_nfev_tracked(self):
        """Number of function evaluations is tracked."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        call_count = [0]
        
        def quadratic(x):
            call_count[0] += 1
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        res = minimize(quadratic, x0, method='LN_BOBYQA', maxeval=50)
        
        # nfev should be close to our manual count
        # (may differ slightly due to internal implementation)
        assert res.nfev > 0
        assert res.nfev <= 50


class TestNloptAlgorithms:
    """Test various NLopt algorithms work through the wrapper."""

    def test_ln_cobyla(self):
        """Test LN_COBYLA algorithm."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        res = minimize(quadratic, x0, method='LN_COBYLA', maxeval=100)
        
        assert res.x is not None

    def test_ln_sbplx(self):
        """Test LN_SBPLX (Subplex) algorithm."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        res = minimize(quadratic, x0, method='LN_SBPLX', maxeval=200)
        
        np.testing.assert_allclose(res.x, [0.0, 0.0], atol=0.1)

    def test_ld_mma(self):
        """Test LD_MMA (Method of Moving Asymptotes) algorithm."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        def grad(x):
            return 2 * x
        
        x0 = np.array([1.0, 2.0])
        res = minimize(quadratic, x0, method='LD_MMA', jac=grad, maxeval=100)
        
        np.testing.assert_allclose(res.x, [0.0, 0.0], atol=0.1)

    def test_gn_direct(self):
        """Test GN_DIRECT global optimization algorithm."""
        from quop_mpi.__utils.__nlopt_wrap import minimize
        
        def quadratic(x):
            return np.sum(x**2)
        
        x0 = np.array([1.0, 2.0])
        bounds = [(-5, 5), (-5, 5)]  # DIRECT requires bounds
        
        res = minimize(quadratic, x0, method='GN_DIRECT', bounds=bounds, maxeval=500)
        
        # DIRECT should find near minimum
        np.testing.assert_allclose(res.x, [0.0, 0.0], atol=0.5)


class TestNloptAlgorithmsRegistry:
    """Tests for the NLOPT_ALGORITHMS registry."""

    def test_nlopt_algorithms_not_empty(self):
        """NLOPT_ALGORITHMS contains algorithms."""
        from quop_mpi.__utils.__nlopt_wrap import NLOPT_ALGORITHMS
        
        assert len(NLOPT_ALGORITHMS) > 0

    def test_nlopt_algorithms_contains_bobyqa(self):
        """NLOPT_ALGORITHMS contains LN_BOBYQA."""
        from quop_mpi.__utils.__nlopt_wrap import NLOPT_ALGORITHMS
        
        assert 'LN_BOBYQA' in NLOPT_ALGORITHMS

    def test_nlopt_algorithms_contains_lbfgs(self):
        """NLOPT_ALGORITHMS contains LD_LBFGS."""
        from quop_mpi.__utils.__nlopt_wrap import NLOPT_ALGORITHMS
        
        assert 'LD_LBFGS' in NLOPT_ALGORITHMS

    def test_nlopt_algorithms_contains_cobyla(self):
        """NLOPT_ALGORITHMS contains LN_COBYLA."""
        from quop_mpi.__utils.__nlopt_wrap import NLOPT_ALGORITHMS
        
        assert 'LN_COBYLA' in NLOPT_ALGORITHMS

    def test_nlopt_algorithms_keys_format(self):
        """All algorithm keys match expected format (G/L)(N/D)_*."""
        from quop_mpi.__utils.__nlopt_wrap import NLOPT_ALGORITHMS_KEYS
        import re
        
        pattern = r'^[GL][ND]_'
        for key in NLOPT_ALGORITHMS_KEYS:
            assert re.match(pattern, key), f"Key {key} doesn't match pattern"
