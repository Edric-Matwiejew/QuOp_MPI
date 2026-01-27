"""
Unit tests for the quop_mpi.toolkit module.

This module contains tests for:
- kronecker.py: Tensor (Kronecker) product utilities
- pauli.py: Pauli matrix generation
- string.py: Bit-string to quantum state conversion
"""
import pytest
import numpy as np
from scipy import sparse


class TestKronecker:
    """Tests for quop_mpi.toolkit.kronecker module."""

    def test_kron_empty_list_returns_one(self):
        """kron of empty list returns 1."""
        from quop_mpi.toolkit.kronecker import kron
        
        result = kron([])
        assert result == 1

    def test_kron_single_matrix_returns_same(self):
        """kron of single matrix returns that matrix."""
        from quop_mpi.toolkit.kronecker import kron
        
        matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        result = kron([matrix])
        
        assert sparse.issparse(result)
        np.testing.assert_array_equal(result.toarray(), matrix.toarray())

    def test_kron_two_identity_matrices(self):
        """kron of two 2x2 identity matrices gives 4x4 identity."""
        from quop_mpi.toolkit.kronecker import kron
        
        I2 = sparse.identity(2, format='csr')
        result = kron([I2, I2])
        
        expected = sparse.identity(4, format='csr')
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_kron_pauli_z_tensored(self):
        """Test kronecker product of Pauli Z matrices."""
        from quop_mpi.toolkit.kronecker import kron
        
        Z = sparse.csr_matrix([[1, 0], [0, -1]])
        result = kron([Z, Z])
        
        # Z (x) Z = diag(1, -1, -1, 1)
        expected = np.diag([1, -1, -1, 1])
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_kron_with_dense_arrays(self):
        """kron works with numpy arrays (dense)."""
        from quop_mpi.toolkit.kronecker import kron
        
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[1, 2], [3, 4]])
        result = kron([a, b])
        
        expected = np.kron(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_kron_three_matrices(self):
        """kron of three matrices."""
        from quop_mpi.toolkit.kronecker import kron
        
        I2 = sparse.identity(2, format='csr')
        result = kron([I2, I2, I2])
        
        expected = sparse.identity(8, format='csr')
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_kron_power_zero(self):
        """kron_power with n=0 returns 1."""
        from quop_mpi.toolkit.kronecker import kron_power
        
        matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        result = kron_power(matrix, 0)
        
        assert result == 1

    def test_kron_power_one(self):
        """kron_power with n=1 returns the original matrix."""
        from quop_mpi.toolkit.kronecker import kron_power
        
        matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        result = kron_power(matrix, 1)
        
        np.testing.assert_array_equal(result.toarray(), matrix.toarray())

    def test_kron_power_two(self):
        """kron_power with n=2 is equivalent to kron([m, m])."""
        from quop_mpi.toolkit.kronecker import kron_power, kron
        
        matrix = sparse.csr_matrix([[1, 0], [0, -1]])
        result = kron_power(matrix, 2)
        expected = kron([matrix, matrix])
        
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_kron_power_three_qubits(self):
        """kron_power with n=3 gives correct dimensions."""
        from quop_mpi.toolkit.kronecker import kron_power
        
        matrix = sparse.csr_matrix([[1, 0], [0, 1]])  # 2x2 identity
        result = kron_power(matrix, 3)
        
        assert result.shape == (8, 8)
        np.testing.assert_array_almost_equal(result.toarray(), np.eye(8))


class TestPauli:
    """Tests for quop_mpi.toolkit.pauli module."""

    def test_pauli_x_matrix_correct(self):
        """Verify Pauli X matrix definition."""
        from quop_mpi.toolkit import pauli
        
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(pauli.x.toarray(), expected)

    def test_pauli_y_matrix_correct(self):
        """Verify Pauli Y matrix definition."""
        from quop_mpi.toolkit import pauli
        
        expected = np.array([[0, -1j], [1j, 0]])
        np.testing.assert_array_equal(pauli.y.toarray(), expected)

    def test_pauli_z_matrix_correct(self):
        """Verify Pauli Z matrix definition."""
        from quop_mpi.toolkit import pauli
        
        expected = np.array([[1, 0], [0, -1]])
        np.testing.assert_array_equal(pauli.z.toarray(), expected)

    def test_identity_single_qubit(self):
        """Identity for 1 qubit is 2x2."""
        from quop_mpi.toolkit.pauli import I
        
        result = I(1)
        expected = np.eye(2)
        
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_identity_two_qubits(self):
        """Identity for 2 qubits is 4x4."""
        from quop_mpi.toolkit.pauli import I
        
        result = I(2)
        expected = np.eye(4)
        
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_identity_three_qubits(self):
        """Identity for 3 qubits is 8x8."""
        from quop_mpi.toolkit.pauli import I
        
        result = I(3)
        assert result.shape == (8, 8)
        np.testing.assert_array_almost_equal(result.toarray(), np.eye(8))

    def test_pauli_X_on_first_qubit_two_qubit_system(self):
        """X on qubit 0 in 2-qubit system: X (x) I."""
        from quop_mpi.toolkit.pauli import X
        
        result = X(0, 2)
        
        # X (x) I
        X_mat = np.array([[0, 1], [1, 0]])
        I_mat = np.eye(2)
        expected = np.kron(X_mat, I_mat)
        
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_pauli_X_on_second_qubit_two_qubit_system(self):
        """X on qubit 1 in 2-qubit system: I (x) X."""
        from quop_mpi.toolkit.pauli import X
        
        result = X(1, 2)
        
        # I (x) X
        X_mat = np.array([[0, 1], [1, 0]])
        I_mat = np.eye(2)
        expected = np.kron(I_mat, X_mat)
        
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_pauli_Z_on_middle_qubit_three_qubit_system(self):
        """Z on qubit 1 in 3-qubit system: I (x) Z (x) I."""
        from quop_mpi.toolkit.pauli import Z
        
        result = Z(1, 3)
        
        Z_mat = np.array([[1, 0], [0, -1]])
        I_mat = np.eye(2)
        expected = np.kron(np.kron(I_mat, Z_mat), I_mat)
        
        assert result.shape == (8, 8)
        np.testing.assert_array_almost_equal(result.toarray(), expected)

    def test_pauli_Y_hermitian(self):
        """Pauli Y is Hermitian."""
        from quop_mpi.toolkit.pauli import Y
        
        result = Y(0, 1)
        
        # Y^dag = Y
        np.testing.assert_array_almost_equal(
            result.toarray(),
            result.toarray().conj().T
        )

    def test_pauli_matrices_square_to_identity(self):
        """X^2, Y^2, Z^2 = I."""
        from quop_mpi.toolkit.pauli import X, Y, Z, I
        
        n_qubits = 2
        for pauli_op, idx in [(X, 0), (Y, 1), (Z, 0)]:
            op = pauli_op(idx, n_qubits)
            op_squared = op @ op
            identity = I(n_qubits)
            
            np.testing.assert_array_almost_equal(
                op_squared.toarray(),
                identity.toarray()
            )

    def test_pauli_commutation_XY_eq_iZ(self):
        """[X, Y] = 2iZ for single qubit."""
        from quop_mpi.toolkit.pauli import X, Y, Z
        
        X_op = X(0, 1)
        Y_op = Y(0, 1)
        Z_op = Z(0, 1)
        
        commutator = X_op @ Y_op - Y_op @ X_op
        expected = 2j * Z_op
        
        np.testing.assert_array_almost_equal(
            commutator.toarray(),
            expected.toarray()
        )

    def test_pauli_anticommutation(self):
        """Pauli matrices anticommute: {X, Y} = 0."""
        from quop_mpi.toolkit.pauli import X, Y
        
        X_op = X(0, 1)
        Y_op = Y(0, 1)
        
        anticommutator = X_op @ Y_op + Y_op @ X_op
        
        np.testing.assert_array_almost_equal(
            anticommutator.toarray(),
            np.zeros((2, 2))
        )

    def test_pauli_returns_csr_matrix(self):
        """Pauli functions return CSR sparse matrices."""
        from quop_mpi.toolkit.pauli import X, Y, Z, I
        
        assert sparse.isspmatrix_csr(X(0, 2))
        assert sparse.isspmatrix_csr(Y(0, 2))
        assert sparse.isspmatrix_csr(Z(0, 2))
        assert sparse.isspmatrix_csr(I(2))


class TestString:
    """Tests for quop_mpi.toolkit.string module."""

    def test_string_zero(self):
        """String '0' gives |0> state."""
        from quop_mpi.toolkit.string import string
        
        result = string('0')
        expected = np.array([1, 0], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_one(self):
        """String '1' gives |1> state."""
        from quop_mpi.toolkit.string import string
        
        result = string('1')
        expected = np.array([0, 1], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_two_qubits_00(self):
        """String '00' gives |00> = [1,0,0,0]."""
        from quop_mpi.toolkit.string import string
        
        result = string('00')
        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_two_qubits_01(self):
        """String '01' gives |01> = [0,1,0,0]."""
        from quop_mpi.toolkit.string import string
        
        result = string('01')
        expected = np.array([0, 1, 0, 0], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_two_qubits_10(self):
        """String '10' gives |10> = [0,0,1,0]."""
        from quop_mpi.toolkit.string import string
        
        result = string('10')
        expected = np.array([0, 0, 1, 0], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_two_qubits_11(self):
        """String '11' gives |11> = [0,0,0,1]."""
        from quop_mpi.toolkit.string import string
        
        result = string('11')
        expected = np.array([0, 0, 0, 1], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)

    def test_string_three_qubits_101(self):
        """String '101' gives |101> = basis state 5."""
        from quop_mpi.toolkit.string import string
        
        result = string('101')
        
        # |101> is index 5 in an 8-dim space (binary 101 = 5)
        expected = np.zeros(8, dtype=np.complex128)
        expected[5] = 1
        
        np.testing.assert_array_equal(result, expected)

    def test_string_returns_complex128(self):
        """String function returns complex128 array."""
        from quop_mpi.toolkit.string import string
        
        result = string('010')
        assert result.dtype == np.complex128

    def test_string_is_normalized(self):
        """Resulting state vector is normalized."""
        from quop_mpi.toolkit.string import string
        
        for state_str in ['0', '1', '00', '01', '10', '11', '000', '101', '111']:
            result = string(state_str)
            norm = np.linalg.norm(result)
            np.testing.assert_almost_equal(norm, 1.0)

    def test_string_dimension_matches_qubit_count(self):
        """State dimension is 2^n for n-qubit string."""
        from quop_mpi.toolkit.string import string
        
        for n in range(1, 6):
            state_str = '0' * n
            result = string(state_str)
            assert len(result) == 2**n

    def test_string_basis_states_orthogonal(self):
        """All basis states from strings are orthogonal."""
        from quop_mpi.toolkit.string import string
        
        # 2-qubit basis states
        states = [string(s) for s in ['00', '01', '10', '11']]
        
        for i, s1 in enumerate(states):
            for j, s2 in enumerate(states):
                inner = np.vdot(s1, s2)
                if i == j:
                    np.testing.assert_almost_equal(inner, 1.0)
                else:
                    np.testing.assert_almost_equal(inner, 0.0)


class TestToolkitIntegration:
    """Integration tests combining toolkit components."""

    def test_pauli_z_eigenvalues_match_string_states(self):
        """Pauli Z has eigenvalue +1 for |0> and -1 for |1>."""
        from quop_mpi.toolkit.pauli import Z
        from quop_mpi.toolkit.string import string
        
        Z_op = Z(0, 1).toarray()
        
        state_0 = string('0')
        state_1 = string('1')
        
        # Z|0> = +|0>
        result_0 = Z_op @ state_0
        np.testing.assert_array_almost_equal(result_0, state_0)
        
        # Z|1> = -|1>
        result_1 = Z_op @ state_1
        np.testing.assert_array_almost_equal(result_1, -state_1)

    def test_pauli_x_flips_states(self):
        """Pauli X flips |0> <-> |1>."""
        from quop_mpi.toolkit.pauli import X
        from quop_mpi.toolkit.string import string
        
        X_op = X(0, 1).toarray()
        
        state_0 = string('0')
        state_1 = string('1')
        
        # X|0> = |1>
        np.testing.assert_array_almost_equal(X_op @ state_0, state_1)
        
        # X|1> = |0>
        np.testing.assert_array_almost_equal(X_op @ state_1, state_0)

    def test_kron_power_matches_multi_qubit_pauli(self):
        """kron_power of Pauli Z matches multi-qubit Z(x)Z(x)Z."""
        from quop_mpi.toolkit.kronecker import kron_power
        from quop_mpi.toolkit.pauli import z
        
        n_qubits = 3
        result = kron_power(z, n_qubits)
        
        # Z(x)Z(x)Z has diagonal elements based on parity
        # For state |abc>, eigenvalue is (-1)^(a+b+c)
        expected_diag = []
        for i in range(2**n_qubits):
            parity = bin(i).count('1')
            expected_diag.append((-1)**parity)
        
        np.testing.assert_array_almost_equal(
            result.toarray(),
            np.diag(expected_diag)
        )
