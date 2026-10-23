"""
Minimal proof-of-concept object model for QuOp_MPI v2.

The aim of this module is architectural rather than performance-oriented. It
separates:

    logical state <-> register index       (Indexer)
    register index <-> register basis      (Register)
    register indices <-> global index      (StateSpace)
    logical operator -> indexed operator   (IndexedOperator)

MPI distribution and numerical propagation are intentionally out of scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import factorial, prod
from typing import Any, Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Mixed-radix helpers
# ---------------------------------------------------------------------------


def _normalise_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    if not shape:
        raise ValueError("A register must contain at least one qudit.")

    if any((not isinstance(d, int)) or d < 1 for d in shape):
        raise ValueError("All qudit dimensions must be positive integers.")

    return shape


def _flatten_basis(basis: Sequence[int], shape: Sequence[int]) -> int:
    """Flatten a qudit-basis tuple using row-major mixed-radix ordering."""
    basis = tuple(basis)
    shape = tuple(shape)

    if len(basis) != len(shape):
        raise ValueError(
            f"Basis has length {len(basis)}, expected {len(shape)}."
        )

    index = 0
    for value, dimension in zip(basis, shape):
        if not 0 <= value < dimension:
            raise ValueError(
                f"Basis value {value} is outside [0, {dimension})."
            )
        index = index * dimension + value

    return index


def _unflatten_basis(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    """Inverse of _flatten_basis."""
    shape = tuple(shape)
    size = prod(shape)

    if not 0 <= index < size:
        raise IndexError(f"Index {index} is outside [0, {size}).")

    basis = [0] * len(shape)

    for axis in range(len(shape) - 1, -1, -1):
        index, basis[axis] = divmod(index, shape[axis])

    return tuple(basis)


# ---------------------------------------------------------------------------
# Indexers
# ---------------------------------------------------------------------------


class Indexer(ABC):
    """Maps logical states to flat indices within one register."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Native qudit structure of the register."""

    @property
    def size(self) -> int:
        return prod(self.shape)

    @abstractmethod
    def index(self, state: Any) -> int:
        """Return the flat register index associated with a logical state."""

    @abstractmethod
    def unindex(self, index: int) -> Any:
        """Return the logical state associated with a flat register index."""


@dataclass(frozen=True)
class NativeIndexer(Indexer):
    """Indexer for registers whose native basis is already the logical state."""

    _shape: int | Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_shape", _normalise_shape(self._shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape  # type: ignore[return-value]

    def index(self, state: Sequence[int]) -> int:
        return _flatten_basis(state, self.shape)

    def unindex(self, index: int) -> tuple[int, ...]:
        return _unflatten_basis(index, self.shape)


@dataclass(frozen=True)
class PermutationIndexer(Indexer):
    """Lexicographic ranking/unranking of permutations of range(n)."""

    n: int

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be non-negative.")

    @property
    def shape(self) -> tuple[int, ...]:
        return (factorial(self.n),)

    def index(self, state: Sequence[int]) -> int:
        permutation = tuple(state)

        if len(permutation) != self.n:
            raise ValueError(
                f"Permutation has length {len(permutation)}, expected {self.n}."
            )

        if set(permutation) != set(range(self.n)):
            raise ValueError(f"Expected a permutation of range({self.n}).")

        remaining = list(range(self.n))
        rank = 0

        for position, value in enumerate(permutation):
            offset = remaining.index(value)
            rank += offset * factorial(self.n - position - 1)
            remaining.pop(offset)

        return rank

    def unindex(self, index: int) -> tuple[int, ...]:
        if not 0 <= index < self.size:
            raise IndexError(f"Index {index} is outside [0, {self.size}).")

        remaining = list(range(self.n))
        permutation: list[int] = []

        for position in range(self.n):
            block = factorial(self.n - position - 1)
            offset, index = divmod(index, block)
            permutation.append(remaining.pop(offset))

        return tuple(permutation)


@dataclass(frozen=True)
class AssignmentIndexer(Indexer):
    """Index assignments of n items to k labels."""

    n: int
    k: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.k < 1:
            raise ValueError("k must be at least 1.")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.k,) * self.n

    def index(self, state: Sequence[int]) -> int:
        return _flatten_basis(state, self.shape)

    def unindex(self, index: int) -> tuple[int, ...]:
        return _unflatten_basis(index, self.shape)


# ---------------------------------------------------------------------------
# Registers and global state space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Qudit:
    """Internal elementary tensor factor."""

    dimension: int

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("A qudit dimension must be positive.")


@dataclass(frozen=True)
class Register:
    """Named tensor factor of the full state space."""

    name: str
    indexer: Indexer
    _qudits: tuple[_Qudit, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Register name cannot be empty.")

        object.__setattr__(
            self,
            "_qudits",
            tuple(_Qudit(dimension) for dimension in self.indexer.shape),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.indexer.shape

    @property
    def dimension(self) -> int:
        return self.indexer.size

    def flatten(self, basis: Sequence[int]) -> int:
        return _flatten_basis(basis, self.shape)

    def unflatten(self, index: int) -> tuple[int, ...]:
        return _unflatten_basis(index, self.shape)


@dataclass(frozen=True)
class StateSpace:
    """Ordered tensor product of registers."""

    registers: tuple[Register, ...]

    def __init__(self, registers: Iterable[Register]):
        registers = tuple(registers)

        if not registers:
            raise ValueError("StateSpace requires at least one register.")

        names = [register.name for register in registers]
        if len(names) != len(set(names)):
            raise ValueError("Register names must be unique.")

        object.__setattr__(self, "registers", registers)

    @property
    def shape(self) -> tuple[int, ...]:
        """Dimensions of the register-level tensor product."""
        return tuple(register.dimension for register in self.registers)

    @property
    def dimension(self) -> int:
        return prod(self.shape)

    def register(self, name: str) -> Register:
        for register in self.registers:
            if register.name == name:
                return register
        raise KeyError(f"Unknown register {name!r}.")

    def register_position(self, name: str) -> int:
        for position, register in enumerate(self.registers):
            if register.name == name:
                return position
        raise KeyError(f"Unknown register {name!r}.")

    def flatten(self, register_indices: Sequence[int]) -> int:
        """Flatten one flat index per register into a global index."""
        return _flatten_basis(register_indices, self.shape)

    def unflatten(self, global_index: int) -> tuple[int, ...]:
        """Split a global index into one flat index per register."""
        return _unflatten_basis(global_index, self.shape)

    def logical_state(
        self,
        global_index: int,
        registers: Sequence[str],
    ) -> tuple[Any, ...]:
        """Unindex selected registers at a global statevector index."""
        register_indices = self.unflatten(global_index)

        states = []
        for name in registers:
            position = self.register_position(name)
            register = self.registers[position]
            states.append(register.indexer.unindex(register_indices[position]))

        return tuple(states)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """User cost function together with the registers it consumes."""

    function: Callable[..., float]
    registers: tuple[str, ...]

    def __init__(
        self,
        function: Callable[..., float],
        registers: Sequence[str],
    ):
        registers = tuple(registers)
        if not registers:
            raise ValueError("A Cost must depend on at least one register.")

        object.__setattr__(self, "function", function)
        object.__setattr__(self, "registers", registers)

    def __call__(self, *states: Any) -> float:
        return float(self.function(*states))

    def evaluate(self, space: StateSpace, global_index: int) -> float:
        states = space.logical_state(global_index, self.registers)
        return self(*states)


# ---------------------------------------------------------------------------
# Logical and indexed operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalTransition:
    """One off-diagonal transition in logical-state form."""

    states: tuple[Any, ...]
    value: complex = 1.0


class LogicalOperator(ABC):
    """Operator defined on logical states of named registers."""

    def __init__(self, registers: Sequence[str]):
        registers = tuple(registers)
        if not registers:
            raise ValueError("An operator must target at least one register.")
        if len(registers) != len(set(registers)):
            raise ValueError("Operator register names must be unique.")

        self.registers = registers

    def diagonal_element(self, *states: Any) -> complex:
        return 0.0

    def transitions(self, *states: Any) -> Iterable[LogicalTransition]:
        return ()


class CostOperator(LogicalOperator):
    """Diagonal logical operator generated from a Cost."""

    def __init__(self, cost: Cost):
        super().__init__(cost.registers)
        self.cost = cost

    def diagonal_element(self, *states: Any) -> complex:
        return self.cost(*states)


class TranspositionOperator(LogicalOperator):
    """Connect a permutation to all permutations one transposition away."""

    def __init__(self, register: str):
        super().__init__((register,))

    def transitions(
        self,
        permutation: Sequence[int],
    ) -> Iterable[LogicalTransition]:
        permutation = tuple(permutation)

        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                neighbour = list(permutation)
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                yield LogicalTransition((tuple(neighbour),), 1.0)


class NativeQuditCycleOperator(LogicalOperator):
    """PoC operator that increments one qudit of a native register."""

    def __init__(self, register: str, axis: int, dimension: int):
        super().__init__((register,))
        if axis < 0:
            raise ValueError("axis must be non-negative.")
        if dimension < 1:
            raise ValueError("dimension must be positive.")

        self.axis = axis
        self.dimension = dimension

    def transitions(
        self,
        basis: Sequence[int],
    ) -> Iterable[LogicalTransition]:
        basis = list(basis)

        if self.axis >= len(basis):
            raise ValueError(
                f"Axis {self.axis} does not exist for basis {tuple(basis)}."
            )

        basis[self.axis] = (basis[self.axis] + 1) % self.dimension
        yield LogicalTransition((tuple(basis),), 1.0)


@dataclass(frozen=True)
class IndexedTransition:
    """One off-diagonal transition in global statevector-index form."""

    index: int
    value: complex


class IndexedOperator:
    """LogicalOperator expressed in the global StateSpace ordering."""

    def __init__(self, operator: LogicalOperator, space: StateSpace):
        self.operator = operator
        self.space = space

        for name in operator.registers:
            space.register(name)

    def _logical_states(
        self,
        register_indices: Sequence[int],
    ) -> tuple[Any, ...]:
        states = []

        for name in self.operator.registers:
            position = self.space.register_position(name)
            register = self.space.registers[position]
            states.append(register.indexer.unindex(register_indices[position]))

        return tuple(states)

    def diagonal_element(self, global_index: int) -> complex:
        register_indices = self.space.unflatten(global_index)
        states = self._logical_states(register_indices)
        return self.operator.diagonal_element(*states)

    def transitions(
        self,
        global_index: int,
    ) -> Iterable[IndexedTransition]:
        register_indices = list(self.space.unflatten(global_index))
        states = self._logical_states(register_indices)

        for transition in self.operator.transitions(*states):
            if len(transition.states) != len(self.operator.registers):
                raise ValueError(
                    "Logical transition returned the wrong number of states."
                )

            target_indices = register_indices.copy()

            for name, target_state in zip(
                self.operator.registers,
                transition.states,
            ):
                position = self.space.register_position(name)
                register = self.space.registers[position]
                target_indices[position] = register.indexer.index(target_state)

            yield IndexedTransition(
                index=self.space.flatten(target_indices),
                value=transition.value,
            )


# ---------------------------------------------------------------------------
# Unitary, ansatz and problem containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unitary:
    """Evolution generated by one logical operator."""

    operator: LogicalOperator

    def indexed(self, space: StateSpace) -> IndexedOperator:
        return IndexedOperator(self.operator, space)


@dataclass(frozen=True)
class Ansatz:
    """Ordered sequence of unitaries."""

    unitaries: tuple[Unitary, ...]

    def __init__(self, unitaries: Iterable[Unitary]):
        object.__setattr__(self, "unitaries", tuple(unitaries))


@dataclass(frozen=True)
class Problem:
    """Top-level problem description for the proof of concept."""

    space: StateSpace
    cost: Cost
    ansatz: Ansatz

    def __post_init__(self) -> None:
        for register_name in self.cost.registers:
            self.space.register(register_name)

        for unitary in self.ansatz.unitaries:
            for register_name in unitary.operator.registers:
                self.space.register(register_name)

    def indexed_operators(self) -> tuple[IndexedOperator, ...]:
        return tuple(
            unitary.indexed(self.space)
            for unitary in self.ansatz.unitaries
        )


# ---------------------------------------------------------------------------
# Small executable demonstration
# ---------------------------------------------------------------------------


def build_demo_problem() -> Problem:
    permutation = Register(
        "permutation",
        PermutationIndexer(4),
    )

    assignment = Register(
        "assignment",
        AssignmentIndexer(n=4, k=2),
    )

    ancilla = Register(
        "ancilla",
        NativeIndexer(2),
    )

    space = StateSpace((permutation, assignment, ancilla))

    def example_cost(
        permutation_state: tuple[int, ...],
        assignment_state: tuple[int, ...],
    ) -> float:
        permutation_cost = sum(
            abs(a - b)
            for a, b in zip(permutation_state, permutation_state[1:])
        )
        assignment_cost = sum(assignment_state)
        return permutation_cost + assignment_cost

    cost = Cost(
        example_cost,
        registers=("permutation", "assignment"),
    )

    ansatz = Ansatz(
        (
            Unitary(CostOperator(cost)),
            Unitary(TranspositionOperator("permutation")),
            Unitary(
                NativeQuditCycleOperator(
                    register="ancilla",
                    axis=0,
                    dimension=2,
                )
            ),
        )
    )

    return Problem(
        space=space,
        cost=cost,
        ansatz=ansatz,
    )


if __name__ == "__main__":
    problem = build_demo_problem()

    print("Registers:")
    for register in problem.space.registers:
        print(
            f"  {register.name:12s} "
            f"shape={register.shape!r} "
            f"dimension={register.dimension}"
        )

    print(f"\nGlobal statevector dimension: {problem.space.dimension}")

    global_index = 37
    register_indices = problem.space.unflatten(global_index)

    print(f"\nGlobal index: {global_index}")
    print(f"Register indices: {register_indices}")

    for register, register_index in zip(
        problem.space.registers,
        register_indices,
    ):
        print(
            f"  {register.name:12s} -> "
            f"{register.indexer.unindex(register_index)!r}"
        )

    print(f"\nCost: {problem.cost.evaluate(problem.space, global_index)}")

    indexed = problem.indexed_operators()

    print("\nFirst few permutation-mixer transitions:")
    for transition in list(indexed[1].transitions(global_index))[:4]:
        print(f"  {global_index} -> {transition.index}")

    print("\nAncilla transition:")
    for transition in indexed[2].transitions(global_index):
        print(f"  {global_index} -> {transition.index}")
