# cspell:words subcomm subcomms jaccomm
"""MPI communicator scope decorators for Ansatz methods.

Provides a ``scope`` decorator that classifies methods by their MPI
communicator membership requirement, enforces scope-nesting at runtime,
and short-circuits excluded ranks.

Usage
-----
::

    from quop_mpi._scope import scope

    class Ansatz:

        @scope("subcomm", returns="all")
        def evaluate(self, variational_parameters):
            ...

        @scope("subcomm", returns="root")
        def get_final_state(self):
            ...

        @scope("world")
        def execute(self, variational_parameters=None):
            ...

Scope names and hierarchy
-------------------------
``world`` strictly contains both ``subcomm`` and ``jaccomm``, but
``subcomm`` and ``jaccomm`` are **peer** scopes -- neither nests inside
the other::

    world  =>  { subcomm , jaccomm }

``JACCOMM`` is the Fortran-side communicator that spans **all** ranks of
every worker subcomm **plus** rank 0 of the optimizer subcomm.  It is
*not* a subset of any single ``SUBCOMM``; rather it is roughly their
union.  Methods that use ``JACCOMM`` routinely delegate computation to
``SUBCOMM`` collectives (e.g. ``_mpi_jacobian`` -> ``evaluate``), so
calls between the two peer levels are permitted.

Each scope level is assigned a numeric *level* (higher = narrower):

=========  =====  ==========================================================
Name       Level  Description
=========  =====  ==========================================================
``world``    0    All ranks in the communicator passed to the constructor.
``subcomm``  1    Ranks that received work during the negotiate phase.
``jaccomm``  1    Ranks in the Jacobian communicator (peer of subcomm).
=========  =====  ==========================================================

A decorated method may call methods with **equal or narrower** scope.
Calling a *wider*-scoped method from a *narrower* context is a bug
(not all required ranks are executing) and raises ``ScopeError``.

The ``returns`` classifier
--------------------------
``returns`` is **metadata only**.  It records the return semantics of the
decorated method for documentation and introspection purposes.  The
decorator does **not** enforce these semantics at runtime -- the method
body handles root-vs-broadcast logic as before.

``"none"``
    No meaningful return value (setters, lifecycle, print helpers).

``"all"``
    Every rank *within the scope* receives the same non-trivial value
    (broadcast/allreduce methods like ``evaluate()``).

``"root"``
    Only the root rank receives a meaningful value;
    other ranks may receive ``None`` (gather-to-root methods).

Introspection
-------------
::

    >>> type(alg).evaluate._scope
    'subcomm'
    >>> type(alg).evaluate._returns
    'all'
    >>> type(alg).evaluate._scope_level
    1
"""

from __future__ import annotations

import functools
from typing import Callable

from ._collective import mpi_check

# ---------------------------------------------------------------------------
# Scope hierarchy
# ---------------------------------------------------------------------------

_SCOPE_LEVELS: dict[str, int] = {
    "world": 0,
    "subcomm": 1,
    "jaccomm": 1,
}

# ---------------------------------------------------------------------------
# Membership checks: scope name  ->  callable(self) -> bool
# ---------------------------------------------------------------------------

_SCOPE_CHECKS: dict[str, Callable] = {
    "world": lambda self: True,
    "subcomm": lambda self: (self.subcomms is not None and self.subcomms.in_subcomm()),
    "jaccomm": lambda self: (self.subcomms is not None and self.subcomms.jaccomm is not None),
}


# ---------------------------------------------------------------------------
# Comm resolvers: scope name  ->  callable(self) -> Intracomm | None
# ---------------------------------------------------------------------------
#
# Used only when ``collective_raise=True`` is requested on a decorated
# method.  Returns the actual MPI communicator object on which the
# method's collective error broadcast should run, or ``None`` if this
# rank is not a member of the relevant comm (in which case the wrapper
# falls back to the existing short-circuit behaviour).

_SCOPE_COMMS: dict[str, Callable] = {
    "world": lambda self: getattr(self, "MPI_COMM_WORLD", None),
    "subcomm": lambda self: (
        self.subcomms.SUBCOMM
        if self.subcomms is not None and self.subcomms.in_subcomm()
        else None
    ),
    "jaccomm": lambda self: (
        self.subcomms.JACCOMM
        if self.subcomms is not None and self.subcomms.jaccomm is not None
        else None
    ),
}

_VALID_RETURNS = {"none", "all", "root"}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ScopeError(RuntimeError):
    """Raised when a wider-scoped method is called from a narrower scope.

    This always indicates a bug: not all ranks required by the wider
    scope are guaranteed to be executing inside the narrower context,
    which would cause a deadlock or silent data corruption.
    """


# ---------------------------------------------------------------------------
# Stack helpers
# ---------------------------------------------------------------------------


def _push_scope(self, level: int, method_name: str) -> None:
    """Push *level* onto the instance's scope stack and validate nesting."""
    stack = self._scope_stack
    if stack:
        caller_level, caller_name = stack[-1]
        if level < caller_level:
            raise ScopeError(
                f"{method_name}() [scope level {level}] called from "
                f"{caller_name}() [scope level {caller_level}] -- "
                f"cannot widen scope from a narrower context"
            )
    stack.append((level, method_name))


def _pop_scope(self) -> None:
    """Pop the most recent scope entry."""
    self._scope_stack.pop()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def scope(comm_name: str, *, returns: str = "none", collective_raise: bool = False):
    """Decorator factory for MPI communicator scope classification.

    Parameters
    ----------
    comm_name : ``{'world', 'subcomm', 'jaccomm'}``
        The MPI scope in which the decorated method operates.
    returns : ``{'none', 'all', 'root'}``, optional
        Metadata classifier describing the return semantics.
    collective_raise : bool, optional
        When ``True`` the wrapper enforces the contract "either every
        rank of the named scope returns successfully or every rank
        raises".  Implemented via :func:`quop_mpi._collective.mpi_check`,
        which performs an ``allgather`` after the body to detect any
        per-rank failures and re-raise them everywhere.  Default
        ``False`` to preserve the cost profile of hot inner methods;
        set to ``True`` on outer entry points (``execute``,
        ``evolve_state``, ``setup``, ``destroy`` ...) so asymmetric
        raises in their inner call chain are lifted to the full scope.

    Returns
    -------
    Callable
        A decorator that wraps a method.  On ranks outside the named
        scope the wrapper returns ``None`` without calling the
        underlying method.  Scope-nesting is validated via a per-instance
        stack (``self._scope_stack``).

    Raises
    ------
    ValueError
        If *comm_name* is not a recognised scope or *returns* is invalid.
    ScopeError
        At runtime, if the call would widen scope inside a narrower
        context.
    """
    if comm_name not in _SCOPE_CHECKS:
        raise ValueError(f"Unknown scope {comm_name!r}; " f"choose from {set(_SCOPE_CHECKS)}")
    if returns not in _VALID_RETURNS:
        raise ValueError(
            f"Unknown returns classifier {returns!r}; " f"choose from {_VALID_RETURNS}"
        )
    check = _SCOPE_CHECKS[comm_name]
    resolve_comm = _SCOPE_COMMS[comm_name]
    level = _SCOPE_LEVELS[comm_name]

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not check(self):
                return None
            _push_scope(self, level, method.__qualname__)
            try:
                if collective_raise:
                    comm = resolve_comm(self)
                    return mpi_check(comm, method, self, *args, **kwargs)
                return method(self, *args, **kwargs)
            finally:
                _pop_scope(self)

        # Attach metadata for introspection
        wrapper._scope = comm_name
        wrapper._scope_level = level
        wrapper._returns = returns
        wrapper._collective_raise = collective_raise
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def get_scope(method) -> str:
    """Return the scope name attached to *method*, or ``'world'``."""
    return getattr(method, "_scope", "world")


def get_scope_level(method) -> int:
    """Return the numeric scope level attached to *method*, or ``0``."""
    return getattr(method, "_scope_level", 0)


def get_returns(method) -> str:
    """Return the returns classifier attached to *method*, or ``'none'``."""
    return getattr(method, "_returns", "none")
