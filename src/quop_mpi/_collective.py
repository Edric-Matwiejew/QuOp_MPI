"""Collective error propagation primitives for QuOp's MPI scopes.

The functions here let a method that is collective on some communicator
``comm`` honour the contract:

    Either every rank of ``comm`` returns successfully, or every rank of
    ``comm`` raises the same exception.

Without this contract, a raise on a strict subset of ranks (e.g. only the
optimiser-leader rank inside ``Ansatz.execute``) leaves the other ranks
to either deadlock at the next collective or silently miss the failure
and proceed with inconsistent state.

Two surfaces are provided:

* :func:`mpi_check` -- a callable wrapper, suitable for direct use at a
  call site (e.g. wrapping a user-supplied callback or a third-party
  function that you cannot decorate).
* The ``collective_raise=True`` option on :func:`quop_mpi._scope.scope`
  -- the preferred mechanism for QuOp methods, since the scope is
  already declared at the method definition.

Both surfaces share the implementation in :func:`mpi_check`.
"""

# cspell:words allgather

from __future__ import annotations

import pickle
import sys
from typing import Any, Callable

from mpi4py import MPI


class CollectiveError(RuntimeError):
    """Raised on ranks that observed another rank's failure during a
    collective method.

    The originating rank raises the original exception unchanged so the
    user sees its traceback.  Other ranks raise :class:`CollectiveError`
    with the original exception attached as ``__cause__``.
    """


def _encode_exc(comm: MPI.Intracomm, exc: BaseException | None) -> tuple[int, bytes] | None:
    """Encode an exception for allgather.

    Returns ``None`` when there is no exception, otherwise a tuple of
    ``(rank, payload)`` where ``payload`` is a pickled exception (or a
    pickled :class:`RuntimeError` carrying ``repr(exc)`` if the original
    cannot be pickled).
    """
    if exc is None:
        return None
    rank = comm.Get_rank()
    try:
        return (rank, pickle.dumps(exc))
    except Exception:  # noqa: BLE001 -- last-resort pickling fallback
        fallback = RuntimeError(f"{type(exc).__name__}: {exc!r}")
        return (rank, pickle.dumps(fallback))


def _earliest_failure(
    payloads: list[tuple[int, bytes] | None],
) -> tuple[int, bytes] | None:
    """Return the failure payload from the lowest-numbered failing rank."""
    failures = [p for p in payloads if p is not None]
    if not failures:
        return None
    return min(failures, key=lambda p: p[0])


def mpi_check(
    comm: MPI.Intracomm | None,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``fn(*args, **kwargs)`` collectively on ``comm``.

    If ``fn`` raises on any rank of ``comm``, the exception is broadcast
    via an allgather and re-raised on every rank: the originating rank
    raises the original exception (preserving its traceback) and every
    other rank raises :class:`CollectiveError` with the original
    exception attached as ``__cause__``.

    Parameters
    ----------
    comm
        MPI communicator on which the call is collective.  If ``None``
        (e.g. a rank that is not a member of the relevant comm) the
        function is executed without any collective synchronisation;
        callers are responsible for ensuring such ranks have nothing to
        report.
    fn
        Callable to invoke.
    *args, **kwargs
        Forwarded to ``fn``.

    Returns
    -------
    Any
        Whatever ``fn`` returns when it succeeds.

    Raises
    ------
    BaseException
        The original exception on the originating rank.
    CollectiveError
        On every other rank of ``comm``.
    """
    if comm is None:
        return fn(*args, **kwargs)

    local_exc: BaseException | None = None
    local_result: Any = None
    try:
        local_result = fn(*args, **kwargs)
    except BaseException as e:  # noqa: BLE001 -- intentional: propagate any failure
        local_exc = e

    payload = _encode_exc(comm, local_exc)
    failures = comm.allgather(payload)
    first = _earliest_failure(failures)

    if first is None:
        return local_result

    # Log all failures from rank 0; propagate only the earliest to preserve
    # a single-exception contract (pytest.raises / except ExcClass).
    all_failures = [p for p in failures if p is not None]
    if len(all_failures) > 1 and comm.Get_rank() == 0:
        lines = [
            f"[quop_mpi.mpi_check] {len(all_failures)} ranks failed during "
            f"collective call (comm size={comm.Get_size()}); "
            f"propagating origin (rank {all_failures[0][0]}). "
            f"All failures:"
        ]
        for r, blob in all_failures:
            try:
                e = pickle.loads(blob)
                lines.append(f"  rank {r}: {type(e).__name__}: {e}")
            except Exception as unpickle_err:  # noqa: BLE001
                lines.append(
                    f"  rank {r}: <unpickle failed: {unpickle_err!r}>"
                )
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()

    origin, blob = first
    try:
        origin_exc = pickle.loads(blob)
    except Exception as unpickle_err:  # noqa: BLE001
        origin_exc = CollectiveError(
            f"Collective failure on rank {origin}; "
            f"failed to unpickle origin exception: {unpickle_err!r}"
        )

    if comm.Get_rank() == origin and local_exc is not None:
        # Re-raise the local exception object so the user sees its
        # full traceback (the unpickled copy would lose the frames).
        raise local_exc

    # Raise the unpickled exception on non-origin ranks so class/message
    # match across the comm; origin context attached via __cause__.
    raise origin_exc from CollectiveError(
        f"originated on rank {origin} of comm "
        f"(size={comm.Get_size()}, this_rank={comm.Get_rank()})"
    )
