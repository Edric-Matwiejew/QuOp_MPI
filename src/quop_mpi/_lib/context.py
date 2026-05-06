class Context:
    """Immediate Python adapter for the native context wrapper.

    The state and observables accessors expose host-resident buffers using the
    negotiated layout sizes:

    - ``state`` uses ``alloc_local``
    - ``observables`` uses ``local_i``

    These accessors -- and :meth:`get_local_probabilities` -- are
    **collective over** ``SUBCOMM``: every rank in the layout's SUBCOMM
    must enter together.  Backends may satisfy that contract trivially
    (the MPI backend's host mirrors alias the authoritative buffers) or
    via internal device-to-host / host-to-device transfers plus a
    SUBCOMM-scoped error reduction (the wavefront backend).  Calling
    these accessors from a subset of SUBCOMM ranks is undefined.

    Read-only view contract: the ndarrays returned by ``state`` and
    ``observables`` are cached zero-copy views onto the host mirror and
    are marked ``writeable=False``. To replace the buffer contents,
    assign through the property setter (``ctx.state = arr`` /
    ``ctx.observables = arr``); the setter performs a memcpy into the
    cached buffer and -- on backends that need it -- a host-to-device
    push.  In-place mutation through the view is rejected at the NumPy
    layer because on GPU backends the host mirror is just a snapshot
    refreshed on every getter call from the device-side authoritative
    buffer, so silent in-place writes would be lost on the next sync.

    The state buffer has shape ``(alloc_local,)`` so backends with
    extra padding on the trailing slice can use it as scratch.  Only
    ``state[:local_i]`` carries the rank's share of the wavefunction;
    ``state[local_i:alloc_local]`` is implementation-defined padding
    and its value is unspecified after any sync (in particular, the
    wavefront backend may overwrite it during ``gpu_allgatherv_dtoh``).
    Callers must not rely on the padded region's contents.
    """

    @staticmethod
    def _raise_context_status(action, error_code, status_map=None):
        """Raise a Python exception for a non-zero native context status."""
        if error_code == 0:
            return

        if status_map is None:
            status_map = {}

        exc_type, message = status_map.get(
            int(error_code),
            (RuntimeError, f"Cannot {action}: backend returned status {int(error_code)}"),
        )
        raise exc_type(message)

    def __init__(self, backend, comm_info):
        """
        Initialize a context for quantum state management.

        Parameters
        ----------
        backend :
            The backend module (mpi or wavefront).
        comm_info : QuopMpiLayout
            A locked ``QuopMpiLayout`` wrapping the Fortran
            ``quop_mpi_layout_t`` handle.  All partitioning, communicators
            and sizes are read from this single object.
        """
        self.initialised = False
        self.ptr = 0
        self._ctx = None  # Live Context PyObject from the native context wrapper.

        # The active backend exposes the CPython context-wrapper extension as
        # ``backend.context_wrapper``.  See ``cmake/QuOpContextExtension.cmake``
        # for the build-time wiring shared by all backends.
        self.context_wrapper = backend.context_wrapper

        # Store reference to the layout so it stays alive while the context exists
        self._comm_info = comm_info

        # Read the negotiated host partitioning/allocation contract from layout.
        self.system_size = comm_info.system_size
        self.host_local_i = comm_info.local_i
        self.host_local_i_offset = comm_info.local_i_offset
        self.host_alloc_local = comm_info.alloc_local
        self.SUBCOMM = comm_info.subcomm

        # setup(ci_ptr, alloc_local, local_i) -> (Context, error_code)
        # The returned Context object owns the Python-side state and
        # observables buffers (both attached to the Fortran context).
        ctx_obj, error_code = self.context_wrapper.setup(
            comm_info.handle, comm_info.alloc_local, comm_info.local_i
        )
        self._raise_context_status(
            "initialize context",
            error_code,
            {
                100: (
                    RuntimeError,
                    "Cannot initialize context: failed to allocate native context wrapper",
                ),
            },
        )
        self._ctx = ctx_obj
        # Keep the int64 Fortran handle accessible so legacy call sites that
        # marshal it directly into propagator entry points keep working.
        self.ptr = int(ctx_obj.handle)

        self.SUBCOMM.barrier()
        self.initialised = True

    def _require_initialised(self, action):
        """Raise if the native context handle is unavailable."""
        if not self.initialised or self.ptr in (None, 0):
            raise RuntimeError(f"Cannot {action}: context is not initialised or has been destroyed")

    def __enter__(self):
        """Return ``self`` so the native context can be scoped with ``with``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Deterministically release native context resources at scope exit."""
        self.destroy()
        return False

    def destroy(self):
        if self.initialised and self._ctx is not None:
            # destroy() releases the Fortran-side context and drops the
            # Python state-buffer reference.  Equivalent to letting the
            # Context object be garbage collected, but deterministic.
            self.context_wrapper.destroy(self._ctx)
        # Mirror the Ansatz destroy path: once native cleanup completes,
        # release borrowed Python-side references to the negotiated layout
        # and communicator so a still-live Context object cannot outlive
        # the layout teardown with stale MPI handles attached.
        self.SUBCOMM = None
        self._comm_info = None
        self._ctx = None
        self.ptr = 0
        self.initialised = False

    @property
    def observables(self):
        """Collectively fetch the host observable buffer over ``SUBCOMM``.

        Returns a read-only zero-copy NumPy view onto the cached
        observables buffer (length ``local_i``).  Mutate by assignment
        to the property: ``ctx.observables = arr``.  In-place writes
        through the returned view are rejected by NumPy (the array is
        marked ``writeable=False``); see the class docstring for the
        rationale.
        """
        if self.initialised:
            observables, error_code = self.context_wrapper.get_observables(
                self._ctx
            )
            self._raise_context_status(
                "fetch context observables",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " host buffer is smaller than local_i",
                    ),
                    2: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                    ),
                    3: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " host buffer exceeds internal allocation",
                    ),
                },
            )
            return observables
        return None

    @observables.setter
    def observables(self, obs):
        """Collectively update the host observable buffer over ``SUBCOMM``."""
        self._require_initialised("update context observables")
        error_code = self.context_wrapper.set_observables(self._ctx, obs)
        self._raise_context_status(
            "update context observables",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " host buffer is smaller than local_i",
                ),
                2: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                ),
                3: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " host buffer exceeds internal allocation",
                ),
            },
        )

    @property
    def state(self):
        """Collectively fetch the host state buffer over ``SUBCOMM``.

        Returns a read-only zero-copy NumPy view onto the cached state
        buffer owned by the native ``Context`` object.  The view has
        length ``alloc_local``; only ``state[:local_i]`` is the rank's
        share of the wavefunction, and ``state[local_i:alloc_local]``
        is implementation-defined padding (see the class docstring).

        On GPU backends the cached buffer is refreshed from the
        device-side authoritative copy before being returned
        (unconditional ``gpu_allgatherv_dtoh`` on wavefront); on the
        MPI backend the buffer *is* the authoritative copy.

        Mutate by assignment to the property: ``ctx.state = arr``.
        In-place writes through the returned view are rejected by
        NumPy (the array is marked ``writeable=False``).

        The view is invalidated when ``destroy()`` is called.
        """
        if self.initialised:
            state, error_code = self.context_wrapper.get_state(self._ctx)
            self._raise_context_status(
                "fetch context state",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context state: host buffer is smaller than alloc_local",
                    ),
                    2: (
                        RuntimeError,
                        "Cannot fetch context state:"
                        " native buffers or transfer metadata"
                        " are unavailable",
                    ),
                    3: (
                        RuntimeError,
                        "Cannot fetch context state: host buffer exceeds internal allocation",
                    ),
                },
            )
            return state
        return None

    @state.setter
    def state(self, state):
        """Collectively update the host state buffer over ``SUBCOMM``."""
        self._require_initialised("update context state")
        error_code = self.context_wrapper.set_state(self._ctx, state)
        self._raise_context_status(
            "update context state",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot update context state: host buffer is smaller than alloc_local",
                ),
                2: (
                    RuntimeError,
                    "Cannot update context state:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                ),
                3: (
                    RuntimeError,
                    "Cannot update context state: host buffer exceeds internal allocation",
                ),
            },
        )

    def get_local_probabilities(self):
        """Return ``|state[:local_i]|**2`` from a context-owned host buffer.

        Collective over ``SUBCOMM`` -- on the wavefront backend the
        underlying ``compute_local_probabilities`` first runs
        ``sync_host_state`` (a NODECOMM-scoped device-to-host gather
        wrapped in a SUBCOMM-scoped error reduction).  On the MPI
        backend the operation is local but callers must still treat
        the call as SUBCOMM-collective so the same call site is
        correct on every backend.

        The buffer is allocated lazily by the native extension on the
        first call (length ``host_local_i``, dtype ``float64``) and
        reused on every subsequent call: only the contents are
        recomputed from the current state vector.  The buffer's
        lifetime is bound to the Context — it is released by
        :meth:`destroy`.

        .. warning::

            The returned array aliases the context-internal storage
            and is **invalidated by any subsequent call to this
            method** (the contents are recomputed in place from the
            current state).  Callers that need to retain the
            probabilities across a state mutation or another
            ``get_local_probabilities`` call must take an explicit
            ``.copy()``.  In-tree call sites
            (:meth:`quop_mpi.ansatz.Ansatz.__get_expectation_value`,
            :mod:`quop_mpi._sampling`) consume the array within a
            single iteration and so do not need a copy.

        Returns
        -------
        ndarray[float64]
            Read-only zero-copy view onto the cached buffer (marked
            ``writeable=False``).
        """
        self._require_initialised("compute local probabilities")
        return self.context_wrapper.get_local_probabilities(self._ctx)

    def get_expectation_value(self):
        if self.initialised:
            expectation_value, error_code = self.context_wrapper.get_expectation_value(self._ctx)
            self._raise_context_status(
                "fetch context expectation value",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context expectation value:"
                        " native buffers or reduction state"
                        " are unavailable",
                    ),
                },
            )
            return expectation_value
        return None

    def get_state_norm(self):
        if self.initialised:
            state_norm, error_code = self.context_wrapper.get_state_norm(self._ctx)
            self._raise_context_status(
                "fetch context state norm",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context state norm:"
                        " native buffers or reduction state"
                        " are unavailable",
                    ),
                },
            )
            return state_norm
        return None
