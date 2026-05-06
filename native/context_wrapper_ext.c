/*
 * context_wrapper_ext.c
 *
 * CPython C extension that replaces the f2py-generated context_wrapper
 * module with explicit host-side memory management for the state buffer.
 *
 * Design summary
 * --------------
 * The Python-facing handle ("context_ptr") is no longer a bare int64 — it
 * is a `Context` PyObject that owns:
 *
 *   - the int64 Fortran handle returned by cw_setup,
 *   - a strong reference to a NumPy complex128 array used as the state
 *     buffer (true zero-copy on every get_state call),
 *   - cached size metadata (alloc_local).
 *
 * On setup() the extension:
 *   1. calls cw_setup, which allocates the Fortran context and lets it
 *      perform its normal internal allocation;
 *   2. allocates a Python-owned complex128 array of length alloc_local;
 *   3. calls cw_attach_state, which deallocates the Fortran-side buffer
 *      and re-binds ctx%state to the Python pointer.
 *
 * On destroy() the extension:
 *   1. calls cw_destroy_external, which nullifies ctx%state and
 *      ctx%observables before invoking ctx%destroy() so Python-owned
 *      memory is not freed by Fortran;
 *   2. drops its references to the state and observables arrays,
 *      allowing NumPy to free the buffers when the last Python
 *      reference goes away.
 *
 * Both state (complex128, length alloc_local) and observables (float64,
 * length local_i) are attached as Python-owned NumPy buffers at setup().
 * Set/get on either is therefore zero-copy when the caller's input is the
 * cached array (identity short-circuit) or matches the dtype/layout.
 *
 * Python-facing API (consumed by src/quop_mpi/_lib/context.py)
 * ------------------------------------------------------------
 *   setup(ci_ptr, alloc_local, local_i) -> (Context, error_code)
 *       SIGNATURE CHANGE vs. f2py: alloc_local and local_i are now
 *       required so the extension can size both Python-owned buffers up
 *       front.
 *
 *   destroy(ctx)                       -> None
 *   get_state(ctx)                     -> (ndarray[complex128], 0)
 *       SIGNATURE CHANGE vs. f2py: no size argument; returns the cached
 *       zero-copy view.
 *   set_state(ctx, state)              -> error_code
 *   get_observables(ctx)               -> (ndarray[float64], 0)
 *       SIGNATURE CHANGE vs. f2py: no size argument; returns the cached
 *       zero-copy view.
 *   set_observables(ctx, obs)          -> error_code
 *   get_expectation_value(ctx)         -> (float, error_code)
 *   get_state_norm(ctx)                -> (float, error_code)
 *
 * Build: see context_wrapper_cmake_snippet.cmake.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* -------------------------------------------------------------------------
 * Fortran shim entry points (see context_wrapper_c.f90).
 * ------------------------------------------------------------------------- */

extern void cw_setup            (int64_t ci_ptr_val, int64_t *ctx_out, int32_t *err);
extern void cw_destroy          (int64_t ctx);
extern void cw_destroy_external (int64_t ctx);
extern void cw_attach_host_state              (int64_t ctx, void *data, int64_t n);
extern void cw_attach_host_observables        (int64_t ctx, void *data, int64_t n);
extern void cw_attach_host_local_probabilities(int64_t ctx, void *data, int64_t n);
extern void cw_sync_host_state         (int64_t ctx, int32_t *err);
extern void cw_sync_device_state       (int64_t ctx, int32_t *err);
extern void cw_sync_host_observables   (int64_t ctx, int32_t *err);
extern void cw_sync_device_observables (int64_t ctx, int32_t *err);
extern void cw_compute_local_probabilities(int64_t ctx, int32_t *err);
extern void cw_get_expectation_value(int64_t ctx, double *val, int32_t *err);
extern void cw_get_state_norm   (int64_t ctx, double *val, int32_t *err);


/* =========================================================================
 * Context PyObject
 *
 * Holds the Fortran handle and the Python-owned state ndarray.  The
 * Context must outlive any view returned by get_state, otherwise Fortran
 * would be left with a dangling self%state pointer.
 * ========================================================================= */
typedef struct {
    PyObject_HEAD
    int64_t        handle;        /* opaque Fortran context_type pointer  */
    PyArrayObject *state;         /* strong ref; NULL once destroyed      */
    PyArrayObject *observables;   /* strong ref; NULL once destroyed      */
    PyArrayObject *local_probabilities; /* lazy; NULL until first request */
    int64_t        alloc_local;   /* length of the state buffer           */
    int64_t        local_i;       /* length of the observables buffer     */
    int            destroyed;     /* idempotency guard                    */
} Context;


static void
Context_release(Context *self)
{
    /* Nullify Fortran's pointers first, then deallocate the context. */
    if (!self->destroyed && self->handle != 0) {
        cw_destroy_external(self->handle);
        self->handle = 0;
    }
    self->destroyed = 1;
    Py_CLEAR(self->state);
    Py_CLEAR(self->observables);
    Py_CLEAR(self->local_probabilities);
}


static void
Context_dealloc(Context *self)
{
    Context_release(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


/* Read-only int64 handle: required so legacy call sites (e.g. the
 * propagator wrapper's plan(), which takes context.ptr as an int64) can
 * continue to operate without holding a reference to the PyObject. */
static PyObject *
Context_get_handle(Context *self, void *closure)
{
    (void)closure;
    return PyLong_FromLongLong((long long)self->handle);
}

static PyObject *
Context_get_alloc_local(Context *self, void *closure)
{
    (void)closure;
    return PyLong_FromLongLong((long long)self->alloc_local);
}

static PyGetSetDef Context_getset[] = {
    {"handle",      (getter)Context_get_handle,      NULL,
     "Opaque int64 pointer to the underlying Fortran context_type.", NULL},
    {"alloc_local", (getter)Context_get_alloc_local, NULL,
     "Length of the Python-owned state buffer.",                       NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject ContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "context_wrapper.Context",
    .tp_doc       = "Opaque native context handle owning a Python-allocated state buffer.",
    .tp_basicsize = sizeof(Context),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_dealloc   = (destructor)Context_dealloc,
    .tp_getset    = Context_getset,
    /* No tp_new / tp_init: instances are produced exclusively by setup(). */
};


/* Validate that an arbitrary PyObject is a live Context. */
static Context *
as_context(PyObject *obj, const char *funcname)
{
    if (!PyObject_TypeCheck(obj, &ContextType)) {
        PyErr_Format(PyExc_TypeError,
                     "%s: expected Context object, got %s",
                     funcname, Py_TYPE(obj)->tp_name);
        return NULL;
    }
    Context *ctx = (Context *)obj;
    if (ctx->destroyed || ctx->handle == 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s: context has been destroyed", funcname);
        return NULL;
    }
    return ctx;
}


/* =========================================================================
 * setup(ci_ptr: int, alloc_local: int, local_i: int) -> (Context, error_code)
 * ========================================================================= */
static PyObject *
py_setup(PyObject *module, PyObject *args)
{
    int64_t ci_ptr_val;
    int64_t alloc_local;
    int64_t local_i;

    if (!PyArg_ParseTuple(args, "LLL", &ci_ptr_val, &alloc_local, &local_i))
        return NULL;

    if (alloc_local < 0) {
        PyErr_SetString(PyExc_ValueError, "alloc_local must be non-negative");
        return NULL;
    }
    if (local_i < 0) {
        PyErr_SetString(PyExc_ValueError, "local_i must be non-negative");
        return NULL;
    }

    /* Step 1: native allocation + initialisation. */
    int64_t handle     = 0;
    int32_t error_code = 0;
    cw_setup(ci_ptr_val, &handle, &error_code);
    if (error_code != 0 || handle == 0) {
        Py_INCREF(Py_None);
        return Py_BuildValue("Ni", Py_None, (int)error_code);
    }

    /* Step 2: Python-owned state buffer. */
    npy_intp state_dims[1] = { (npy_intp)alloc_local };
    PyArrayObject *state =
        (PyArrayObject *)PyArray_ZEROS(1, state_dims, NPY_COMPLEX128, 0);
    if (!state) {
        cw_destroy(handle);
        return NULL;
    }

    /* Step 3: Python-owned observables buffer. */
    npy_intp obs_dims[1] = { (npy_intp)local_i };
    PyArrayObject *observables =
        (PyArrayObject *)PyArray_ZEROS(1, obs_dims, NPY_FLOAT64, 0);
    if (!observables) {
        Py_DECREF(state);
        cw_destroy(handle);
        return NULL;
    }

    /* Step 4: rebind ctx%host_state and ctx%host_observables to the
     * Python buffers.  On MPI this also retargets ctx%state /
     * ctx%observables (single host copy); on wavefront the device
     * buffers are left intact and host_* becomes the host mirror. */
    cw_attach_host_state(handle, PyArray_DATA(state), (int64_t)alloc_local);
    cw_attach_host_observables(handle, PyArray_DATA(observables), (int64_t)local_i);

    /* Step 5: assemble the Context PyObject. */
    Context *ctx = PyObject_New(Context, &ContextType);
    if (!ctx) {
        Py_DECREF(state);
        Py_DECREF(observables);
        cw_destroy_external(handle);
        return NULL;
    }
    ctx->handle      = handle;
    ctx->state       = state;        /* steal the new reference */
    ctx->observables = observables;  /* steal the new reference */
    ctx->local_probabilities = NULL; /* allocated lazily */
    ctx->alloc_local = alloc_local;
    ctx->local_i     = local_i;
    ctx->destroyed   = 0;

    return Py_BuildValue("Ni", (PyObject *)ctx, 0);
}


/* =========================================================================
 * destroy(ctx) -> None
 * ========================================================================= */
static PyObject *
py_destroy(PyObject *module, PyObject *arg)
{
    if (!PyObject_TypeCheck(arg, &ContextType)) {
        PyErr_SetString(PyExc_TypeError, "destroy: expected Context object");
        return NULL;
    }
    Context_release((Context *)arg);
    Py_RETURN_NONE;
}


/* =========================================================================
 * get_state(ctx) -> (ndarray[complex128], error_code)
 *
 * Refreshes the host mirror from the authoritative copy (no-op on MPI;
 * device->host gather on wavefront), then returns a new reference to
 * the cached buffer.  Truly zero-copy when error_code == 0.
 * ========================================================================= */
static PyObject *
py_get_state(PyObject *module, PyObject *arg)
{
    Context *ctx = as_context(arg, "get_state");
    if (!ctx) return NULL;

    int32_t error_code = 0;
    cw_sync_host_state(ctx->handle, &error_code);

    Py_INCREF(ctx->state);
    return Py_BuildValue("Ni", (PyObject *)ctx->state, (int)error_code);
}


/* =========================================================================
 * set_state(ctx, state) -> error_code
 *
 * Writes into the cached buffer (identity short-circuits when given the
 * cached array), then pushes the host mirror to the authoritative copy
 * (no-op on MPI; host->device scatter on wavefront).
 * ========================================================================= */
static PyObject *
py_set_state(PyObject *module, PyObject *args)
{
    PyObject      *ctx_obj;
    PyArrayObject *arr;

    if (!PyArg_ParseTuple(args, "OO!", &ctx_obj, &PyArray_Type, &arr))
        return NULL;

    Context *ctx = as_context(ctx_obj, "set_state");
    if (!ctx) return NULL;

    if ((PyObject *)arr != (PyObject *)ctx->state) {
        PyArrayObject *src = (PyArrayObject *)PyArray_FROM_OTF(
            (PyObject *)arr, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (!src) return NULL;

        npy_intp src_size = PyArray_SIZE(src);
        if (src_size > ctx->alloc_local) {
            Py_DECREF(src);
            return PyLong_FromLong(3); /* matches existing wrapper status code */
        }

        const size_t elt = sizeof(npy_complex128);
        memcpy(PyArray_DATA(ctx->state), PyArray_DATA(src), (size_t)src_size * elt);
        if (src_size < ctx->alloc_local) {
            memset((char *)PyArray_DATA(ctx->state) + (size_t)src_size * elt,
                   0,
                   (size_t)(ctx->alloc_local - src_size) * elt);
        }
        Py_DECREF(src);
    }

    int32_t error_code = 0;
    cw_sync_device_state(ctx->handle, &error_code);
    return PyLong_FromLong((long)error_code);
}


/* =========================================================================
 * Observables (zero-copy: same pattern as state).
 *
 * get_observables triggers a host-side refresh (no-op on MPI; dtoh on
 * GPU backends) and returns a new reference to the cached buffer.
 * set_observables identity-shortcuts when given the cached array,
 * otherwise memcpy-s the source in, then triggers a device-side push
 * (no-op on MPI; htod on GPU).
 * ========================================================================= */
static PyObject *
py_get_observables(PyObject *module, PyObject *arg)
{
    Context *ctx = as_context(arg, "get_observables");
    if (!ctx) return NULL;

    int32_t error_code = 0;
    cw_sync_host_observables(ctx->handle, &error_code);

    Py_INCREF(ctx->observables);
    return Py_BuildValue("Ni", (PyObject *)ctx->observables, (int)error_code);
}


static PyObject *
py_set_observables(PyObject *module, PyObject *args)
{
    PyObject      *ctx_obj;
    PyArrayObject *arr;

    if (!PyArg_ParseTuple(args, "OO!", &ctx_obj, &PyArray_Type, &arr))
        return NULL;

    Context *ctx = as_context(ctx_obj, "set_observables");
    if (!ctx) return NULL;

    if ((PyObject *)arr != (PyObject *)ctx->observables) {
        PyArrayObject *src = (PyArrayObject *)PyArray_FROM_OTF(
            (PyObject *)arr, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!src) return NULL;

        npy_intp src_size = PyArray_SIZE(src);
        if (src_size > ctx->local_i) {
            Py_DECREF(src);
            return PyLong_FromLong(3); /* matches existing wrapper status code */
        }

        const size_t elt = sizeof(double);
        memcpy(PyArray_DATA(ctx->observables), PyArray_DATA(src),
               (size_t)src_size * elt);
        if (src_size < ctx->local_i) {
            memset((char *)PyArray_DATA(ctx->observables) + (size_t)src_size * elt,
                   0,
                   (size_t)(ctx->local_i - src_size) * elt);
        }
        Py_DECREF(src);
    }

    int32_t error_code = 0;
    cw_sync_device_observables(ctx->handle, &error_code);
    return PyLong_FromLong((long)error_code);
}


/* =========================================================================
 * Scalars.
 * ========================================================================= */
static PyObject *
py_get_expectation_value(PyObject *module, PyObject *arg)
{
    Context *ctx = as_context(arg, "get_expectation_value");
    if (!ctx) return NULL;

    double  value      = 0.0;
    int32_t error_code = 0;
    cw_get_expectation_value(ctx->handle, &value, &error_code);
    return Py_BuildValue("di", value, (int)error_code);
}


static PyObject *
py_get_state_norm(PyObject *module, PyObject *arg)
{
    Context *ctx = as_context(arg, "get_state_norm");
    if (!ctx) return NULL;

    double  value      = 0.0;
    int32_t error_code = 0;
    cw_get_state_norm(ctx->handle, &value, &error_code);
    return Py_BuildValue("di", value, (int)error_code);
}


/* =========================================================================
 * Local probabilities (lazy zero-copy buffer owned by Context).
 *
 * Returns the cached float64 buffer of length local_i, populated with
 * |state[i]|**2 for i in [0, local_i).  Allocated and attached on the
 * first call (Fortran side keeps a host_local_probabilities pointer to
 * the same memory) and reused thereafter; freed by Context_release
 * alongside the state and observables buffers.  The |psi|^2 fill is
 * performed by cw_compute_local_probabilities (host-side loop after a
 * sync_host_state, identical algorithm on every backend).
 * ========================================================================= */
static PyObject *
py_get_local_probabilities(PyObject *module, PyObject *arg)
{
    Context *ctx = as_context(arg, "get_local_probabilities");
    if (!ctx) return NULL;

    if (ctx->local_probabilities == NULL) {
        npy_intp dims[1] = { (npy_intp)ctx->local_i };
        PyArrayObject *buf =
            (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        if (!buf) return NULL;
        cw_attach_host_local_probabilities(ctx->handle, PyArray_DATA(buf),
                                            (int64_t)ctx->local_i);
        ctx->local_probabilities = buf;
    }

    int32_t error_code = 0;
    cw_compute_local_probabilities(ctx->handle, &error_code);
    if (error_code != 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "compute_local_probabilities failed (status %d)",
                     (int)error_code);
        return NULL;
    }

    Py_INCREF(ctx->local_probabilities);
    return (PyObject *)ctx->local_probabilities;
}


/* =========================================================================
 * Module definition.
 * ========================================================================= */
static PyMethodDef ContextWrapperMethods[] = {
    {"setup",                 py_setup,                 METH_VARARGS,
     "setup(ci_ptr, alloc_local) -> (Context, error_code)"},
    {"destroy",               py_destroy,               METH_O,
     "destroy(ctx) -> None"},
    {"get_state",             py_get_state,             METH_O,
     "get_state(ctx) -> (ndarray[complex128], 0)  -- zero-copy view"},
    {"set_state",             py_set_state,             METH_VARARGS,
     "set_state(ctx, state) -> error_code"},
    {"get_observables",       py_get_observables,       METH_O,
     "get_observables(ctx) -> (ndarray[float64], 0)  -- zero-copy view"},
    {"set_observables",       py_set_observables,       METH_VARARGS,
     "set_observables(ctx, obs) -> error_code"},
    {"get_expectation_value", py_get_expectation_value, METH_O,
     "get_expectation_value(ctx) -> (float, error_code)"},
    {"get_state_norm",        py_get_state_norm,        METH_O,
     "get_state_norm(ctx) -> (float, error_code)"},
    {"get_local_probabilities", py_get_local_probabilities, METH_O,
     "get_local_probabilities(ctx) -> ndarray[float64]  -- cached |state[:local_i]|**2"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "context_wrapper",
    "CPython extension wrapping the Fortran context_type with a "
    "Python-owned state buffer for true zero-copy access.",
    -1,
    ContextWrapperMethods
};


PyMODINIT_FUNC
PyInit_context_wrapper(void)
{
    import_array();

    if (PyType_Ready(&ContextType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&ContextType);
    if (PyModule_AddObject(m, "Context", (PyObject *)&ContextType) < 0) {
        Py_DECREF(&ContextType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
