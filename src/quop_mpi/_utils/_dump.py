# src/quop_mpi/_utils/_dump.py
"""Diagnostic dump utility for quop_mpi_layout_t.

Triggered by the ``QUOP_DUMP_COMM_INFO`` environment variable:

- ``QUOP_DUMP_COMM_INFO=1``    -> dump to CWD
- ``QUOP_DUMP_COMM_INFO=<dir>`` -> dump to *<dir>* (created if needed)
- unset                         -> no-op (zero overhead)

Only rank 0 of the layout's root communicator writes the file.

The implementation is entirely in Fortran (``comm_info_module::layout_dump_comm_info``).
This thin Python wrapper simply forwards the opaque handle and phase string.
"""

from quop_mpi._lib.comm_info_wrapper import comm_info_wrapper as _ciw


def dump_comm_info(comm_info, phase="init", output_dir=None):
    """Forward the dump request to the Fortran-native implementation.

    Parameters
    ----------
    comm_info : QuopMpiLayout
        The QuopMpiLayout handle wrapping a Fortran quop_mpi_layout_t.
    phase : str
        Label for this dump point (``"init"`` or ``"locked"``).
    output_dir : str, optional
        Ignored -- the Fortran side reads QUOP_DUMP_COMM_INFO directly.
        Retained for API compatibility.
    """
    ptr = getattr(comm_info, "_ptr", None)
    if ptr is None:
        return
    _ciw.wrapper_dump_comm_info(ptr, phase)
