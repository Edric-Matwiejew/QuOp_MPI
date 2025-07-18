import os
import sys
import time
import io
from mpi4py import MPI

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_LOG_DIR = None
_LOG_PATH = None

_call_stack = []
_start_times = {}


def _profiler(frame, event, arg):
    module = frame.f_globals.get("__name__")
    if not isinstance(module, str) or not module.startswith("quop_mpi"):
        return

    if event == "call":
        _call_stack.append(frame)
        _start_times[frame] = time.perf_counter()

    elif event == "return":
        if not _call_stack or _call_stack[-1] is not frame:
            return
        t0 = _start_times.pop(frame)
        elapsed = time.perf_counter() - t0

        depth = len(_call_stack) - 1
        indent = "  " * depth

        func = frame.f_code.co_name
        cls = ""
        if "self" in frame.f_locals:
            cls = frame.f_locals["self"].__class__.__name__ + "."

        fullname = frame.f_code.co_filename
        try:
            relfile = os.path.relpath(fullname, _PACKAGE_ROOT)
        except ValueError:
            relfile = fullname
        lineno = frame.f_lineno

        with io.open(_LOG_PATH, "a") as f:
            f.write(
                f"{indent}{MPI.COMM_WORLD.Get_rank()}, "
                f"{relfile}:{lineno}, {cls}{func}, {elapsed:.6f}s\n"
            )

        _call_stack.pop()

    return _profiler


def enable():
    """
    Profile if QUOP_PROFILE=1
    """
    global _LOG_DIR, _LOG_PATH

    if os.getenv("QUOP_PROFILE") != "1":
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        folder = f"quop_profile_{timestamp}"
        os.makedirs(folder, exist_ok=False)
    else:
        folder = None

    _LOG_DIR = comm.bcast(folder, root=0)

    _LOG_PATH = os.path.join(_LOG_DIR, f"trace_{rank}.txt")

    sys.setprofile(_profiler)


# auto-enable when imported
enable()
