import atexit
import io
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime

from mpi4py import MPI

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PROFILER_FILE = os.path.abspath(__file__)
_TRACE_DIR = None
_PROFILE_LOG_PATH = None
_LINE_TRACE_PATH = None
_LINE_TRACE_HISTORY_PATH = None
_TRACE_THREAD = None
_TRACE_STOP = None
_TRACE_INTERVAL = 1.0
_MAIN_THREAD_ID = None
_RANK = None
_HOSTNAME = socket.gethostname()
_WRITE_LOCK = threading.Lock()
_LAST_FOCUS = None
_PREVIOUS_SIGNAL_HANDLERS = {}
_TRACE_RUN_ID = None

_call_stack = []
_start_times = {}


def _relfile(fullname):
    try:
        return os.path.relpath(fullname, _PACKAGE_ROOT)
    except ValueError:
        return fullname


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _resolve_run_id():
    explicit = os.getenv("QUOP_TRACE_RUN_ID")
    if explicit:
        return explicit

    for name in (
        "PMIX_NAMESPACE",
        "OMPI_MCA_ess_base_jobid",
        "SLURM_JOB_ID",
        "PMI_JOBID",
        "PBS_JOBID",
        "LSB_JOBID",
        "COBALT_JOBID",
    ):
        value = os.getenv(name)
        if value:
            return value

    return "latest"


def _format_frame(frame):
    if frame is None:
        return "<no-python-frame>"

    cls = ""
    if "self" in frame.f_locals:
        cls = frame.f_locals["self"].__class__.__name__ + "."

    relfile = _relfile(frame.f_code.co_filename)
    return f"{relfile}:{frame.f_lineno} in {cls}{frame.f_code.co_name}"


def _iter_stack(frame):
    while frame is not None:
        yield frame
        frame = frame.f_back


def _select_focus_frame(frame):
    fallback = frame
    quop_fallback = None

    for current in _iter_stack(frame):
        fullname = os.path.abspath(current.f_code.co_filename)
        if fullname == _PROFILER_FILE:
            continue
        if quop_fallback is None:
            quop_fallback = current
        if fullname.startswith(_PACKAGE_ROOT):
            return current

    if quop_fallback is not None:
        return quop_fallback

    return fallback


def _atomic_write(path, payload):
    temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with io.open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(temp_path, path)


def _append_line(path, line):
    with io.open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _format_snapshot(reason):
    frames = sys._current_frames()
    frame = frames.get(_MAIN_THREAD_ID)
    focus = _select_focus_frame(frame)

    lines = [
        f"timestamp: {_timestamp()}",
        f"reason: {reason}",
        f"run_id: {_TRACE_RUN_ID}",
        f"rank: {_RANK}",
        f"pid: {os.getpid()}",
        f"hostname: {_HOSTNAME}",
        f"focus: {_format_frame(focus)}",
        "stack:",
    ]

    if frame is None:
        lines.append("  <main-thread frame unavailable>")
    else:
        for depth, current in enumerate(_iter_stack(frame)):
            marker = "*" if current is focus else " "
            lines.append(f"{marker} {depth:02d}: {_format_frame(current)}")

    other_threads = []
    for thread in threading.enumerate():
        ident = thread.ident
        if ident is None or ident == _MAIN_THREAD_ID:
            continue
        thread_frame = frames.get(ident)
        other_threads.append((thread.name, ident, _format_frame(_select_focus_frame(thread_frame))))

    if other_threads:
        lines.append("other_threads:")
        for name, ident, summary in sorted(other_threads, key=lambda item: item[0]):
            lines.append(f"  {name} [{ident}]: {summary}")

    return "\n".join(lines) + "\n", focus


def _write_line_snapshot(reason):
    global _LAST_FOCUS

    if _LINE_TRACE_PATH is None:
        return

    snapshot, focus = _format_snapshot(reason)

    with _WRITE_LOCK:
        _atomic_write(_LINE_TRACE_PATH, snapshot)

        focus_summary = _format_frame(focus)
        if _LINE_TRACE_HISTORY_PATH is not None and focus_summary != _LAST_FOCUS:
            timestamp = _timestamp()
            _append_line(
                _LINE_TRACE_HISTORY_PATH,
                f"{timestamp} | {reason} | {focus_summary}\n",
            )
            _LAST_FOCUS = focus_summary


def _heartbeat_loop():
    while _TRACE_STOP is not None and not _TRACE_STOP.wait(_TRACE_INTERVAL):
        _write_line_snapshot("heartbeat")


def _signal_dump(signum, frame):
    _write_line_snapshot(f"signal {signum}")

    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
    if callable(previous):
        previous(signum, frame)


def _install_signal_handlers():
    for name in ("SIGUSR1", "SIGUSR2"):
        if not hasattr(signal, name):
            continue

        signum = getattr(signal, name)
        previous = signal.getsignal(signum)
        if previous is _signal_dump:
            continue

        _PREVIOUS_SIGNAL_HANDLERS[signum] = previous
        signal.signal(signum, _signal_dump)


def _shutdown():
    if _TRACE_STOP is not None:
        _TRACE_STOP.set()
    if _LINE_TRACE_HISTORY_PATH is not None:
        _append_line(_LINE_TRACE_HISTORY_PATH, f"{_timestamp()} | atexit\n")


def _parse_interval():
    raw_value = os.getenv("QUOP_LINE_TRACE_INTERVAL", "1.0")
    try:
        interval = float(raw_value)
    except ValueError:
        interval = 1.0
    return max(interval, 0.05)


def _trace_dir_prefix():
    if os.getenv("QUOP_LINE_TRACE") == "1":
        return "quop_line_trace"
    return "quop_profile"


def _ensure_trace_dir():
    global _TRACE_DIR
    global _TRACE_RUN_ID

    if _TRACE_DIR is not None:
        return _TRACE_DIR

    requested_dir = os.getenv("QUOP_TRACE_DIR")
    _TRACE_RUN_ID = _resolve_run_id()

    if requested_dir:
        folder = os.path.abspath(requested_dir)
    else:
        folder = os.path.abspath(f"{_trace_dir_prefix()}_{_TRACE_RUN_ID}")

    os.makedirs(folder, exist_ok=True)
    _TRACE_DIR = folder
    return _TRACE_DIR


def _reset_rank_history_file():
    global _LAST_FOCUS

    if _LINE_TRACE_HISTORY_PATH is None:
        return

    with io.open(_LINE_TRACE_HISTORY_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"# run_id: {_TRACE_RUN_ID}\n")
        handle.write(f"# rank: {_RANK}\n")
        handle.write(f"# pid: {os.getpid()}\n")
        handle.write(f"# hostname: {_HOSTNAME}\n")
        handle.write(f"# started: {_timestamp()}\n")

    _LAST_FOCUS = None


def _enable_line_trace(comm):
    global _LINE_TRACE_PATH
    global _LINE_TRACE_HISTORY_PATH
    global _TRACE_THREAD
    global _TRACE_STOP
    global _TRACE_INTERVAL
    global _MAIN_THREAD_ID

    if _TRACE_THREAD is not None and _TRACE_THREAD.is_alive():
        return

    trace_dir = _ensure_trace_dir()
    _TRACE_INTERVAL = _parse_interval()
    _MAIN_THREAD_ID = threading.main_thread().ident or threading.get_ident()
    _LINE_TRACE_PATH = os.path.join(trace_dir, f"rank_{_RANK}.status")
    _LINE_TRACE_HISTORY_PATH = os.path.join(trace_dir, f"rank_{_RANK}.history")
    _TRACE_STOP = threading.Event()

    _reset_rank_history_file()
    _install_signal_handlers()
    _write_line_snapshot("startup")

    _TRACE_THREAD = threading.Thread(
        target=_heartbeat_loop,
        name=f"quop-line-trace-{_RANK}",
        daemon=True,
    )
    _TRACE_THREAD.start()


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

        relfile = _relfile(frame.f_code.co_filename)
        lineno = frame.f_lineno

        with io.open(_PROFILE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                f"{indent}{MPI.COMM_WORLD.Get_rank()}, "
                f"{relfile}:{lineno}, {cls}{func}, {elapsed:.6f}s\n"
            )

        _call_stack.pop()

    return _profiler


def enable():
    """
    Enable opt-in tracing tools.

    Environment variables
    ---------------------
    QUOP_PROFILE=1
        Record per-rank function return timings under the trace directory.
    QUOP_LINE_TRACE=1
        Write a per-rank heartbeat file with the current Python source line and
        stack, plus a history file of line changes. This is intended for MPI
        hangs where ranks must be force-quit.
    QUOP_LINE_TRACE_INTERVAL=<seconds>
        Heartbeat period for QUOP_LINE_TRACE. Defaults to 1.0 seconds.
    QUOP_TRACE_DIR=<path>
        Optional directory to reuse instead of creating a timestamped folder.
    """
    global _PROFILE_LOG_PATH
    global _RANK

    profile_enabled = os.getenv("QUOP_PROFILE") == "1"
    line_trace_enabled = os.getenv("QUOP_LINE_TRACE") == "1"

    if not profile_enabled and not line_trace_enabled:
        return

    # Don't enable profiling if MPI is not properly initialized
    # (e.g., when imported by Sphinx for documentation)
    if not MPI.Is_initialized():
        return

    comm = MPI.COMM_WORLD
    _RANK = comm.Get_rank()
    trace_dir = _ensure_trace_dir()

    if profile_enabled:
        _PROFILE_LOG_PATH = os.path.join(trace_dir, f"trace_{_RANK}.txt")
        sys.setprofile(_profiler)

    if line_trace_enabled:
        _enable_line_trace(comm)


# auto-enable when imported
atexit.register(_shutdown)
enable()
