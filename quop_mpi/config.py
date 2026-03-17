"""Process-wide backend selection for QuOp extension modules.

Rules
-----
- ``QUOP_BACKEND`` is read once, when this module is imported.
- Supported values are ``"mpi"`` and ``"wavefront"``.
- Unset or invalid values fall back to ``"mpi"``.
- Changing ``QUOP_BACKEND`` after import has no effect.
- A single Python process may use only one backend.
- Launchers and tests must set ``QUOP_BACKEND`` before importing ``quop_mpi``.
"""

from os import environ

backends = ["mpi", "wavefront"]
env = environ.get("QUOP_BACKEND", "").strip()
backend = env if env in backends else "mpi"
