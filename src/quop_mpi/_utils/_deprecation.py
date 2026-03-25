"""Shared helper for module-level deprecation aliases (PEP 562)."""

from __future__ import annotations

import warnings


def deprecated_alias_getattr(
    module_name: str,
    module_globals: dict,
    deprecated_aliases: dict[str, str],
):
    """Return a module-level ``__getattr__`` that emits :class:`DeprecationWarning`
    for old names and resolves them to the new names already present in
    *module_globals*.

    Parameters
    ----------
    module_name
        The ``__name__`` of the calling module (used in the warning message).
    module_globals
        The ``globals()`` dict of the calling module, used to look up the
        replacement object.
    deprecated_aliases
        Mapping of ``{"old_name": "new_name", ...}``.

    Returns
    -------
    Callable[[str], Any]
        A ``__getattr__`` function suitable for assignment at module scope.

    Notes
    -----
    When a class is imported from a submodule (e.g. ``from .unitary import
    Unitary``), Python implicitly binds the submodule itself as an
    attribute of the parent package.  If the deprecated alias has the same name
    as that submodule, the implicit binding shadows ``__getattr__``.  This
    function therefore pops any conflicting names from *module_globals* so that
    attribute lookup falls through to ``__getattr__``.  Direct imports of the
    submodule (``import pkg.submodule``) are unaffected because they resolve
    via ``sys.modules``.
    """

    # Remove implicit submodule bindings that would shadow __getattr__.
    for old_name in deprecated_aliases:
        module_globals.pop(old_name, None)

    def _module_getattr(name: str):
        if name in deprecated_aliases:
            new_name = deprecated_aliases[name]
            warnings.warn(
                f"{module_name}.{name} is deprecated, use {new_name} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            obj = module_globals[new_name]
            # Cache so subsequent lookups hit __dict__ directly and
            # avoid a second warning from the import machinery.
            module_globals[name] = obj
            return obj
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return _module_getattr
