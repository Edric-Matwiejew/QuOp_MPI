"""Wavefront backend extension exports."""

from importlib import import_module

_MODULE_ALIASES = {
    "context_wrapper": "context_wrapper",
    "diagonal_propagator": "wavefront_diagonal_propagator",
    "sparse_propagator": "wavefront_sparse_propagator",
    "circulant_propagator": "wavefront_circulant_propagator",
    "composite_propagator": "wavefront_composite_propagator",
    "momentum_propagator": "wavefront_momentum_propagator",
    "transverse_field_propagator": "wavefront_transverse_field_propagator",
}

__all__ = list(_MODULE_ALIASES)


def __getattr__(name):
    module_name = _MODULE_ALIASES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
