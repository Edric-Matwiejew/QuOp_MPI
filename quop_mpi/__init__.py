from . import algorithm, config, observable, param, propagator, state, toolkit
from ._utils._deprecation import deprecated_alias_getattr
from .ansatz import Ansatz
from .unitary import UnitaryBase

__all__ = [
    "UnitaryBase",
    "Unitary",
    "Ansatz",
    "propagator",
    "observable",
    "state",
    "param",
    "toolkit",
    "algorithm",
    "meta",
    "config",
]

__getattr__ = deprecated_alias_getattr(__name__, globals(), {"Unitary": "UnitaryBase"})
