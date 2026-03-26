from ..._utils._deprecation import deprecated_alias_getattr
from . import operator
from .unitary import Unitary

__all__ = ["Unitary", "unitary", "operator"]

__getattr__ = deprecated_alias_getattr(__name__, globals(), {"unitary": "Unitary"})
