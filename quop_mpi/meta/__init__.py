from .._utils._deprecation import deprecated_alias_getattr
from .swarm import Swarm

__all__ = ["Swarm", "swarm"]

__getattr__ = deprecated_alias_getattr(__name__, globals(), {"swarm": "Swarm"})
