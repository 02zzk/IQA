# flake8: noqa
# Lazy imports for performance optimization
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api_helpers import create_metric, list_models, get_dataset_info
    from .version import __gitsha__, __version__

def __getattr__(name):
    """Lazy import to improve startup time by deferring heavy imports."""
    if name == 'create_metric':
        from .api_helpers import create_metric
        return create_metric
    elif name == 'list_models':
        from .api_helpers import list_models
        return list_models
    elif name == 'get_dataset_info':
        from .api_helpers import get_dataset_info
        return get_dataset_info
    elif name == '__version__':
        from .version import __version__
        return __version__
    elif name == '__gitsha__':
        from .version import __gitsha__
        return __gitsha__
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name__}")

# Keep these as module-level attributes for backward compatibility
__all__ = ['create_metric', 'list_models', 'get_dataset_info', '__version__', '__gitsha__']