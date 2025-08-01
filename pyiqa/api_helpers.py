import fnmatch
import re
import weakref
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.dataset_info import DATASET_INFO

from pyiqa.utils import get_root_logger

# Global model cache to avoid recreating models
_model_cache = weakref.WeakValueDictionary()

def create_metric(metric_name, as_loss=False, device=None, use_cache=True, **kwargs):
    """Create a metric with optional caching for performance.
    
    Args:
        metric_name: Name of the metric to create
        as_loss: Whether to use as loss
        device: Device to use
        use_cache: Whether to use cached models (default: True)
        **kwargs: Additional arguments
    """
    assert metric_name in DEFAULT_CONFIGS.keys(), f'Metric {metric_name} not implemented yet.' 
    
    # Create cache key based on metric parameters
    cache_key = (metric_name, as_loss, str(device), frozenset(kwargs.items()) if kwargs else frozenset())
    
    # Try to get from cache first if caching is enabled
    if use_cache and cache_key in _model_cache:
        logger = get_root_logger()
        logger.info(f'Using cached metric [{metric_name}].')
        return _model_cache[cache_key]
    
    # Lazy import to avoid loading heavy dependencies at startup
    from pyiqa.models.inference_model import InferenceModel
    
    metric = InferenceModel(metric_name, as_loss=as_loss, device=device, **kwargs)
    logger = get_root_logger()
    logger.info(f'Metric [{metric.net.__class__.__name__}] is created.')
    
    # Cache the model if caching is enabled and not used as loss
    if use_cache and not as_loss:
        _model_cache[cache_key] = metric
    
    return metric


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(metric_mode=None, filter='', exclude_filters=''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
    Example:
        model_list('*ssim*') -- returns all models including 'ssim'
    """
    if metric_mode is None:
        all_models = DEFAULT_CONFIGS.keys()
    else:
        assert metric_mode in ['FR', 'NR'], f'Metric mode only support [FR, NR], but got {metric_mode}'
        all_models = [key for key in DEFAULT_CONFIGS.keys() if DEFAULT_CONFIGS[key]['metric_mode'] == metric_mode]

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    return list(sorted(models, key=_natural_key))


def get_dataset_info(dataset_name):
    assert dataset_name in DATASET_INFO.keys(), f'Dataset {dataset_name} not implemented yet.'
    return DATASET_INFO[dataset_name]
