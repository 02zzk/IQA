"""
Optimization Configuration for PyIQA

This module contains configuration settings for various performance optimizations
that can be applied to the PyIQA library.
"""

import os
import torch
from typing import Dict, Any, Optional

class OptimizationConfig:
    """Configuration class for PyIQA performance optimizations."""
    
    def __init__(self):
        # Import optimization settings
        self.lazy_imports = True
        self.enable_caching = True
        self.cache_max_size = 10  # Maximum number of cached models
        
        # Memory optimization settings
        self.memory_efficient = True
        self.enable_mixed_precision = False  # Enable only if supported
        self.garbage_collect_frequency = 10  # Clean up every N inferences
        
        # Device optimization settings
        self.optimize_device_transfers = True
        self.use_non_blocking_transfers = True
        self.pin_memory = True
        
        # Model optimization settings
        self.enable_jit_compilation = True
        self.enable_model_fusion = True
        self.use_cudnn_benchmark = True
        
        # Image processing optimizations
        self.image_backend = 'auto'  # 'auto', 'pil', 'cv2'
        self.use_optimized_transforms = True
        self.enable_contiguous_tensors = True
        
        # Parallel processing settings
        self.num_workers = min(4, os.cpu_count() or 1)
        self.prefetch_factor = 2
        
        # Environment-based auto-configuration
        self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure based on system capabilities."""
        # Check CUDA availability for mixed precision
        if torch.cuda.is_available():
            # Enable mixed precision for modern GPUs
            if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
                self.enable_mixed_precision = True
            
            # Enable CUDNN benchmark for better performance
            self.use_cudnn_benchmark = True
            
            # Adjust memory settings based on GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 4 * 1024**3:  # Less than 4GB
                self.memory_efficient = True
                self.cache_max_size = 5
        else:
            # CPU-only optimizations
            self.enable_mixed_precision = False
            self.use_cudnn_benchmark = False
            self.num_workers = min(2, os.cpu_count() or 1)
        
        # Check for environment variables
        if os.environ.get('PYIQA_DISABLE_OPTIMIZATIONS'):
            self.disable_all_optimizations()
        
        if os.environ.get('PYIQA_MEMORY_EFFICIENT'):
            self.memory_efficient = True
            self.cache_max_size = 3
    
    def disable_all_optimizations(self):
        """Disable all optimizations for debugging or compatibility."""
        self.lazy_imports = False
        self.enable_caching = False
        self.memory_efficient = False
        self.enable_mixed_precision = False
        self.optimize_device_transfers = False
        self.enable_jit_compilation = False
        self.enable_model_fusion = False
        self.use_cudnn_benchmark = False
        self.use_optimized_transforms = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration option: {key}")
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get configuration specifically for inference models."""
        return {
            'memory_efficient': self.memory_efficient,
            'enable_mixed_precision': self.enable_mixed_precision,
            'optimize_device_transfers': self.optimize_device_transfers,
            'use_non_blocking_transfers': self.use_non_blocking_transfers,
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for model creation."""
        return {
            'use_cache': self.enable_caching,
            'enable_jit_compilation': self.enable_jit_compilation,
            'enable_model_fusion': self.enable_model_fusion,
        }
    
    def apply_torch_optimizations(self):
        """Apply PyTorch-level optimizations."""
        if torch.cuda.is_available() and self.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Set number of threads for CPU operations
        if not torch.cuda.is_available():
            torch.set_num_threads(self.num_workers)
    
    def print_config(self):
        """Print current configuration."""
        print("PyIQA Optimization Configuration:")
        print("-" * 40)
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")

# Global configuration instance
_global_config = OptimizationConfig()

def get_config() -> OptimizationConfig:
    """Get the global optimization configuration."""
    return _global_config

def set_config(**kwargs):
    """Update the global optimization configuration."""
    _global_config.update(**kwargs)

def reset_config():
    """Reset configuration to defaults."""
    global _global_config
    _global_config = OptimizationConfig()

# Convenience functions for common optimization scenarios

def optimize_for_speed():
    """Configure for maximum speed."""
    set_config(
        enable_caching=True,
        memory_efficient=False,
        enable_mixed_precision=True,
        enable_jit_compilation=True,
        use_cudnn_benchmark=True,
        garbage_collect_frequency=20
    )

def optimize_for_memory():
    """Configure for minimum memory usage."""
    set_config(
        memory_efficient=True,
        enable_caching=False,
        cache_max_size=3,
        garbage_collect_frequency=5,
        num_workers=1
    )

def optimize_for_accuracy():
    """Configure for maximum accuracy (disable potentially lossy optimizations)."""
    set_config(
        enable_mixed_precision=False,
        enable_jit_compilation=False,
        use_cudnn_benchmark=False,
        memory_efficient=False
    )

# Environment variable support
def load_config_from_env():
    """Load configuration from environment variables."""
    config = get_config()
    
    # Map environment variables to config options
    env_mapping = {
        'PYIQA_LAZY_IMPORTS': 'lazy_imports',
        'PYIQA_ENABLE_CACHING': 'enable_caching', 
        'PYIQA_MEMORY_EFFICIENT': 'memory_efficient',
        'PYIQA_MIXED_PRECISION': 'enable_mixed_precision',
        'PYIQA_JIT_COMPILATION': 'enable_jit_compilation',
        'PYIQA_CUDNN_BENCHMARK': 'use_cudnn_benchmark',
        'PYIQA_NUM_WORKERS': 'num_workers',
        'PYIQA_CACHE_SIZE': 'cache_max_size',
    }
    
    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert string to appropriate type
            if config_key in ['num_workers', 'cache_max_size', 'garbage_collect_frequency']:
                value = int(value)
            elif config_key in ['lazy_imports', 'enable_caching', 'memory_efficient', 
                               'enable_mixed_precision', 'enable_jit_compilation', 'use_cudnn_benchmark']:
                value = value.lower() in ('true', '1', 'yes', 'on')
            
            config.update(**{config_key: value})

# Load configuration from environment on import
load_config_from_env()