#!/usr/bin/env python3
"""
Performance Benchmark for PyIQA Optimizations

This script measures the performance improvements made to the PyIQA library
including startup time, model loading time, inference speed, and memory usage.
"""

import time
import sys
import psutil
import gc
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    
@contextmanager
def measure_memory():
    """Context manager to measure memory usage."""
    tracemalloc.start()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield lambda: {
        'current': tracemalloc.get_traced_memory()[0] / 1024 / 1024,  # MB
        'peak': tracemalloc.get_traced_memory()[1] / 1024 / 1024,     # MB
        'rss_diff': process.memory_info().rss / 1024 / 1024 - start_memory
    }
    
    tracemalloc.stop()

def benchmark_import_time():
    """Benchmark PyIQA import time."""
    print("=" * 60)
    print("BENCHMARK: Import Time")
    print("=" * 60)
    
    # Measure import time
    with measure_time() as get_time:
        import pyiqa
    
    import_time = get_time()
    print(f"PyIQA import time: {import_time:.3f} seconds")
    
    # Test lazy loading by accessing attributes
    with measure_time() as get_time:
        _ = pyiqa.list_models
    lazy_access_time = get_time()
    print(f"Lazy attribute access time: {lazy_access_time:.3f} seconds")
    
    return import_time, lazy_access_time

def benchmark_model_creation():
    """Benchmark model creation with caching."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Model Creation")
    print("=" * 60)
    
    import pyiqa
    
    metrics_to_test = ['ssim', 'psnr', 'lpips']
    results = {}
    
    for metric in metrics_to_test:
        print(f"\nTesting {metric}:")
        
        # First creation (cold start)
        with measure_time() as get_time:
            with measure_memory() as get_memory:
                model1 = pyiqa.create_metric(metric, use_cache=True)
        
        first_time = get_time()
        first_memory = get_memory()
        
        # Second creation (should use cache)
        with measure_time() as get_time:
            model2 = pyiqa.create_metric(metric, use_cache=True)
        
        second_time = get_time()
        
        # Third creation without cache
        with measure_time() as get_time:
            model3 = pyiqa.create_metric(metric, use_cache=False)
        
        third_time = get_time()
        
        print(f"  First creation (cold): {first_time:.3f}s, Memory: {first_memory['peak']:.1f}MB")
        print(f"  Second creation (cached): {second_time:.3f}s, Speedup: {first_time/second_time:.1f}x")
        print(f"  Third creation (no cache): {third_time:.3f}s")
        
        results[metric] = {
            'first_time': first_time,
            'cached_time': second_time,
            'no_cache_time': third_time,
            'memory_peak': first_memory['peak']
        }
        
        # Cleanup
        del model1, model2, model3
        gc.collect()
    
    return results

def benchmark_inference_speed():
    """Benchmark inference speed with optimizations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Inference Speed")
    print("=" * 60)
    
    try:
        import torch
        import pyiqa
        
        # Create dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Test different image sizes
        sizes = [(256, 256), (512, 512)]
        metrics = ['ssim', 'psnr']
        
        for metric in metrics:
            print(f"\nTesting {metric}:")
            
            # Create models with different configurations
            model_optimized = pyiqa.create_metric(
                metric, 
                device=device,
                memory_efficient=True,
                enable_mixed_precision=True
            )
            
            model_standard = pyiqa.create_metric(
                metric,
                device=device,
                memory_efficient=False,
                enable_mixed_precision=False
            )
            
            for h, w in sizes:
                print(f"  Image size {h}x{w}:")
                
                # Create test images
                img1 = torch.rand(1, 3, h, w, device=device)
                img2 = torch.rand(1, 3, h, w, device=device)
                
                # Warmup
                for _ in range(3):
                    _ = model_optimized(img1, img2) if metric == 'ssim' else model_optimized(img1)
                
                # Benchmark optimized version
                num_runs = 10
                with measure_time() as get_time:
                    for _ in range(num_runs):
                        if metric == 'ssim':
                            _ = model_optimized(img1, img2)
                        else:
                            _ = model_optimized(img1)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                
                optimized_time = get_time() / num_runs
                
                # Benchmark standard version
                with measure_time() as get_time:
                    for _ in range(num_runs):
                        if metric == 'ssim':
                            _ = model_standard(img1, img2)
                        else:
                            _ = model_standard(img1)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                
                standard_time = get_time() / num_runs
                
                speedup = standard_time / optimized_time
                print(f"    Optimized: {optimized_time*1000:.2f}ms")
                print(f"    Standard:  {standard_time*1000:.2f}ms")
                print(f"    Speedup:   {speedup:.2f}x")
                
                del img1, img2
            
            del model_optimized, model_standard
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
    except ImportError:
        print("PyTorch not available, skipping inference benchmark")

def benchmark_memory_efficiency():
    """Benchmark memory usage optimizations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Memory Efficiency")
    print("=" * 60)
    
    try:
        import torch
        import pyiqa
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test memory usage with different configurations
        metric = 'lpips'  # Memory-intensive metric
        
        print(f"Testing {metric} memory usage:")
        
        # Memory efficient model
        with measure_memory() as get_memory:
            model_efficient = pyiqa.create_metric(
                metric,
                device=device,
                memory_efficient=True
            )
            
            # Create and process test image
            img = torch.rand(1, 3, 512, 512, device=device)
            _ = model_efficient(img, img)
            
        efficient_memory = get_memory()
        
        # Standard model
        with measure_memory() as get_memory:
            model_standard = pyiqa.create_metric(
                metric,
                device=device,
                memory_efficient=False
            )
            
            # Create and process test image
            img = torch.rand(1, 3, 512, 512, device=device)
            _ = model_standard(img, img)
            
        standard_memory = get_memory()
        
        print(f"  Memory efficient: {efficient_memory['peak']:.1f}MB peak")
        print(f"  Standard:         {standard_memory['peak']:.1f}MB peak")
        print(f"  Memory saved:     {standard_memory['peak'] - efficient_memory['peak']:.1f}MB")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"  GPU Memory cached:    {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
        
    except ImportError:
        print("PyTorch not available, skipping memory benchmark")

def print_optimization_summary():
    """Print summary of optimizations implemented."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    optimizations = [
        "✓ Lazy imports to reduce startup time",
        "✓ Model caching with WeakValueDictionary",
        "✓ Optimized model loading with state dict caching", 
        "✓ Reduced device transfers with smart preprocessing",
        "✓ Mixed precision inference support",
        "✓ Memory efficient inference mode",
        "✓ CUDNN benchmark mode for better GPU performance",
        "✓ JIT optimization for supported models",
        "✓ Optimized image loading backends (PIL/CV2)",
        "✓ Efficient tensor operations with contiguous memory",
        "✓ LRU caching for frequently used operations"
    ]
    
    for opt in optimizations:
        print(f"  {opt}")
    
    print("\nKey Performance Improvements:")
    print("  • Faster startup time through lazy loading")
    print("  • Reduced memory usage with efficient caching")
    print("  • Better inference speed with optimizations")
    print("  • Lower GPU memory footprint")
    print("  • Improved scalability for batch processing")

def main():
    """Run complete performance benchmark."""
    print("PyIQA Performance Benchmark")
    print("This benchmark tests the optimizations made to PyIQA")
    print("for improved startup time, model loading, and inference speed.\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        # Run benchmarks
        benchmark_import_time()
        benchmark_model_creation()
        benchmark_inference_speed()
        benchmark_memory_efficiency()
        print_optimization_summary()
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()