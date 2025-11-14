# demo_attention.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now the imports will work
from llm_inference_engine.src.baseline import BaselineAttention
from llm_inference_engine.src.optimized_attention import FlashAttention, MemoryEfficientAttention
from llm_inference_engine.src.cuda_attention import CUDAAttention

# Add the package to path
sys.path.append(str(Path(__file__).parent))

from llm_inference_engine.src.baseline import BaselineAttention
from llm_inference_engine.src.optimized_attention import FlashAttention, MemoryEfficientAttention
from llm_inference_engine.src.cuda_attention import CUDAAttention

def benchmark_attention(name, implementation, batch_size, num_heads, seq_lens, head_dim, num_runs=10):
    """Benchmark attention implementation across different sequence lengths."""
    results = []
    
    for seq_len in seq_lens:
        print(f"Testing {name} with sequence length {seq_len}...")
        
        # Create random tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = implementation.forward(q, k, v)
        
        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = implementation.forward(q, k, v)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        results.append(avg_time * 1000)  # Convert to ms
        
        print(f"  Average time: {avg_time * 1000:.2f} ms")
    
    return results

def run_benchmarks():
    # Parameters
    batch_size = 1
    num_heads = 8
    seq_lens = [128, 256, 512, 1024]
    head_dim = 64
    num_runs = 10
    
    # Create implementations
    baseline = BaselineAttention()
    flash = FlashAttention()
    memory_efficient = MemoryEfficientAttention()
    cuda_optimized = CUDAAttention()
    
    # Run benchmarks
    print("Running benchmarks...")
    results = {
        "Baseline": benchmark_attention("Baseline", baseline, batch_size, num_heads, seq_lens, head_dim, num_runs),
        "Flash": benchmark_attention("Flash Attention", flash, batch_size, num_heads, seq_lens, head_dim, num_runs),
        "Memory-Efficient": benchmark_attention("Memory-Efficient", memory_efficient, batch_size, num_heads, seq_lens, head_dim, num_runs),
        "CUDA-Optimized": benchmark_attention("CUDA-Optimized", cuda_optimized, batch_size, num_heads, seq_lens, head_dim, num_runs)
    }
    
    # Calculate speedups
    baseline_times = results["Baseline"]
    speedups = {}
    
    for name, times in results.items():
        if name != "Baseline":
            speedups[name] = [b/t for b, t in zip(baseline_times, times)]
            print(f"\nSpeedups for {name}:")
            for seq_len, speedup in zip(seq_lens, speedups[name]):
                print(f"  Seq len {seq_len}: {speedup:.2f}x faster")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    width = 0.2
    x = np.arange(len(seq_lens))
    
    for i, (name, times) in enumerate(results.items()):
        plt.bar(x + i*width - 0.3, times, width, label=name)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Implementation Performance')
    plt.xticks(x, seq_lens)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('benchmark_results.png')
    plt.show()
    
    # Plot speedups
    plt.figure(figsize=(12, 6))
    
    for name, speedup in speedups.items():
        plt.plot(seq_lens, speedup, marker='o', label=f"{name}")
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup (x times faster)')
    plt.title('Speedup Over Baseline')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('speedup_results.png')
    plt.show()
    
    print("\nResults saved as 'benchmark_results.png' and 'speedup_results.png'")
    
    # Display optimization techniques
    print("\nCUDA Optimization Techniques:")
    if hasattr(cuda_optimized, 'describe_optimizations'):
        for i, opt in enumerate(cuda_optimized.describe_optimizations(), 1):
            print(f"{i}. {opt}")

if __name__ == "__main__":
    run_benchmarks()