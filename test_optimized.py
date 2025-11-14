import torch
from src.baseline import BaselineAttention, benchmark_attention
from src.optimized_attention import MemoryEfficientAttention, FlashAttention

# Test parameters
batch_size, num_heads, seq_len, head_dim = 1, 4, 512, 64

# Create random tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

# Create implementations
baseline = BaselineAttention()
memory_efficient = MemoryEfficientAttention()
flash = FlashAttention()

# Compute outputs
with torch.no_grad():
    baseline_out = baseline.forward(q, k, v)
    memory_efficient_out = memory_efficient.forward(q, k, v)
    flash_out = flash.forward(q, k, v)

# Check that results are similar
print("Comparing Memory-Efficient to Baseline:")
diff = torch.abs(baseline_out - memory_efficient_out).mean().item()
print(f"Mean absolute difference: {diff}")

print("\nComparing Flash to Baseline:")
diff = torch.abs(baseline_out - flash_out).mean().item()
print(f"Mean absolute difference: {diff}")

# Benchmark
print("\nBenchmarking:")
baseline_time = benchmark_attention(baseline, batch_size, num_heads, seq_len, head_dim)
memory_time = benchmark_attention(memory_efficient, batch_size, num_heads, seq_len, head_dim)
flash_time = benchmark_attention(flash, batch_size, num_heads, seq_len, head_dim)

print(f"Baseline: {baseline_time * 1000:.2f} ms")
print(f"Memory-Efficient: {memory_time * 1000:.2f} ms (speedup: {baseline_time/memory_time:.2f}x)")
print(f"Flash: {flash_time * 1000:.2f} ms (speedup: {baseline_time/flash_time:.2f}x)")