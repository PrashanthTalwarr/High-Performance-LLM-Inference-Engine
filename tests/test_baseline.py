# test_baseline.py
import torch
from src.baseline import BaselineAttention, benchmark_attention

# Test with small dimensions
batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 64
baseline = BaselineAttention()

# Run benchmark
avg_time = benchmark_attention(baseline, batch_size, num_heads, seq_len, head_dim)
print(f"Average execution time: {avg_time * 1000:.2f} ms")