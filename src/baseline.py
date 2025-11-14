import torch
import torch.nn.functional as F
import time
   
class BaselineAttention:
    def __init__(self):
        self.name = "Baseline"
    
    def forward(self, q, k, v, mask=None):
        """Standard attention implementation from the Transformer paper."""
        # q, k, v shape: [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / (head_dim ** 0.5)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        # [batch_size, num_heads, seq_len, seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_probs, v)
        
        return context

# Benchmarking function
def benchmark_attention(attn_impl, batch_size, num_heads, seq_len, head_dim, num_runs=10):
    """Measure the execution time of an attention implementation."""
    # Create random tensors on GPU (if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    # Create a causal mask (optional)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    
    # Warm-up runs
    for _ in range(5):
        with torch.no_grad():
            _ = attn_impl.forward(q, k, v, mask)
    
    # Ensure all CUDA operations are completed
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure execution time
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attn_impl.forward(q, k, v, mask)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time