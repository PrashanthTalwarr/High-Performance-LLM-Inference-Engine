import torch
import torch.nn.functional as F
import math

class FlashAttention:
    def __init__(self):
        self.name = "FlashAttention"
    
    def forward(self, q, k, v, mask=None):
        """
        Improved implementation inspired by FlashAttention paper.
        Uses adaptive blocking and efficient numerics.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Scale query
        q = q / math.sqrt(head_dim)
        
        # For short sequences, use standard attention which is faster on GPU
        if seq_len <= 256:
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-1, -2))
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Compute context vectors
            context = torch.matmul(attn_probs, v)
            
            return context
        
        # For longer sequences, use block-wise processing
        # Choose block sizes adaptively based on sequence length
        block_size = min(256, seq_len // 4)
        
        # Initialize output
        context = torch.zeros_like(q)
        
        # Process blocks of queries
        for i in range(0, seq_len, block_size):
            i_end = min(i + block_size, seq_len)
            q_block = q[:, :, i:i_end, :]
            
            # Compute attention scores for this block
            scores = torch.matmul(q_block, k.transpose(-1, -2))
            
            # Apply mask if needed
            if mask is not None:
                scores = scores.masked_fill(mask[:, :, i:i_end, :] == 0, -1e9)
            
            # Use a more numerically stable softmax
            scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
            scores = scores - scores_max
            attention_weights = torch.exp(scores)
            attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = attention_weights / (attention_weights_sum + 1e-6)
            
            # Compute context vectors for this block
            context[:, :, i:i_end, :] = torch.matmul(attention_weights, v)
        
        return context


class MemoryEfficientAttention:
    def __init__(self):
        self.name = "Memory-Efficient"
    
    def forward(self, q, k, v, mask=None):
        """
        More efficient implementation with optimized chunk size
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Scale query
        q = q / math.sqrt(head_dim)
        
        # Use a different strategy for different sequence lengths
        # For shorter sequences, process everything at once
        if seq_len <= 256:
            # Standard attention for small sequences
            attn_scores = torch.matmul(q, k.transpose(-1, -2))
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_probs = F.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, v)
        
        # For longer sequences, use chunked computation to save memory
        output = torch.zeros_like(q)
        
        # Adaptive chunk size based on sequence length
        chunk_size = min(128, seq_len // 2)
        
        for chunk_idx in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_idx + chunk_size, seq_len)
            
            # Current chunk of queries
            q_chunk = q[:, :, chunk_idx:chunk_end, :]
            
            # Compute attention scores for this chunk more efficiently
            chunk_scores = torch.bmm(
                q_chunk.view(-1, chunk_end - chunk_idx, head_dim),
                k.view(-1, seq_len, head_dim).transpose(1, 2)
            ).view(batch_size, num_heads, chunk_end - chunk_idx, seq_len)
            
            # Apply mask if provided
            if mask is not None:
                chunk_mask = mask[:, :, chunk_idx:chunk_end, :]
                chunk_scores = chunk_scores.masked_fill(chunk_mask == 0, -1e9)
            
            # Apply softmax for this chunk
            chunk_probs = F.softmax(chunk_scores, dim=-1)
            
            # Compute weighted sum for this chunk more efficiently
            chunk_output = torch.bmm(
                chunk_probs.view(-1, chunk_end - chunk_idx, seq_len),
                v.view(-1, seq_len, head_dim)
            ).view(batch_size, num_heads, chunk_end - chunk_idx, head_dim)
            
            # Store result
            output[:, :, chunk_idx:chunk_end, :] = chunk_output
        
        return output