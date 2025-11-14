#include <torch/extension.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

// CUDA kernel for fused attention calculation
template <typename scalar_t>
__global__ void fused_attention_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements to process
    const int total_elements = batch_size * num_heads * seq_len;
    
    // Stride between threads
    const int stride = blockDim.x * gridDim.x;
    
    // Loop over elements with stride
    for (int i = idx; i < total_elements; i += stride) {
        // Calculate indices
        const int b = i / (num_heads * seq_len);
        const int h = (i / seq_len) % num_heads;
        const int s = i % seq_len;
        
        // Find maximum score for numerical stability
        scalar_t max_val = -INFINITY;
        
        // Loop through sequence to find max score
        for (int j = 0; j <= s; j++) { // Causal attention: only up to current position
            scalar_t score = 0;
            
            // Compute dot product between q and k
            for (int d = 0; d < head_dim; d++) {
                const int q_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
                const int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                score += q[q_idx] * k[k_idx];
            }
            
            // Scale by sqrt(head_dim)
            score /= sqrt(static_cast<scalar_t>(head_dim));
            
            // Update max
            max_val = max(max_val, score);
        }
        
        // Compute softmax denominator
        scalar_t denominator = 0;
        for (int j = 0; j <= s; j++) { // Causal attention: only up to current position
            scalar_t score = 0;
            
            // Compute dot product
            for (int d = 0; d < head_dim; d++) {
                const int q_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
                const int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                score += q[q_idx] * k[k_idx];
            }
            
            // Scale and apply softmax
            score /= sqrt(static_cast<scalar_t>(head_dim));
            score = exp(score - max_val);
            denominator += score;
        }
        
        // For each output dimension
        for (int d = 0; d < head_dim; d++) {
            scalar_t weighted_sum = 0;
            
            // Compute weighted average
            for (int j = 0; j <= s; j++) { // Causal attention
                scalar_t score = 0;
                
                // Compute dot product
                for (int d2 = 0; d2 < head_dim; d2++) {
                    const int q_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d2;
                    const int k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d2;
                    score += q[q_idx] * k[k_idx];
                }
                
                // Scale and apply softmax
                score /= sqrt(static_cast<scalar_t>(head_dim));
                score = exp(score - max_val) / denominator;
                
                // Multiply by value and accumulate
                const int v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                weighted_sum += score * v[v_idx];
            }
            
            // Write output
            const int out_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
            output[out_idx] = weighted_sum;
        }
    }
}

// Wrapper function for different datatypes
torch::Tensor fused_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    // Get dimensions
    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);
    
    // Create output tensor
    auto output = torch::zeros_like(q);
    
    // Set up CUDA parameters
    const int threads = 256;
    const int blocks = (batch_size * num_heads * seq_len + threads - 1) / threads;
    
    // Launch CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(q.type(), "fused_attention_kernel", ([&] {
        fused_attention_kernel<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim
        );
    }));
    
    return output;
}