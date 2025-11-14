#include <torch/extension.h>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif
   
   // Forward declaration of CUDA functions
   torch::Tensor fused_attention_cuda(
       torch::Tensor q,
       torch::Tensor k,
       torch::Tensor v
   );
   
   // Wrapper function
   torch::Tensor fused_attention_forward(
       torch::Tensor q,
       torch::Tensor k,
       torch::Tensor v
   ) {
       // Check input dimensions
       TORCH_CHECK(q.dim() == 4, "Query tensor must be 4-dimensional");
       TORCH_CHECK(k.dim() == 4, "Key tensor must be 4-dimensional");
       TORCH_CHECK(v.dim() == 4, "Value tensor must be 4-dimensional");
       
       // Check that tensors are on CUDA
       TORCH_CHECK(q.is_cuda(), "Query tensor must be on CUDA");
       TORCH_CHECK(k.is_cuda(), "Key tensor must be on CUDA");
       TORCH_CHECK(v.is_cuda(), "Value tensor must be on CUDA");
       
       // Check dimensions match
       TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "Batch size mismatch");
       TORCH_CHECK(q.size(1) == k.size(1) && k.size(1) == v.size(1), "Number of heads mismatch");
       TORCH_CHECK(q.size(2) == v.size(2), "Sequence length mismatch for query and value");
       TORCH_CHECK(k.size(2) == v.size(2), "Sequence length mismatch for key and value");
       TORCH_CHECK(q.size(3) == k.size(3), "Head dimension mismatch for query and key");
       TORCH_CHECK(k.size(3) == v.size(3), "Head dimension mismatch for key and value");
       
       // Call CUDA implementation
       return fused_attention_cuda(q, k, v);
   }
   
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("fused_attention_forward", &fused_attention_forward, "Fused attention forward");
   }