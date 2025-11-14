from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA path from environment or use default
cuda_path = os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x')
# Replace v12.x with your actual CUDA version
cuda_include = os.path.join(cuda_path, 'include')

setup(
    name="llm_inference_engine",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_attention_ext",
            sources=[
                "llm_inference_engine/src/cuda/attention_kernel.cu",
                "llm_inference_engine/src/cuda/cuda_extension.cpp",
            ],
            include_dirs=[
                cuda_include,
            ],
            extra_compile_args={
                'cxx': ['/Ox', '/std:c++17'],
                'nvcc': ['-O3', '--use_fast_math', '--extended-lambda']
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
