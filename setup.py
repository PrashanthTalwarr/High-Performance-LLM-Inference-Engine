from setuptools import setup, find_packages

setup(
    name="llm_inference_engine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers",
        "matplotlib",
        "numpy",
    ],
)