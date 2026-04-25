from setuptools import setup, Extension
import pybind11
import os

nvcc_flags = ["-O3", "-arch=sm_75"] 

ext = Extension(
    "nms_cuda",
    sources=["bindings.cpp", "nms.cu"],
    include_dirs=[pybind11.get_include()],
    library_dirs=["/usr/local/cuda/lib64"],
    libraries=["cudart"],
    extra_compile_args={"nvcc": nvcc_flags},
    language="c++",
)

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    ext = CUDAExtension(
        "nms_cuda",
        sources=["bindings.cpp", "nms.cu"],
        extra_compile_args={"nvcc": nvcc_flags, "cxx": ["-O3"]},
    )
    setup(name="nms_cuda", ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
except ImportError:
    setup(name="nms_cuda", ext_modules=[ext])
