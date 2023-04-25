import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dgNN",
    version="0.1",
    author="HenryChang fishming",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "dgNN"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension('multigpu_gcnconv',['dgNN/src/multigpu_gcnconv/multigpu_gcnconv.cpp','dgNN/src/multigpu_gcnconv/multigpu_gcnconv_kernel.cu'],extra_cflags=["-lcublas"],extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=[],extra_include_paths=['/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/nccl/include']),
        CUDAExtension('fused_gatconv', ['dgNN/src/fused_gatconv/fused_gatconv.cpp', 'dgNN/src/fused_gatconv/fused_gatconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        CUDAExtension('fused_edgeconv',['dgNN/src/fused_edgeconv/fused_edgeconv.cpp','dgNN/src/fused_edgeconv/fused_edgeconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        CUDAExtension('fused_gmmconv',['dgNN/src/fused_gmmconv/fused_gmmconv.cpp','dgNN/src/fused_gmmconv/fused_gmmconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        CUDAExtension('mhsddmm',['dgNN/src/sddmm/mhsddmm.cc','dgNN/src/sddmm/mhsddmm_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        CUDAExtension('mhtranspose',['dgNN/src/csr2csc/mhtranspose.cc','dgNN/src/csr2csc/mhtranspose_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        CUDAExtension('format_conversion',['dgNN/src/convert/convert.cpp','dgNN/src/convert/convert_kernel.cu'],extra_cflags=["-lcusparse"],extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80']}, extra_link_args=['-lcurand']),
        ],
    
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)