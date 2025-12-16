from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GaborRender',
    ext_modules=[
        CUDAExtension('GaborRender', [
            'GaborRender_cuda.cpp',
            'GaborRender_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })