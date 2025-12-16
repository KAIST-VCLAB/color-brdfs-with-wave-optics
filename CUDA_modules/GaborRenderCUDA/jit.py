# Example file for how to load the cuda extension with JIT

from torch.utils.cpp_extension import load
import os
this_folder_path = __file__.split('jit.py')[0]
print(this_folder_path)
periodic_primitives = load(name='gabor_render', 
        sources=[os.path.join(this_folder_path,'GaborRender_cuda.cpp'), 
        os.path.join(this_folder_path,'GaborRender_cuda_kernel.cu')])