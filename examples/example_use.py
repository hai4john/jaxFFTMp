
import sys
sys.path.append("/home/john/works/pcl_cosmos/proj/jaxFTTMp")   # Temporary settings in the development environment

import jaxfftmp as jf

from jax._src import api

typeExample = jf.CUFFT_R2C
stream = jf.init_stream()
print(typeExample)
print(stream)
jf.free_stream(stream)

print("\nNormal evaluation:")
print("rfft3d = ", jf.irfft3d_prim(2., 10., 10.))

print("\nJit evaluation:")
print("jit(irfft3d_prim) = ", api.jit(jf.irfft3d_prim)(2.0, 10., 10.))



# print("\nGradient evaluation:")
# print("grad(square_add_numpy) = ", api.grad(jf.irfft3d_prim)(2.0, 10., 10.))