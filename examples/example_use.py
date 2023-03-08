
import sys
sys.path.append("/home/john/works/pcl_cosmos/proj/jaxFTTMp")   # Temporary settings in the development environment

import jaxfftmp as jf
import numpy as np
import jax.numpy as jnp
from jax import dtypes
from jax.abstract_arrays import ShapedArray

from jax._src import api

typeExample = jf.CUFFT_R2C
stream = jf.init_stream()
print(typeExample)
print(stream)
jf.free_stream(stream)

print("\nNormal evaluation:")
plan = 123456
global_shape = 256 #np.array([256,] *3)
cell_data = 100
mpi_rank = 1
mpi_size = 8

nd3 =np.random.random(5)
ecc = np.random.randn(5)
# dtype = dtypes.canonicalize_dtype(cell_data.dtype)
# shapedarr = ShapedArray(cell_data.shape, dtype)
jf.rfft3d_prim(nd3, ecc, ecc)
# print("rfft3d = ", jf.rfft3d_prim(plan, global_shape, cell_data, mpi_rank, mpi_size))

# print("\nJit evaluation:")
# print("jit(irfft3d_prim) = ", api.jit(jf.irfft3d_prim)(2.0, 10., 10.))



# print("\nGradient evaluation:")
# print("grad(square_add_numpy) = ", api.grad(jf.irfft3d_prim)(2.0, 10., 10.))