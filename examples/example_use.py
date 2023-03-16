
import sys
sys.path.append("/home/john/works/pcl_cosmos/proj/jaxFFTMp")   # Temporary settings in the development environment

from mpi4py import MPI
import jax
import jaxfftmp as jf
import numpy as np
from numpy.testing import assert_allclose

def example_main():
    jax.config.update('jax_platform_name', 'cpu')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank, size)

    nX = 1024
    nY = 1024
    nZ = 1024
    print("-------1-------")
    stream = jf.init_stream()
    print("-------2-------")
    plan_r2c = jf.init_plan(stream, nX, nY, nZ, jf.CUFFT_R2C)
    plan_c2r = jf.init_plan(stream, nX, nY, nZ, jf.CUFFT_C2R)
    print("-------3-------")
    global_shape = np.array([nX, nY, nZ])

    # Initialize an array with the expected gobal size
    array = jax.random.normal(shape=[nX // size,
                                     nY,
                                     nZ],
                              key=jax.random.PRNGKey(rank)).astype('complex64')
    print("-------4-------")
    # Forward FFT, note that the output FFT is transposed
    desc = jf.rfft3d_prim(plan_r2c, global_shape, array, rank, size)
    desc = 123456   # after work, must remark

    # Reverse FFT
    rec_array = jf.irfft3d_prim(plan_c2r, desc)

    assert_allclose(array, rec_array, rtol=1e-10, atol=1e-10)

    '''
    # print("\nJit evaluation:")
    # print("jit(irfft3d_prim) = ", api.jit(jf.irfft3d_prim)(2.0, 10., 10.))

    # print("\nGradient evaluation:")
    # print("grad(square_add_numpy) = ", api.grad(jf.irfft3d_prim)(2.0, 10., 10.))
    '''

    jf.free_desc(desc)

    jf.free_plan(plan_c2r)
    jf.free_plan(plan_r2c)

    jf.free_stream(stream)

if __name__ == '__main__':
    example_main()