
from functools import partial, reduce
import jax
from jax import core
from jax._src import abstract_arrays
from jax._src.lib import xla_client
from jax.interpreters import ad, batching, mlir, xla
from jaxlib.mhlo_helpers import custom_call
from jax.interpreters import ad
from jax._src.lib.mlir.dialects import mhlo

from .util import trace
from jax.abstract_arrays import ShapedArray
from . import _cufftMp

from jax import dtypes
from jax.abstract_arrays import ShapedArray

for name, fn in _cufftMp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

@trace("rfft3d_prim")
def rfft3d_prim(cuda_plan, global_shape, cell_data):
# def rfft3d_prim(cuda_plan, global_shape, cell_data, mpi_rank, mpi_size):
    return rfft3d_p.bind(cuda_plan, global_shape, cell_data)

def _rfft3d_abstract(cuda_plan, global_shape, cell_data):
# def _rfft3d_abstract(cuda_plan, global_shape, cell_data, mpi_rank, mpi_size):
    shape = cuda_plan.shape
    dtype = dtypes.canonicalize_dtype(cuda_plan.dtype)
    assert dtypes.canonicalize_dtype(global_shape.dtype) == dtype
    assert global_shape.shape == shape
    # return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))
    ret = cuda_plan.update()
    return ret
@trace("_rfft3d_lowering")
def _rfft3d_lowering(ctx, mean_anom, ecc, cell_data):
    print("+++++++++call _rfft3d_lowering+++++++++")
    # nd3 = jax.random.normal(shape=[5], key=jax.random.PRNGKey(0))
    # ecc = jax.random.normal(shape=[5], key=jax.random.PRNGKey(0))
    # return nd3,  ecc

    # Extract the numpy type of the inputs
    mean_anom_aval, _, _ = ctx.avals_in
    (aval_out,) = ctx.avals_out

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(mean_anom.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))


    # We dispatch a different call depending on the dtype
    op_name = 'rfft3d_wrapper'

    # On the GPU, we do things a little differently and encapsulate the
    # dimension using the 'opaque' parameter
    # opaque = 100

    result = custom_call(
        op_name,
        # Output types
        out_types=[dtype],
        # The inputs:
        operands=[mean_anom, ecc, cell_data],
        # Layout specification:
        operand_layouts=[layout, layout, layout],
        result_layouts=[layout],
        has_side_effect=True,
        # GPU specific additional data
        # backend_config=opaque
    )

    return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results

# Defining new JAX primitives
rfft3d_p = core.Primitive("rfft3d")  # Create the primitive
# rfft3d_p.multiple_results = True
rfft3d_p.def_impl(partial(xla.apply_primitive, rfft3d_p))
rfft3d_p.def_abstract_eval(_rfft3d_abstract)

# ad.deflinear2(rfft3d_p, _fft_transpose_rule)
mlir.register_lowering(
        rfft3d_p,
        _rfft3d_lowering,
        platform='gpu')
