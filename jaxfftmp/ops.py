
from functools import partial
from jax import core
from jax._src.lib import xla_client
from jax.interpreters import mlir, xla
from jaxlib.mhlo_helpers import custom_call

from jax._src.lib.mlir.dialects import mhlo

from .util import trace
from . import _cufftMp

from jax import dtypes
from jax.abstract_arrays import ShapedArray

for name, fn in _cufftMp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="cpu")

@trace("rfft3d_prim")
def rfft3d_prim(fft_plan, global_shape, cell_data, mpi_rank, mpi_size):
    return rfft3d_p.bind(fft_plan, global_shape, cell_data, mpi_rank, mpi_size)

def _rfft3d_abstract(fft_plan, global_shape, cell_data, mpi_rank, mpi_size):
    shape = fft_plan.shape
    dtype = dtypes.canonicalize_dtype(fft_plan.dtype)
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))

@trace("_rfft3d_lowering")
def _rfft3d_lowering(ctx, fft_plan, global_shape, cell_data, mpi_rank, mpi_size):
    print("+++++++++call _rfft3d_lowering+++++++++")
    aval_in, _, _, _, _ = ctx.avals_in
    (aval_out, _) = ctx.avals_out

    # The inputs and outputs all have the same shape and memory layout
    dtype = mlir.ir.RankedTensorType(fft_plan.type)
    dims = dtype.shape
    plan_layout = tuple(range(len(dims) - 1, -1, -1))

    global_dtype = mlir.ir.RankedTensorType(global_shape.type)
    dims = global_dtype.shape
    global_layout = tuple(range(len(dims) - 1, -1, -1))

    cell_dtype = mlir.ir.RankedTensorType(cell_data.type)
    dims = cell_dtype.shape
    cell_layout = tuple(range(len(dims) - 1, -1, -1))

    dtype = mlir.ir.RankedTensorType(mpi_rank.type)
    dims = dtype.shape
    rank_layout = tuple(range(len(dims) - 1, -1, -1))

    dtype = mlir.ir.RankedTensorType(mpi_size.type)
    dims = dtype.shape
    size_layout = tuple(range(len(dims) - 1, -1, -1))

    return custom_call(
        'rfft3d_wrapper',
        # Output types
        out_types=[dtype, dtype],
        # The inputs:
        operands=[fft_plan, global_shape, cell_data, mpi_rank, mpi_size],
        # Layout specification:
        operand_layouts=[plan_layout, global_layout, cell_layout, rank_layout, size_layout],
        result_layouts=[plan_layout, plan_layout],
        # GPU specific additional data
        # backend_config=opaque
    )
    # return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results

# Defining new JAX primitives
rfft3d_p = core.Primitive("rfft3d")  # Create the primitive
rfft3d_p.multiple_results = True
rfft3d_p.def_impl(partial(xla.apply_primitive, rfft3d_p))
rfft3d_p.def_abstract_eval(_rfft3d_abstract)

# ad.deflinear2(rfft3d_p, _fft_transpose_rule)
mlir.register_lowering(
        rfft3d_p,
        _rfft3d_lowering,
        platform='cpu')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@trace("irfft3d_prim")
def irfft3d_prim(fft_plan, desc):
    return irfft3d_p.bind(fft_plan, desc)

def _irfft3d_abstract(fft_plan, desc):
    shape = fft_plan.shape
    dtype = dtypes.canonicalize_dtype(fft_plan.dtype)
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))

@trace("_irfft3d_lowering")
def _irfft3d_lowering(ctx, fft_plan, desc):
    print("+++++++++call _irfft3d_lowering+++++++++")
    aval_in, _ = ctx.avals_in
    (aval_out, _) = ctx.avals_out

    # The inputs and outputs all have the same shape and memory layout
    dtype = mlir.ir.RankedTensorType(fft_plan.type)
    dims = dtype.shape
    plan_layout = tuple(range(len(dims) - 1, -1, -1))

    desc_dtype = mlir.ir.RankedTensorType(desc.type)
    dims = desc_dtype.shape
    desc_layout = tuple(range(len(dims) - 1, -1, -1))

    return custom_call(
        'irfft3d_wrapper',
        # Output types
        out_types=[dtype, dtype],
        # The inputs:
        operands=[fft_plan, desc],
        # Layout specification:
        operand_layouts=[plan_layout, desc_layout],
        result_layouts=[plan_layout, plan_layout],
        # GPU specific additional data
        # backend_config=opaque
    )
    # return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results

# Defining new JAX primitives
irfft3d_p = core.Primitive("irfft3d")  # Create the primitive
irfft3d_p.multiple_results = True
irfft3d_p.def_impl(partial(xla.apply_primitive, irfft3d_p))
irfft3d_p.def_abstract_eval(_irfft3d_abstract)

mlir.register_lowering(
        irfft3d_p,
        _irfft3d_lowering,
        platform='cpu')