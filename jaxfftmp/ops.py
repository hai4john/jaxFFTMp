
from jax import core
from jax._src import abstract_arrays
from jax._src.lib import xla_client
from jax.interpreters import xla

from .util import trace
from . import _cufftMp

for name, fn in _cufftMp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")



# Defining new JAX primitives
rfft3d_p = core.Primitive("rfft3d")  # Create the primitive
@trace("rfft3d_prim")
def rfft3d_prim(x, y, z):
    return rfft3d_p.bind(x, y, z)

import numpy as np
@trace("rfft3d_impl")
def rfft3d_impl(x, y, z):
  # Note that we can use the original numpy, which is not JAX traceable
  return np.add(np.multiply(x, y), z)

# Now we register the primal implementation with JAX
rfft3d_p.def_impl(rfft3d_impl)

@trace("rfft3d_abstract_eval")
def rfft3d_abstract_eval(xs, ys, zs):
  assert xs.shape == ys.shape
  assert xs.shape == zs.shape
  return abstract_arrays.ShapedArray(xs.shape, xs.dtype)

# Now we register the abstract evaluation with JAX
rfft3d_p.def_abstract_eval(rfft3d_abstract_eval)

@trace("rfft3d_xla_translation")
def rfft3d_xla_translation(ctx, avals_in, avals_out, xc, yc, zc):
  return [xla_client.ops.Add(xla_client.ops.Mul(xc, yc), zc)]

xla.register_translation(rfft3d_p, rfft3d_xla_translation, platform='gpu')




irfft3d_p = core.Primitive("irfft3d")  # Create the primitive
@trace("irfft3d_prim")
def irfft3d_prim(x, y, z):
    return irfft3d_p.bind(x, y, z)

@trace("irfft3d_impl")
def irfft3d_impl(x, y, z):
  # Note that we can use the original numpy, which is not JAX traceable
  return np.add(np.multiply(x, y), z)
# Now we register the primal implementation with JAX
irfft3d_p.def_impl(irfft3d_impl)

@trace("irfft3d_abstract_eval")
def irfft3d_abstract_eval(xs, ys, zs):
  assert xs.shape == ys.shape
  assert xs.shape == zs.shape
  return abstract_arrays.ShapedArray(xs.shape, xs.dtype)

# Now we register the abstract evaluation with JAX
irfft3d_p.def_abstract_eval(irfft3d_abstract_eval)

@trace("irfft3d_xla_translation")
def irfft3d_xla_translation(ctx, avals_in, avals_out, xc, yc, zc):
  return [xla_client.ops.Add(xla_client.ops.Mul(xc, yc), zc)]

xla.register_translation(irfft3d_p, irfft3d_xla_translation, platform='gpu')