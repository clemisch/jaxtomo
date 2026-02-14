import os

import numpy as np
import jax
import jax.numpy as jnp



def set_cuda_device(*args, verbose=True):
    assert all([isinstance(a, int) for a in args])

    devices = ",".join(map(str, args))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    if verbose:
        print("CUDA_DEVICE_ORDER=PCI_BUS_ID")
        print(f"CUDA_VISIBLE_DEVICES={devices}")


def set_platform(platform):
    assert platform in {"cpu", "gpu"}
    jax.config.update("jax_platform_name", platform)
    if platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""



def set_preallocation(is_prealloc):
    if not is_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str("0.90")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"


def multi_vmap(fun, in_axes, out_axes):
    """ vmap over several axes """
    batched_fun = fun
    for inax, outax in zip(in_axes, out_axes):
        batched_fun = jax.vmap(batched_fun, inax, outax)
    return batched_fun


def interp2d(x, y, xlims, ylims, vals):
    x_lo, x_hi = xlims
    y_lo, y_hi = ylims
    n_x, n_y = vals.shape

    # transform x,y into pixel values
    x = (x - x_lo) * (n_x - 1.) / (x_hi - x_lo)
    y = (y - y_lo) * (n_y - 1.) / (y_hi - y_lo)

    x, y = jnp.broadcast_arrays(x, y)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0.astype(x.dtype)
    wy = y - y0.astype(y.dtype)
    one = jnp.array(1.0, dtype=wx.dtype)

    x0c = jnp.clip(x0, 0, n_x - 1)
    x1c = jnp.clip(x1, 0, n_x - 1)
    y0c = jnp.clip(y0, 0, n_y - 1)
    y1c = jnp.clip(y1, 0, n_y - 1)

    v00 = vals[x0c, y0c]
    v01 = vals[x0c, y1c]
    v10 = vals[x1c, y0c]
    v11 = vals[x1c, y1c]

    m00 = (x0 >= 0) & (x0 < n_x) & (y0 >= 0) & (y0 < n_y)
    m01 = (x0 >= 0) & (x0 < n_x) & (y1 >= 0) & (y1 < n_y)
    m10 = (x1 >= 0) & (x1 < n_x) & (y0 >= 0) & (y0 < n_y)
    m11 = (x1 >= 0) & (x1 < n_x) & (y1 >= 0) & (y1 < n_y)

    v00 = jnp.where(m00, v00, 0.0)
    v01 = jnp.where(m01, v01, 0.0)
    v10 = jnp.where(m10, v10, 0.0)
    v11 = jnp.where(m11, v11, 0.0)

    w00 = (one - wx) * (one - wy)
    w01 = (one - wx) * wy
    w10 = wx * (one - wy)
    w11 = wx * wy

    vals_interp = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return vals_interp


def jaxmap(f, xs, unroll=1):
    """ Redefine jax.lax.map to get unroll support """
    g = lambda _, x: ((), f(x))
    _, ys = jax.lax.scan(g, (), xs, unroll=unroll)
    return ys


def roundmask(ny, nx):
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, ny, endpoint=True),
        np.linspace(-1, 1, nx, endpoint=True),
        indexing="ij",
        sparse=True
    )
    r2 = yy**2 + xx**2
    mask = np.where(r2 > 1., 0., 1.)

    return mask
