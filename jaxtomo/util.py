import os

import jax
import jax.scipy as jsp



def set_cuda_device(n):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n)


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

    vals_interp = jsp.ndimage.map_coordinates(
        vals, 
        (x, y), 
        order=1,
        mode="constant", 
        cval=0.0,
    )
    return vals_interp


def jaxmap(f, xs, unroll=1):
    """ Redefine jax.lax.map to get unroll support """
    g = lambda _, x: ((), f(x))
    _, ys = jax.lax.scan(g, (), xs, unroll=unroll)
    return ys
