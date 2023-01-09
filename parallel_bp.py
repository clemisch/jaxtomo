import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from util import multi_vmap


print("UPDATED PARALLEL BP")


# redefine jax.lax.map to get unroll support
def map(f, xs, unroll=1):
  g = lambda _, x: ((), f(x))
  _, ys = jax.lax.scan(g, (), xs, unroll=unroll)
  return ys


def interp2d(x, y, xlim, ylim, vals):
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim
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


def get_voxel(proj, theta, x, y, z, uu, vv):
    u = x * jnp.cos(theta) + z * jnp.sin(theta)
    val = interp2d(
        y, u, 
        (vv[0], vv[-1]),
        (uu[0], uu[-1]),
        proj
    ).squeeze()

    # weight with length through voxel
    # val = val * jnp.abs(jnp.cos(theta))

    return val



@partial(jax.jit, static_argnames=("X", "Y"))
def get_bp_once(proj, theta, dU, dV, X, Y, dX):
    dY = dX
    V = proj.shape[0]
    U = proj.shape[1]

    # width in px/vx => one px/vx smaller than physical volume/detector!
    width_img = dX * (X - 1)
    height_img = dY * (Y - 1)
    width_proj = dU * (U - 1)
    height_proj = dV * (V - 1)

    # origins
    O_X = dX * (-0.5 * X + 0.5)
    O_Y = dY * (-0.5 * Y + 0.5)
    O_U = dU * (-0.5 * U + 0.5)
    O_V = dV * (-0.5 * V + 0.5)

    xx = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    yy = jnp.linspace(0., 1., Y, endpoint=True) * height_img + O_Y
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V


    get_voxels = multi_vmap(
        get_voxel,
        (
            (None, None, 0   , None, None, None, None),
            (None, None, None, 1   , None, None, None),
            (None, None, None, None, 2   , None, None),
        ),
        (0, 0, 0)
    )

    vol = get_voxels(
        proj, theta,
        xx[:   , None, None], 
        yy[None, :   , None], 
        xx[None, None, :   ], 
        uu, vv
    )

    return vol



@partial(jax.jit, static_argnames=("X", "Y"))
def get_bp(projs, thetas, dU, dV, X, Y, dX):

    def body_fun(carry, elem):
        proj, theta = elem
        bp = get_bp_once(proj, theta, dU, dV, X, Y, dX)
        carry = carry + bp
        return carry, None

    vol, _ = jax.lax.scan(
        body_fun,
        jnp.zeros((X, Y, X), dtype=projs.dtype),
        (projs, thetas)
    )

    return vol
