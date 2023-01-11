import jax
import jax.numpy as jnp
from functools import partial

from .util import multi_vmap, interp2d, jaxmap


def _solve_2d(A, b):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    A_inv = jnp.array(
        (A[1, 1], -A[0, 1], -A[1, 0], A[0, 0]), 
        dtype=A.dtype
    ).reshape(2, 2) / det
    x = A_inv @ b

    return x



def _get_ray_2d(vol, theta, u, v, xx, yy, zz, S, D):
    u = jnp.squeeze(u)
    v = jnp.squeeze(v)

    def get_point(z, img_slice):
        A = jnp.array(
            (u / (D - S), -1., jnp.cos(theta), jnp.sin(theta)),
            dtype=vol.dtype
        ).reshape(2, 2)
        b = jnp.array((u * S / (D - S), z), dtype=vol.dtype)

        z_, x_ = _solve_2d(A, b)
        x = x_ * jnp.cos(theta) - z_ * jnp.sin(theta)

        val = interp2d(
            v, x,
            (yy[0], yy[-1]), 
            (xx[0], xx[-1]), 
            img_slice
        )
        return val

    # content of `zz` depends on principal direction
    points = jax.vmap(get_point, (0, 1), 0)(zz, vol)
    ray = jnp.sum(points)

    # weight with length through voxel
    beta = jnp.arctan2(u, D - S)
    ray = ray / jnp.cos(theta - beta)

    return ray



# @partial(jax.jit, static_argnames=("U", "V", "princ_dir"))
@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_angle(vol, theta, dX, U, dU, V, dV, S, D, princ_dir):
    dY = dX
    Y = vol.shape[0]
    X = vol.shape[1]

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

    # axes for volume and projector
    xx = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    yy = jnp.linspace(0., 1., Y, endpoint=True) * height_img + O_Y
    zz = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V

    # handle principal direction
    nrots = princ_dir - 1
    theta = theta - nrots * jnp.pi / 2

    # TODO: avoid for loop
    def body_fun(i, arg):
        xx, zz, vol = arg
        # TODO: change vmap axis of _get_ray_2d instead of transposing
        vol = jnp.transpose(vol, (0, 2, 1))
        xx, zz = zz, xx
        zz = -zz
        return xx, zz, vol

    xx, zz, vol = jax.lax.fori_loop(
        0, nrots,
        body_fun,
        (xx, zz, vol)
    )



    # map over all pixels to get one projection
    get_proj = multi_vmap(
        _get_ray_2d,
        (
            (None, None, 0, None, None, None, None, None, None), 
            (None, None, None, 0, None, None, None, None, None)
        ),
        (0, 0)
    )
    proj = get_proj(vol, theta, uu, vv, xx, yy, zz, S, D)

    return proj




def _get_princ_dir(theta):
    theta = theta + jnp.pi / 4
    theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
    princ_dir = jnp.floor_divide(theta, jnp.pi / 2) + 1
    return princ_dir.astype("int32")

_get_princ_dirs = jax.vmap(_get_princ_dir)



@partial(jax.jit, static_argnames=("U", "V"))
def get_fp(vol, thetas, dX, U, dU, V, dV, S, D):
    
    princ_dirs = _get_princ_dirs(thetas)

    # map over angles to get full FP
    def mapfun(args):
        theta, princ_dir = args
        return _get_fp_angle(vol, theta, dX, U, dU, V, dV, S, D, princ_dir)

    projs = jaxmap(mapfun, (thetas, princ_dirs), unroll=1)

    return projs
