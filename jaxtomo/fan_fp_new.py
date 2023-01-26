import jax
import jax.numpy as jnp
from functools import partial

from .util import multi_vmap, interp2d, jaxmap


def _get_ray(vol, theta, u, v, xx, yy, zz, s, d):
    # pixel coords
    Dx = d * jnp.cos(theta) - u * jnp.sin(theta)
    Dy = d * jnp.sin(theta) + u * jnp.cos(theta)

    # source coords
    Sx = -s * jnp.cos(theta)
    Sy = -s * jnp.sin(theta)

    # ray from source to pixel
    Rx = Dx - Sx
    Ry = Dy - Sy

    def get_point(x, img_slice):
        dx = x - Sx
        dy = dx / Rx * Ry
        y = Sy + dy

        val = interp2d(
            v, y,
            (zz[0], zz[-1]), 
            (yy[0], yy[-1]), 
            img_slice
        )
        return val

    points = jax.vmap(get_point, (0, 1), 0)(xx, vol)
    ray = jnp.sum(points)

    # weight with length through voxel
    angle_tot = jnp.arctan2(Ry, Rx)
    ray = ray / jnp.cos(angle_tot)

    return ray





@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_angle(vol, theta, dX, U, dU, V, dV, s, d, princ_dir):
    dZ = dX  # cubic voxels
    Z = vol.shape[0]
    X = vol.shape[1]

    # width in px/vx => one px/vx smaller than physical volume/detector!
    width_img = dX * (X - 1)
    height_img = dZ * (Z - 1)
    width_proj = dU * (U - 1)
    height_proj = dV * (V - 1)

    # origins
    O_X = dX * (-0.5 * X + 0.5)
    O_Z = dZ * (-0.5 * Z + 0.5)
    O_U = dU * (-0.5 * U + 0.5)
    O_V = dV * (-0.5 * V + 0.5)

    # axes for volume and projector
    xx = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    yy = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    zz = jnp.linspace(0., 1., Z, endpoint=True) * height_img + O_Z
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V

    # handle principal direction
    nrots = princ_dir - 1
    theta = theta - nrots * jnp.pi / 2

    def body_fun(i, arg):
        xx, yy, vol = arg
        vol = jnp.transpose(vol, (0, 2, 1))
        xx, yy = yy, -xx
        return xx, yy, vol

    xx, yy, vol = jax.lax.fori_loop(
        0, nrots,
        body_fun,
        (xx, yy, vol)
    )

    # map over all pixels to get one projection
    get_proj = multi_vmap(
        _get_ray,
        (
            (None, None, 0, None, None, None, None, None, None), 
            (None, None, None, 0, None, None, None, None, None)
        ),
        (0, 0)
    )
    proj = get_proj(vol, theta, uu, vv, xx, yy, zz, s, d)

    return proj




def _get_princ_dir(theta):
    theta = theta + jnp.pi / 4
    theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
    princ_dir = jnp.floor_divide(theta, jnp.pi / 2) + 1
    return princ_dir.astype("int32")

_get_princ_dirs = jax.vmap(_get_princ_dir)



@partial(jax.jit, static_argnames=("U", "V"))
def get_fp(vol, thetas, dX, U, dU, V, dV, s, d):
    
    princ_dirs = _get_princ_dirs(thetas)

    # map over angles to get full FP
    def mapfun(args):
        theta, princ_dir = args
        return _get_fp_angle(vol, theta, dX, U, dU, V, dV, s, d, princ_dir)

    projs = jaxmap(mapfun, (thetas, princ_dirs), unroll=1)

    return projs
