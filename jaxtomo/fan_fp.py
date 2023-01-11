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


def _get_ray_2d(vol, theta, u, v, xx, yy, S, D):
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
            (yy.min(), yy.max()), 
            (xx.min(), xx.max()), 
            img_slice
        )
        return val

    points = jax.vmap(get_point, (0, 1), 0)(xx, vol)
    ray = jnp.sum(points)

    # weight with length through voxel
    beta = jnp.arctan2(u, D - S)
    ray = ray / jnp.cos(theta - beta)

    return ray



@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_angle(vol, theta, dX, U, dU, V, dV, S, D):
    def cond_fun(arg):
        angle, _ = arg
        is_valid_angle = jnp.logical_and(
            (-jnp.pi / 4) < angle, 
            angle <= (jnp.pi / 4)
        )
        return jnp.logical_not(is_valid_angle)

    def body_fun(arg):
        angle, im = arg
        im = im.transpose((0, 2, 1))
        im = im[:, ::-1]
        angle = angle - jnp.pi / 2
        return angle, im

    theta, vol = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (theta, vol)
    )

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
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V

    # map over all pixels to get one projection
    get_proj = multi_vmap(
        _get_ray_2d,
        (
            (None, None, 0, None, None, None, None, None), 
            (None, None, None, 0, None, None, None, None)
        ),
        (0, 0)
    )
    proj = get_proj(vol, theta, uu, vv, xx, yy, S, D)

    return proj


@partial(jax.jit, static_argnames=("U", "V"))
def get_fp(vol, thetas, dX, U, dU, V, dV, S, D):
    
    # map over angles to get full FP
    projs = jaxmap(
        lambda theta: _get_fp_angle(vol, theta, dX, U, dU, V, dV, S, D),
        thetas,
        unroll=1
    )

    return projs
