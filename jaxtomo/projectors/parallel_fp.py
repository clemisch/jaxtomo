import jax
import jax.numpy as jnp
from functools import partial

from ..util import multi_vmap, interp2d, jaxmap


def _get_ray_2d(vol, theta, u, v, xx, yy):
    def get_point(z, img_slice):
        x = 1. / jnp.cos(theta) * (u - z * jnp.sin(theta))
        val = interp2d(
            v, x,
            (yy[0], yy[-1]),
            (xx[0], xx[-1]),
            img_slice
        )
        return val

    points = jax.vmap(get_point, (0, 1), 0)(xx, vol)
    ray = jnp.sum(points)

    # weight with length through voxel
    ray = ray / jnp.cos(theta)

    return ray



@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_angle(vol, theta, dX, U, dU, V, dV):
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

    xx = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    yy = jnp.linspace(0., 1., Y, endpoint=True) * height_img + O_Y
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V

    get_proj = multi_vmap(
        _get_ray_2d,
        (   
            (None, None, 0, None, None, None), 
            (None, None, None, 0, None, None)
        ),
        (0, 0)
    )
    proj = get_proj(vol, theta, uu, vv, xx, yy)

    return proj


@partial(jax.jit, static_argnames=("U", "V"))
def get_fp(vol, thetas, dX, U, dU, V, dV):

    # map over angles to get full FP
    projs = jaxmap(
        lambda theta: _get_fp_angle(vol, theta, dX, U, dU, V, dV),
        thetas,
        unroll=1
    )

    return projs

    
# TODO: change to "static_argnames" once JAX supports it
@partial(
    jax.pmap, 
    in_axes=(None, 0, None, None, None, None, None),
    static_broadcasted_argnums=(3, 5)
)
def _get_fp_pmap(vol, thetas, dX, U, dU, V, dV):
    proj = get_fp(vol, thetas, dX, U, dU, V, dV)
    return proj


def get_fp_pmap(vol, thetas, dX, U, dU, V, dV):
    nangles = thetas.shape[0]
    ndevices = jax.device_count()
    assert nangles % ndevices == 0

    thetas = thetas.reshape(ndevices, -1)
    proj = _get_fp_pmap(vol, thetas, dX, U, dU, V, dV)
    proj = proj.reshape(nangles, V, U)

    return proj
