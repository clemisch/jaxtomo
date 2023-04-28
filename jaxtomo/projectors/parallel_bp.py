import jax
import jax.numpy as jnp
from functools import partial

from ..util import multi_vmap, interp2d, jaxmap


def _get_voxel(proj, theta, x, y, z, uu, vv):
    u = x * jnp.cos(theta) + z * jnp.sin(theta)
    val = interp2d(
        y, u, 
        (vv[0], vv[-1]),
        (uu[0], uu[-1]),
        proj
    ).squeeze()

    return val



@partial(jax.jit, static_argnames=("X", "Y"))
def _get_bp_angle(proj, theta, dU, dV, X, Y, dX):
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
        _get_voxel,
        (
            (None, None, 0   , None, None, None, None),
            (None, None, None, None, 0   , None, None),
            (None, None, None, 0   , None, None, None),
        ),
        (0, 0, 0)
    )
    vol = get_voxels(proj, theta, xx, yy, xx, uu, vv)

    return vol



@partial(jax.jit, static_argnames=("X", "Y"))
def get_bp(projs, thetas, dU, dV, X, Y, dX):

    def body_fun(carry, elem):
        proj, theta = elem
        bp = _get_bp_angle(proj, theta, dU, dV, X, Y, dX)
        carry = carry + bp
        return carry, None

    vol, _ = jax.lax.scan(
        body_fun,
        jnp.zeros((Y, X, X), dtype=projs.dtype),
        (projs, thetas)
    )

    return vol




# TODO: change to "static_argnames" once JAX supports it
@partial(
    jax.pmap, 
    in_axes=(0, 0, None, None, None, None, None),
    static_broadcasted_argnums=(4, 5)
)
def _get_bp_pmap(projs, thetas, dU, dV, X, Z, dX):
    vol = get_bp(projs, thetas, dU, dV, X, Z, dX)
    return vol


# JAX complains about jit of pmap, but it _is_ faster than without jit in this
# case
@partial(jax.jit, static_argnames=("X", "Z"))
def get_bp_pmap(projs, thetas, dU, dV, X, Z, dX):
    nangles = thetas.shape[0]
    ndevices = jax.device_count()
    assert nangles % ndevices == 0

    projs = projs.reshape(ndevices, -1, *projs.shape[1:])
    thetas = thetas.reshape(ndevices, -1)

    vol = _get_bp_pmap(projs, thetas, dU, dV, X, Z, dX)
    vol = vol.sum(0)

    return vol
