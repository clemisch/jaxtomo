import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from .util import multi_vmap, interp2d


def _get_voxel(proj, theta, x, y, z, uu, vv, S, D):
    x_ =  x * jnp.cos(theta) + y * jnp.sin(theta)
    y_ = -x * jnp.sin(theta) + y * jnp.cos(theta)

    frac_ray = (D + S) / (x_ + S)
    u = y_ * frac_ray
    v = z * frac_ray

    val = interp2d(
        v, u, 
        (vv[0], vv[-1]),
        (uu[0], uu[-1]),
        proj
    )

    return val



@jax.jit
def _get_slice(projs, thetas, uu, v, xx, yy, zz, S, D):

    get_voxels = multi_vmap(
        _get_voxel,
        (
            (None, None, None, 0   , None, None, None, None, None),
            (None, None, 0   , None, None, None, None, None, None),
            (None, None, None, None, 0   , None, None, None, None),
        ),
        (0, 0, 0)
    )
    vol = get_voxels(proj, theta, xx, xx, zz, uu, vv, S, D)

    return vol



@partial(jax.jit, static_argnames=("X", "Z"))
def get_bp(projs, thetas, dU, dV, X, Z, dX, S, D):
     # cubic voxels
    dZ = dX
    V = proj.shape[0]
    U = proj.shape[1]

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

    xx = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    zz = jnp.linspace(0., 1., Z, endpoint=True) * height_img + O_Z
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V


    def body_fun(carry, elem):
        proj, theta = elem
        bp = _get_bp_angle(proj, theta, dU, dV, X, Z, dX, S, D)
        carry = carry + bp
        return carry, None

    vol, _ = jax.lax.scan(
        body_fun,
        jnp.zeros((Z, X, X), dtype=projs.dtype),
        (projs, thetas)
    )

    return vol




# TODO: change to "static_argnames" once JAX supports it
@partial(
    jax.pmap, 
    in_axes=(0, 0, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(4, 5)
)
def _get_bp_pmap(projs, thetas, dU, dV, X, Z, dX, S, D):
    vol = get_bp(projs, thetas, dU, dV, X, Z, dX, S, D)
    return vol


# JAX complains about jit of pmap, but it _is_ faster than without jit in this
# case
@partial(jax.jit, static_argnames=("X", "Z"))
def get_bp_pmap(projs, thetas, dU, dV, X, Z, dX, S, D):
    nangles = thetas.shape[0]
    ndevices = jax.device_count()
    assert nangles % ndevices == 0

    projs = projs.reshape(ndevices, -1, *projs.shape[1:])
    thetas = thetas.reshape(ndevices, -1)

    vol = _get_bp_pmap(projs, thetas, dU, dV, X, Z, dX, S, D)
    vol = vol.sum(0)

    return vol
