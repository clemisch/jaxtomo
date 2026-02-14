import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..util import multi_vmap, interp2d, jaxmap


def _rotate_xy_90(vol):
    return jnp.transpose(vol, (0, 2, 1))


def _get_ray(vol, theta, u, v, xx, yy, zz, s, d):
    # pixel coords
    Dx = d * jnp.cos(theta) - u * jnp.sin(theta)
    Dy = d * jnp.sin(theta) + u * jnp.cos(theta)
    Dz = v

    # source coords
    Sx = -s * jnp.cos(theta)
    Sy = -s * jnp.sin(theta)
    Sz = 0.0

    # ray from source to pixel
    Rx = Dx - Sx
    Ry = Dy - Sy
    Rz = Dz - Sz

    # worker function to get interpolated value of `img_slice` at `x`
    def get_point(x, img_slice):
        dx = x - Sx
        dy = dx / Rx * Ry
        dz = dx / Rx * Rz

        y = Sy + dy
        z = Sz + dz

        val = interp2d(
            z, y,
            (zz[0], zz[-1]), 
            (yy[0], yy[-1]), 
            img_slice
        )

        return val

    # all points along ray, one for each slice
    points = jax.vmap(get_point, (0, 1), 0)(xx, vol)
    ray = jnp.sum(points)

    # weight with length through voxel
    raylen = Rx / jnp.sqrt(Rx**2 + Ry**2 + Rz**2)
    ray = ray / raylen

    return ray


@jax.jit
def _get_fp_angle_oriented(vol, theta, xx, yy, zz, uu, vv, s, d):
    # map over all pixels to get one projection
    get_proj = multi_vmap(
        _get_ray,
        (
            (None, None, None, 0   , None, None, None, None, None), 
            (None, None, 0   , None, None, None, None, None, None)
        ),
        (0, 1)
    )
    proj = get_proj(vol, theta, uu, vv, xx, yy, zz, s, d)
    return proj


@jax.jit
def _get_fp_angles_oriented(vol, thetas, xx, yy, zz, uu, vv, s, d):
    return jaxmap(
        lambda theta: _get_fp_angle_oriented(vol, theta, xx, yy, zz, uu, vv, s, d),
        thetas,
        unroll=1
    )


@jax.jit
def _get_fp_angle_jit(vol0, vol1, theta, xx_base, zz, uu, vv, s, d, princ_dir):
    rot_idx = princ_dir - 1
    theta = theta - rot_idx * jnp.pi / 2

    # Mapping from principal direction to axis orientation.
    xx_sign = jnp.where(rot_idx < 2, 1.0, -1.0).astype(xx_base.dtype)
    yy_sign = jnp.where(
        jnp.logical_or(rot_idx == 0, rot_idx == 3), 1.0, -1.0
    ).astype(xx_base.dtype)

    xx = xx_sign * xx_base
    yy = yy_sign * xx_base
    vol = jax.lax.cond(
        jnp.equal(jnp.bitwise_and(rot_idx, jnp.int32(1)), jnp.int32(1)),
        lambda _: vol1,
        lambda _: vol0,
        operand=None
    )
    return _get_fp_angle_oriented(vol, theta, xx, yy, zz, uu, vv, s, d)




def _get_princ_dir(theta):
    theta = theta + jnp.pi / 4
    theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
    princ_dir = jnp.floor_divide(theta, jnp.pi / 2) + 1
    return princ_dir.astype("int32")

_get_princ_dirs = jax.vmap(_get_princ_dir)


def _get_princ_dirs_np(thetas):
    theta = thetas + np.pi / 4
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    princ_dir = np.floor_divide(theta, np.pi / 2) + 1
    return princ_dir.astype(np.int32)


def _build_axes(vol, dX, U, dU, V, dV):
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

    xx_base = jnp.linspace(0., 1., X, endpoint=True) * width_img + O_X
    zz = jnp.linspace(0., 1., Z, endpoint=True) * height_img + O_Z
    uu = jnp.linspace(0., 1., U, endpoint=True) * width_proj + O_U
    vv = jnp.linspace(0., 1., V, endpoint=True) * height_proj + O_V
    return xx_base, zz, uu, vv


@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_jit(vol, thetas, dX, U, dU, V, dV, s, d):
    xx_base, zz, uu, vv = _build_axes(vol, dX, U, dU, V, dV)

    # For runtime-dispatched JIT path, only two rotated volumes are distinct.
    vol0 = vol
    vol1 = _rotate_xy_90(vol0)
    princ_dirs = _get_princ_dirs(thetas)

    # map over angles to get full FP
    def mapfun(args):
        theta, princ_dir = args
        return _get_fp_angle_jit(
            vol0, vol1,
            theta, xx_base, zz, uu, vv, s, d, princ_dir
        )

    projs = jaxmap(mapfun, (thetas, princ_dirs), unroll=1)
    return projs


def _is_tracer(x):
    return isinstance(x, jax.core.Tracer)


def get_fp(vol, thetas, dX, U, dU, V, dV, s, d):
    if _is_tracer(vol) or _is_tracer(thetas):
        return _get_fp_jit(vol, thetas, dX, U, dU, V, dV, s, d)

    xx_base, zz, uu, vv = _build_axes(vol, dX, U, dU, V, dV)

    thetas_np = np.asarray(thetas)
    princ_dirs = _get_princ_dirs_np(thetas_np)
    nangles = thetas_np.shape[0]

    if nangles == 0:
        return jnp.zeros((0, V, U), dtype=vol.dtype)

    # Only two volume orientations are needed with this rotation scheme.
    need_transposed_vol = np.any(np.logical_or(princ_dirs == 2, princ_dirs == 4))
    vol_t = _rotate_xy_90(vol) if need_transposed_vol else None

    proj_chunks = []
    idx_chunks = []
    dir_specs = (
        # (princ_dir, x_sign, y_sign, theta_offset, use_transposed_vol)
        (1,  1.0,  1.0, 0.0, False),
        (2,  1.0, -1.0, np.pi / 2, True),
        (3, -1.0, -1.0, np.pi, False),
        (4, -1.0,  1.0, 3 * np.pi / 2, True),
    )

    for princ_dir, x_sign, y_sign, theta_offset, use_tvol in dir_specs:
        idx_np = np.where(princ_dirs == princ_dir)[0]
        if idx_np.size == 0:
            continue

        idx = jnp.asarray(idx_np, dtype=jnp.int32)
        vol_sel = vol_t if use_tvol else vol
        thetas_sel = thetas[idx] - theta_offset
        xx = x_sign * xx_base
        yy = y_sign * xx_base

        projs = _get_fp_angles_oriented(vol_sel, thetas_sel, xx, yy, zz, uu, vv, s, d)
        proj_chunks.append(projs)
        idx_chunks.append(idx_np)

    if len(proj_chunks) == 1:
        return proj_chunks[0]

    proj_cat = jnp.concatenate(proj_chunks, axis=0)
    idx_cat = np.concatenate(idx_chunks)
    inv_idx = jnp.asarray(np.argsort(idx_cat), dtype=jnp.int32)
    return proj_cat[inv_idx]


# TODO: change to "static_argnames" once JAX supports it
@partial(
    jax.pmap, 
    in_axes=(None, 0, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(3, 5)
)
def _get_fp_pmap(vol, thetas, dX, U, dU, V, dV, s, d):
    proj = _get_fp_jit(vol, thetas, dX, U, dU, V, dV, s, d)
    return proj


@partial(jax.jit, static_argnames=("U", "V"))
def get_fp_pmap(vol, thetas, dX, U, dU, V, dV, s, d):
    nangles = thetas.shape[0]
    ndevices = jax.device_count()
    assert nangles % ndevices == 0

    thetas = thetas.reshape(ndevices, -1)
    proj = _get_fp_pmap(vol, thetas, dX, U, dU, V, dV, s, d)
    proj = proj.reshape(nangles, V, U)

    return proj
