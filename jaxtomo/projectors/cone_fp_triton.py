import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from functools import partial

from ..util import jaxmap
from .cone_fp import _get_princ_dirs


@triton.jit
def _fp_angle_kernel(
    vol,
    theta,
    dX,
    dU,
    dV,
    S,
    D,
    princ_dir,
    proj,
    X: tl.constexpr,
    Z: tl.constexpr,
    U: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    theta = tl.load(theta)
    dX = tl.load(dX)
    dU = tl.load(dU)
    dV = tl.load(dV)
    S = tl.load(S)
    D = tl.load(D)
    princ_dir = tl.load(princ_dir)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (U * V)

    iu = offs % U
    iv = offs // U

    half = 0.5
    pi = 3.141592653589793

    o_x = dX * (-half * X + half)
    o_z = dX * (-half * Z + half)
    o_u = dU * (-half * U + half)
    o_v = dV * (-half * V + half)

    u = o_u + iu.to(tl.float32) * dU
    v = o_v + iv.to(tl.float32) * dV

    nrots = princ_dir - 1
    theta_rot = theta - nrots.to(tl.float32) * (pi * half)
    ct = tl.cos(theta_rot)
    st = tl.sin(theta_rot)

    det_x = D * ct - u * st
    det_y = D * st + u * ct
    det_z = v

    src_x = -S * ct
    src_y = -S * st

    ray_x = det_x - src_x
    ray_y = det_y - src_y
    ray_z = det_z

    odd_rot = (nrots % 2) == 1
    desc_x = nrots >= 2
    desc_y = (nrots == 1) | (nrots == 2)

    y_lo = tl.where(desc_y, -o_x, o_x)
    y_hi = tl.where(desc_y, -o_x - (X - 1) * dX, o_x + (X - 1) * dX)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for ix in tl.range(0, X):
        coord = o_x + ix * dX
        x = tl.where(desc_x, -coord, coord)

        frac = (x - src_x) / ray_x
        y = src_y + frac * ray_y
        z = frac * ray_z

        y_pix = (y - y_lo) * (X - 1) / (y_hi - y_lo)
        z_pix = (z - o_z) / dX

        y0 = tl.floor(y_pix).to(tl.int32)
        z0 = tl.floor(z_pix).to(tl.int32)
        y1 = y0 + 1
        z1 = z0 + 1

        wy = y_pix - y0.to(tl.float32)
        wz = z_pix - z0.to(tl.float32)

        m00 = mask & (y0 >= 0) & (y0 < X) & (z0 >= 0) & (z0 < Z)
        m01 = mask & (y1 >= 0) & (y1 < X) & (z0 >= 0) & (z0 < Z)
        m10 = mask & (y0 >= 0) & (y0 < X) & (z1 >= 0) & (z1 < Z)
        m11 = mask & (y1 >= 0) & (y1 < X) & (z1 >= 0) & (z1 < Z)

        off00_even = (z0 * X + ix) * X + y0
        off01_even = (z0 * X + ix) * X + y1
        off10_even = (z1 * X + ix) * X + y0
        off11_even = (z1 * X + ix) * X + y1

        off00_odd = (z0 * X + y0) * X + ix
        off01_odd = (z0 * X + y1) * X + ix
        off10_odd = (z1 * X + y0) * X + ix
        off11_odd = (z1 * X + y1) * X + ix

        off00 = tl.where(odd_rot, off00_odd, off00_even)
        off01 = tl.where(odd_rot, off01_odd, off01_even)
        off10 = tl.where(odd_rot, off10_odd, off10_even)
        off11 = tl.where(odd_rot, off11_odd, off11_even)

        v00 = tl.load(vol + off00, mask=m00, other=0.0)
        v01 = tl.load(vol + off01, mask=m01, other=0.0)
        v10 = tl.load(vol + off10, mask=m10, other=0.0)
        v11 = tl.load(vol + off11, mask=m11, other=0.0)

        val0 = v00 * (1.0 - wy) + v01 * wy
        val1 = v10 * (1.0 - wy) + v11 * wy
        acc += val0 * (1.0 - wz) + val1 * wz

    raylen = ray_x / tl.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    acc = acc / raylen

    tl.store(proj + offs, acc, mask=mask)


@partial(jax.jit, static_argnames=("U", "V"))
def _get_fp_angle(vol, theta, dX, U, dU, V, dV, s, d, princ_dir):
    out_shape = jax.ShapeDtypeStruct((V, U), vol.dtype)
    block = 256
    grid = (triton.cdiv(U * V, block),)

    return jt.triton_call(
        vol,
        theta,
        dX,
        dU,
        dV,
        s,
        d,
        princ_dir,
        kernel=_fp_angle_kernel,
        out_shape=out_shape,
        grid=grid,
        X=vol.shape[1],
        Z=vol.shape[0],
        U=U,
        V=V,
        BLOCK=block,
        num_warps=8,
    )


@partial(jax.jit, static_argnames=("U", "V"))
def get_fp(vol, thetas, dX, U, dU, V, dV, s, d):
    princ_dirs = _get_princ_dirs(thetas)

    def mapfun(args):
        theta, princ_dir = args
        return _get_fp_angle(vol, theta, dX, U, dU, V, dV, s, d, princ_dir)

    return jaxmap(mapfun, (thetas, princ_dirs), unroll=1)
