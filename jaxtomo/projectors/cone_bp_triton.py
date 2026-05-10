import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from functools import partial


@triton.jit
def _bp_angle_kernel(
    proj,
    theta,
    dU,
    dV,
    dX,
    S,
    D,
    vol,
    X: tl.constexpr,
    Z: tl.constexpr,
    U: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    theta = tl.load(theta)
    dU = tl.load(dU)
    dV = tl.load(dV)
    dX = tl.load(dX)
    S = tl.load(S)
    D = tl.load(D)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (Z * X * X)

    iy = offs % X
    ix = (offs // X) % X
    iz = offs // (X * X)

    half = 0.5
    o_x = dX * (-half * X + half)
    o_z = dX * (-half * Z + half)
    o_u = dU * (-half * U + half)
    o_v = dV * (-half * V + half)

    x = o_x + ix.to(tl.float32) * dX
    y = o_x + iy.to(tl.float32) * dX
    z = o_z + iz.to(tl.float32) * dX

    ct = tl.cos(theta)
    st = tl.sin(theta)

    x_rot = x * ct + y * st
    y_rot = -x * st + y * ct

    frac_ray = (D + S) / (x_rot + S)
    u = y_rot * frac_ray
    v = z * frac_ray

    u_pix = (u - o_u) / dU
    v_pix = (v - o_v) / dV

    u0 = tl.floor(u_pix).to(tl.int32)
    v0 = tl.floor(v_pix).to(tl.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    wu = u_pix - u0.to(tl.float32)
    wv = v_pix - v0.to(tl.float32)

    m00 = mask & (u0 >= 0) & (u0 < U) & (v0 >= 0) & (v0 < V)
    m01 = mask & (u1 >= 0) & (u1 < U) & (v0 >= 0) & (v0 < V)
    m10 = mask & (u0 >= 0) & (u0 < U) & (v1 >= 0) & (v1 < V)
    m11 = mask & (u1 >= 0) & (u1 < U) & (v1 >= 0) & (v1 < V)

    p00 = tl.load(proj + v0 * U + u0, mask=m00, other=0.0)
    p01 = tl.load(proj + v0 * U + u1, mask=m01, other=0.0)
    p10 = tl.load(proj + v1 * U + u0, mask=m10, other=0.0)
    p11 = tl.load(proj + v1 * U + u1, mask=m11, other=0.0)

    val0 = p00 * (1.0 - wu) + p01 * wu
    val1 = p10 * (1.0 - wu) + p11 * wu
    val = val0 * (1.0 - wv) + val1 * wv

    tl.store(vol + offs, val, mask=mask)


@triton.jit
def _bp_fused_kernel(
    projs,
    thetas,
    dU,
    dV,
    dX,
    S,
    D,
    vol,
    NANGLES: tl.constexpr,
    X: tl.constexpr,
    Z: tl.constexpr,
    U: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dU = tl.load(dU)
    dV = tl.load(dV)
    dX = tl.load(dX)
    S = tl.load(S)
    D = tl.load(D)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (Z * X * X)

    iy = offs % X
    ix = (offs // X) % X
    iz = offs // (X * X)

    half = 0.5
    o_x = dX * (-half * X + half)
    o_z = dX * (-half * Z + half)
    o_u = dU * (-half * U + half)
    o_v = dV * (-half * V + half)

    x = o_x + ix.to(tl.float32) * dX
    y = o_x + iy.to(tl.float32) * dX
    z = o_z + iz.to(tl.float32) * dX

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for iangle in tl.range(0, NANGLES):
        theta = tl.load(thetas + iangle)
        ct = tl.cos(theta)
        st = tl.sin(theta)

        x_rot = x * ct + y * st
        y_rot = -x * st + y * ct

        frac_ray = (D + S) / (x_rot + S)
        u = y_rot * frac_ray
        v = z * frac_ray

        u_pix = (u - o_u) / dU
        v_pix = (v - o_v) / dV

        u0 = tl.floor(u_pix).to(tl.int32)
        v0 = tl.floor(v_pix).to(tl.int32)
        u1 = u0 + 1
        v1 = v0 + 1

        wu = u_pix - u0.to(tl.float32)
        wv = v_pix - v0.to(tl.float32)

        m00 = mask & (u0 >= 0) & (u0 < U) & (v0 >= 0) & (v0 < V)
        m01 = mask & (u1 >= 0) & (u1 < U) & (v0 >= 0) & (v0 < V)
        m10 = mask & (u0 >= 0) & (u0 < U) & (v1 >= 0) & (v1 < V)
        m11 = mask & (u1 >= 0) & (u1 < U) & (v1 >= 0) & (v1 < V)

        proj_base = iangle * V * U
        p00 = tl.load(projs + proj_base + v0 * U + u0, mask=m00, other=0.0)
        p01 = tl.load(projs + proj_base + v0 * U + u1, mask=m01, other=0.0)
        p10 = tl.load(projs + proj_base + v1 * U + u0, mask=m10, other=0.0)
        p11 = tl.load(projs + proj_base + v1 * U + u1, mask=m11, other=0.0)

        val0 = p00 * (1.0 - wu) + p01 * wu
        val1 = p10 * (1.0 - wu) + p11 * wu
        acc += val0 * (1.0 - wv) + val1 * wv

    tl.store(vol + offs, acc, mask=mask)


@partial(jax.jit, static_argnames=("X", "Z"))
def _get_bp_angle(proj, theta, dU, dV, X, Z, dX, S, D):
    out_shape = jax.ShapeDtypeStruct((Z, X, X), proj.dtype)
    block = 256
    grid = (triton.cdiv(Z * X * X, block),)

    return jt.triton_call(
        proj,
        theta,
        dU,
        dV,
        dX,
        S,
        D,
        kernel=_bp_angle_kernel,
        out_shape=out_shape,
        grid=grid,
        X=X,
        Z=Z,
        U=proj.shape[1],
        V=proj.shape[0],
        BLOCK=block,
        num_warps=4,
    )


@partial(jax.jit, static_argnames=("X", "Z"))
def get_bp_per_angle(projs, thetas, dU, dV, X, Z, dX, S, D):
    def body_fun(carry, elem):
        proj, theta = elem
        bp = _get_bp_angle(proj, theta, dU, dV, X, Z, dX, S, D)
        carry = carry + bp
        return carry, None

    vol, _ = jax.lax.scan(
        body_fun,
        jnp.zeros((Z, X, X), dtype=projs.dtype),
        (projs, thetas),
    )

    return vol


@partial(jax.jit, static_argnames=("X", "Z"))
def get_bp(projs, thetas, dU, dV, X, Z, dX, S, D):
    out_shape = jax.ShapeDtypeStruct((Z, X, X), projs.dtype)
    block = 256
    grid = (triton.cdiv(Z * X * X, block),)

    return jt.triton_call(
        projs,
        thetas,
        dU,
        dV,
        dX,
        S,
        D,
        kernel=_bp_fused_kernel,
        out_shape=out_shape,
        grid=grid,
        NANGLES=projs.shape[0],
        X=X,
        Z=Z,
        U=projs.shape[2],
        V=projs.shape[1],
        BLOCK=block,
        num_warps=4,
    )
