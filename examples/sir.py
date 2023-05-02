from pylab import *
import scipy.ndimage as nd
from tqdm import tqdm
from functools import partial

import jax
import jax.numpy as jnp

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp
from jaxtomo import fan_fp_bp
from jaxtomo import proj_filter

from jaxtomo import util
util.set_preallocation(True)
util.set_cuda_device(2)

# pre-rendered 3D volume 
# TODO: generate in this package
fname = "/users/schmid/forbild_head.npz"

vol = array(load(fname)["arr_0"])
vol = vol.transpose(1, 0, 2)
N = 128
vol = vol / N
vol = nd.zoom(vol, N / vol.shape[0], order=1)
vol = jax.device_put(vol)

z_source = N
z_det = z_source * 10
M = (z_det + z_source) / abs(z_source)

n_angles = N * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = N
px_width = M
nrows = 8
px_height = 1.

proj = P_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)

vol_sh_x = N
vol_sh_y = nrows

w = proj_filter.gen_fbp_weights(N)

fbp = P_bp.get_bp(
    proj_filter.proj_filter(proj, w), 
    angles, 
    px_width, px_height, 
    vol_sh_x, vol_sh_y, vx_size, 
    z_source, z_det
) / n_angles



T = exp(-proj)
ncounts = 1e4
T_noise = (poisson(T * ncounts) / ncounts).astype("float32")
proj_noise = -log(T_noise)

fbp_noise = P_bp.get_bp(
    proj_filter.proj_filter(proj_noise, w), 
    angles, 
    px_width, px_height, 
    vol_sh_x, vol_sh_y, vx_size, 
    z_source, z_det
) / n_angles


###############################################################################
# SIR
###############################################################################

get_fp, get_bp = fan_fp_bp.get_fp_bp(
    angles, 
    vol_sh_x, vx_size, 
    vol_sh_y, 
    ncols, px_width, 
    nrows, px_height, 
    z_source, z_det
)


@jax.jit
def get_tv(x):
    cost = 0.0
    for i in range(x.ndim):
        xd = jnp.diff(x, axis=i)
        cost += jnp.sum(jnp.abs(xd))
    return cost

get_tv_grad = jax.jit(jax.grad(get_tv))


@jax.jit
def get_loglike(x, y, w):
    mu = get_fp(x)
    T = jnp.exp(-mu)
    cost = jnp.sum(w * jnp.square(T - y))

    return cost

get_loglike_grad = jax.jit(jax.grad(get_loglike))



###############################################################################
# OSOGM
###############################################################################

@partial(jax.jit, static_argnames="niter")
def osogm(x0, y, w, lambd, M, niter):
    def body_fun(carry, _):
        i, x, z, theta = carry

        grad_data = get_loglike_grad(x, y, w)
        grad_reg = get_tv_grad(x)
        grad_tot = grad_data + lambd * grad_reg

        z_ = z
        theta_ = theta
        z = x - grad_tot / M
        theta = 0.5 * (1. + jnp.sqrt(1. + 4. * jnp.square(theta)))
        x = z + (theta_ - 1.) / theta * (z - z_) + theta_ / theta * (z - x)

        l2_data = jnp.mean(grad_data**2)
        l2_reg = jnp.mean(grad_reg**2) * lambd
        factor = l2_data / l2_reg

        jax.debug.print(
            "{i}\t{theta}\t{factor}", 
            i=i, theta=theta, factor=factor
        )

        carry = i+1, x, z, theta
        stack = l2_data, l2_reg, x[4]

        return carry, stack

    (_, xopt, _, _), (l2_data, l2_reg, xlst) = jax.lax.scan(
        body_fun, 
        (0, x0, x0, 1.0),
        None,
        length=niter
    )


    return xopt, l2_data, l2_reg, xlst


# gaussian counts
w = 1. / T_noise
w = jnp.where(T_noise > 0, w, 1)
y = jax.device_put(T_noise)
M = get_bp(2 * w * y**2 * get_fp(jnp.ones_like(vol)))
# M = get_bp(w * y**2 * get_fp(jnp.ones_like(vol)))


xopt, l2_data, l2_reg, xlst = osogm(
    jnp.zeros_like(fbp),
    y, w, 1e-3, M, 100
)


###############################################################################
# LBFGS
###############################################################################

import jaxopt


@jax.jit
def run_sir(x0, y, w, lambd, maxiter):
    def body_fun(i, arg):
        jax.debug.print("{i}", i=i)
        x, state = arg
        x, state = opt.update(x, state)
        return x, state

    opt = jaxopt.LBFGS(
        lambda x: get_loglike(x, y, w) + lambd * get_tv(x),
        maxiter=maxiter,
        tol=1e-3,
    )
    state = opt.init_state(x0)

    xopt, state = jax.lax.fori_loop(
        0, maxiter,
        body_fun,
        (x0, state)
    )

    return xopt



xopt2 = run_sir(jnp.zeros_like(fbp), y, w, 1e-2, 100)






@partial(jax.jit, static_argnames="maxiter")
def run_sir_scan(x0, y, w, lambd, maxiter):
    def body_fun(carry, _):
        i, x, state = carry
        jax.debug.print("{i}", i=i)

        x, state = opt.update(x, state)

        carry = i+1, x, state
        stack = x[4]

        return carry, stack

    opt = jaxopt.LBFGS(
        lambda x: get_loglike(x, y, w) + lambd * get_tv(x),
        maxiter=maxiter,
        tol=1e-3,
    )
    state = opt.init_state(x0)

    (_, xopt, _), xlst = jax.lax.scan(
        body_fun,
        (0, x0, state),
        None,
        length=maxiter
    )

    return xopt, xlst


xopt2, xlst = run_sir_scan(jnp.zeros_like(fbp), y, w, 1e-2, 100)
