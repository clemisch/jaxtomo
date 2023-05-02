from pylab import *
import scipy.ndimage as nd
from tqdm import tqdm
from functools import partial

import jax
import jax.numpy as jnp

from jaxtomo import cone_fp as P_fp
from jaxtomo import cone_bp as P_bp
from jaxtomo import cone_fp_bp
from jaxtomo import proj_filter

from jaxtomo import util
util.set_preallocation(True)
util.set_cuda_device(2)

fname = "/users/schmid/forbild_head.npz"

vol = array(load(fname)["arr_0"])
vol = vol.transpose(1, 0, 2)
N = 128
vol = vol / N
vol = nd.zoom(vol, N / vol.shape[0], order=1)
vol = jax.device_put(vol)



z_source = N
z_det = z_source * 5
M = (z_det + z_source) / abs(z_source)

n_angles = N * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = N
px_width = M * sqrt(2)
nrows = N
px_height = M * sqrt(2)

proj = P_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)

w = proj_filter.gen_fbp_weights(N)

fbp = P_bp.get_bp(
    proj_filter.proj_filter(proj, w), 
    angles, 
    px_width, px_height, 
    N, N, vx_size, 
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
    N, N, vx_size, 
    z_source, z_det
) / n_angles


###############################################################################
# COST FUNCTIONS
###############################################################################

get_fp, get_bp = cone_fp_bp.get_fp_bp(
    angles, 
    N, vx_size, 
    N, 
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
# JAXOPT LBFGS
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


y = jax.device_put(T_noise)
w = 1 / y
w = jnp.where(y > 0, w, 1)
x0 = jnp.zeros_like(fbp)

xopt = run_sir(x0, y, w, 1e-2, 100)
