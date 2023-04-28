import os
import timeit
import jax.numpy as jnp

from jaxtomo import cone_fp, cone_bp
from jaxtomo import util

util.set_platform("gpu")
util.set_preallocation(True)
util.set_cuda_device(2, verbose=False)

###############################################################################
# TIMING
###############################################################################

sh_vol = (256, 256, 256)
sh_proj = (256, 256, 256)

vx_size = 1.
nangles = sh_proj[0]
nrows = sh_proj[1]
ncols = sh_proj[2]
px_height = 2.
px_width = 2.
z_source = sh_vol[1]
z_det = z_source
angles = jnp.linspace(0., 2 * jnp.pi, nangles, False)
vol = jnp.ones(sh_vol, dtype="float32")

def get_proj():
    projs = cone_fp.get_fp(
        vol, angles, 
        vx_size, 
        ncols, px_width, 
        nrows, px_height,
        z_source, z_det
    ).block_until_ready()

    return projs

# warmup
_ = get_proj()

# profile
@profile
def dummy():
    get_proj()

dummy()
