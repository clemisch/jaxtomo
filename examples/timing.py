import os
import scipy.ndimage as nd
import timeit
import jax
import jax.numpy as jnp

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp



from jaxtomo import util

GPU = int(os.environ.get("GPU", 0))
PREALLOC = bool(os.environ.get("PREALLOC", "0") != "0")
SCAN = bool(os.environ.get("SCAN", "0") != "0")

print(f"GPU={GPU}, prealloc={PREALLOC}, scan={SCAN}")

util.set_preallocation(PREALLOC)
util.set_cuda_device(GPU)


def get_timing_fp(sh_vol, sh_proj):
    vx_size = 1.
    nangles = sh_proj[0]
    nrows = sh_proj[1]
    ncols = sh_proj[2]
    px_height = 1.
    px_width = 2.
    z_source = 1000
    z_det = 1000
    angles = jnp.linspace(0., 2 * jnp.pi, nangles, False)
    vol = jnp.ones(sh_vol, dtype="float32")

    def get_proj():
        projs = P_fp.get_fp(
            vol, angles, 
            vx_size, 
            ncols, px_width, 
            nrows, px_height,
            z_source, z_det, SCAN
        ).block_until_ready()
        return projs

    _ = get_proj()
    dt = timeit.timeit(get_proj, number=5) / 5.

    return dt


def get_timing_bp(sh_vol, sh_proj):
    vx_size = 1.
    nangles = sh_proj[0]
    vol_sh_x = sh_vol[1]
    vol_sh_y = sh_vol[0]
    px_height = 1.
    px_width = 2.
    z_source = 1000
    z_det = 1000
    angles = jnp.linspace(0., 2 * jnp.pi, nangles, False)
    proj = jnp.ones(sh_proj, dtype="float32")

    def get_vol():
        vol = P_bp.get_bp(
            proj, angles, 
            px_width, px_height, 
            vol_sh_x, vol_sh_y, vx_size, 
            z_source, z_det
        ).block_until_ready()
        return vol

    _ = get_vol()
    dt = timeit.timeit(get_vol, number=5) / 5.

    return dt



configs = [
    # ((128,) * 3, (256, 8, 128)),
    # ((256,) * 3, (512, 8, 256)),
    # ((256,) * 3, (512, 16, 256)),
    # ((256,) * 3, (512, 32, 256)),
    # ((512,) * 3, (1024, 8, 512)),
    # ((48, 800, 800), (2400, 32, 672)),
    ((800, 800, 800), (2400, 32, 672)),
]

print("*** FP ***")
for config in configs:
    sh_vol, sh_proj = config
    dt = get_timing_fp(sh_vol, sh_proj)

    nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
    dt_ray = dt / nrays

    print(f"{str(sh_vol):15} -> {str(sh_proj):15} : {dt * 1e3:5.0f} ms , {dt_ray * 1e6:5.2f} µs per pixel")




exit(0)


print("*** BP ***")
for config in configs:
    sh_vol, sh_proj = config
    dt = get_timing_bp(sh_vol, sh_proj)

    nvoxels = sh_vol[0] * sh_vol[1] * sh_vol[2]
    dt_voxel = dt / nvoxels

    print(f"{str(sh_proj):15} -> {str(sh_vol):15} : {dt * 1e3:5.0f} ms , {dt_voxel * 1e6:5.2f} µs per voxel")
