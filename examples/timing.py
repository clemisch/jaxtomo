import os
import scipy.ndimage as nd
import timeit
import jax
import jax.numpy as jnp

from jaxtomo import fan_fp
from jaxtomo import cone_fp



from jaxtomo import util
GPU = int(os.environ.get("GPU", 0))
PREALLOC = bool(os.environ.get("PREALLOC", 0) != "0")
PMAP = bool(os.environ.get("PMAP", 0) != "0")
print(f"GPU conf: GPU #{GPU}, prealloc={PREALLOC}, pmap={PMAP}")
util.set_preallocation(PREALLOC)
util.set_cuda_device(GPU)


def get_timing_fp_fan(sh_vol, sh_proj):
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

    if PMAP:
        fp_fun = fan_fp.get_fp_pmap
    else:
        fp_fun = fan_fp.get_fp

    def get_proj():
        projs = fp_fun(
            vol, angles, 
            vx_size, 
            ncols, px_width, 
            nrows, px_height,
            z_source, z_det
        ).block_until_ready()
        return projs

    _ = get_proj()
    dt = timeit.timeit(get_proj, number=5) / 5.

    return dt


def get_timing_fp_cone(sh_vol, sh_proj):
    vx_size = 1.
    nangles = sh_proj[0]
    nrows = sh_proj[1]
    ncols = sh_proj[2]
    px_height = 2.
    px_width = 1.
    z_source = 1000
    z_det = 1000
    angles = jnp.linspace(0., 2 * jnp.pi, nangles, False)
    vol = jnp.ones(sh_vol, dtype="float32")

    if PMAP:
        fp_fun = cone_fp.get_fp_pmap
    else:
        fp_fun = cone_fp.get_fp

    def get_proj():
        projs = fp_fun(
            vol, angles, 
            vx_size, 
            ncols, px_width, 
            nrows, px_height,
            z_source, z_det
        ).block_until_ready()
        return projs

    _ = get_proj()
    dt = timeit.timeit(get_proj, number=5) / 5.

    return dt





configs = [
    ((128,) * 3, (256, 8, 128)),
    ((256,) * 3, (512, 8, 256)),
    ((256,) * 3, (512, 16, 256)),
    ((256,) * 3, (512, 32, 256)),
    # ((512,) * 3, (1024, 8, 512)),
]

print("*** FAN ***")
for config in configs:
    sh_vol, sh_proj = config
    dt = get_timing_fp_fan(sh_vol, sh_proj)

    nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
    dt_ray = dt / nrays

    print(f"{str(sh_vol):15} -> {str(sh_proj):15} : {dt * 1e3:5.0f} ms , {dt_ray * 1e6:5.2f} µs per pixel")



print("*** CONE ***")
for config in configs:
    sh_vol, sh_proj = config
    dt = get_timing_fp_cone(sh_vol, sh_proj)

    nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
    dt_ray = dt / nrays

    print(f"{str(sh_vol):15} -> {str(sh_proj):15} : {dt * 1e3:5.0f} ms , {dt_ray * 1e6:5.2f} µs per pixel")
