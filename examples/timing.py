import os
import timeit
import jax.numpy as jnp
import argparse

from jaxtomo import cone_fp, cone_bp
from jaxtomo import util

###############################################################################
# ARGUMENTS
###############################################################################

# sorry for this horrible usage of argparse
parser = argparse.ArgumentParser(description="Timing of cone FP and BP")
parser.add_argument("--gpu"     , default="None", help="ID(s) of GPUs to use")
parser.add_argument("--prealloc", default="1"   , help="Use preallocation of GPU memory")
parser.add_argument("--pmap"    , default="0"   , help="Use multi-GPU via jax.pmap (requires multiple , identical GPUs)")
parser.add_argument("--fp"      , default="1"   , help="Time forward projection")
parser.add_argument("--bp"      , default="1"   , help="Time back projection")
parser.add_argument("--size"    , default="0"   , help="Size of volume and projections to time")

args = parser.parse_args()
GPU      = eval(args.gpu)   # None, int or tuple
PREALLOC = bool(args.prealloc)
PMAP     = bool(args.pmap)
FP       = bool(args.fp)
BP       = bool(args.bp)
SIZE     = int(args.size)

print("gpu      :", repr(GPU))
print("prealloc :", repr(PREALLOC))
print("pmap     :", repr(PMAP))
print("fp       :", repr(FP))
print("bp       :", repr(BP))
print("size     :", repr(SIZE))

if GPU is None:
    util.set_platform("cpu")
else:
    if isinstance(GPU, int): GPU = (GPU,)
    util.set_platform("gpu")
    util.set_preallocation(PREALLOC)
    util.set_cuda_device(*GPU, verbose=False)

###############################################################################
# TIMING
###############################################################################

def get_timing_fp(sh_vol, sh_proj):
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


def get_timing_bp(sh_vol, sh_proj):
    vx_size = 1.
    nangles = sh_proj[0]
    vol_sh_x = sh_vol[1]
    vol_sh_y = sh_vol[0]
    px_height = 2.
    px_width = 2.
    z_source = sh_vol[1]
    z_det = z_source
    angles = jnp.linspace(0., 2 * jnp.pi, nangles, False)
    proj = jnp.ones(sh_proj, dtype="float32")

    if PMAP:
        bp_fun = cone_bp.get_bp_pmap
    else:
        bp_fun = cone_bp.get_bp

    def get_vol():
        vol = bp_fun(
            proj, angles, 
            px_width, px_height, 
            vol_sh_x, vol_sh_y, vx_size, 
            z_source, z_det
        ).block_until_ready()
        
        return vol

    _ = get_vol()
    dt = timeit.timeit(get_vol, number=5) / 5.

    return dt


if SIZE > 0:
    configs = [
        ((SIZE,) * 3, (SIZE,) * 3),
    ]
else:
    configs = [
        ((128,) * 3, (256, 8, 128)),
        ((256,) * 3, (512, 8, 256)),
        ((256,) * 3, (512, 16, 256)),
        ((256,) * 3, (512, 32, 256)),
        ((512,) * 3, (1024, 8, 512)),
        ((512,) * 3, (1024, 512, 512)),
        ((48, 800, 800), (2400, 32, 672)),
    ]


if FP:
    print("*** FP ***")
    for config in configs:
        sh_vol, sh_proj = config
        dt = get_timing_fp(sh_vol, sh_proj)

        nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
        dt_ray = dt / nrays

        GRay = nrays / 1000.**3
        Grays = GRay / dt

        print((
            f"{str(sh_vol):15} -> {str(sh_proj):15} : {dt * 1e3:5.0f} ms , "
            f"{dt_ray * 1e6:5.2f} µs per pixel , "
            f"{Grays:2.3f} GRays/s"
        ))


if BP:
    print("*** BP ***")
    for config in configs:
        sh_vol, sh_proj = config
        dt = get_timing_bp(sh_vol, sh_proj)

        nvoxels = sh_vol[0] * sh_vol[1] * sh_vol[2]
        dt_voxel = dt / nvoxels

        GRay = nvoxels / 1000.**3
        Grays = GRay / dt

        print((
            f"{str(sh_proj):15} -> {str(sh_vol):15} : {dt * 1e3:5.0f} ms , "
            f"{dt_voxel * 1e6:5.2f} µs per voxel , "
            f"{Grays:2.3f} GRays/s"
        ))
