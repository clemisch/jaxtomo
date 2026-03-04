#!/usr/bin/env python3
import argparse
import json
import timeit

import jax
import jax.numpy as jnp

from jaxtomo import cone_bp, cone_fp
from jaxtomo import util


def get_timing_fp(size: int, dtype: str) -> dict:
    sh_vol = (size, size, size)
    sh_proj = (size, size, size)
    vx_size = 1.0
    nangles, nrows, ncols = sh_proj
    px_height = 2.0
    px_width = 2.0
    z_source = sh_vol[1]
    z_det = z_source
    angles = jnp.linspace(0.0, 2 * jnp.pi, nangles, False)
    vol = jnp.ones(sh_vol, dtype=dtype)

    def run():
        return cone_fp.get_fp(
            vol,
            angles,
            vx_size,
            ncols,
            px_width,
            nrows,
            px_height,
            z_source,
            z_det,
        ).block_until_ready()

    _ = run()
    dt = timeit.timeit(run, number=5) / 5.0
    nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
    return {
        "seconds": dt,
        "milliseconds": dt * 1e3,
        "microseconds_per_pixel": (dt / nrays) * 1e6,
        "grays_per_second": (nrays / 1000.0**3) / dt,
    }


def get_timing_bp(size: int, dtype: str) -> dict:
    sh_vol = (size, size, size)
    sh_proj = (size, size, size)
    vx_size = 1.0
    nangles, _, _ = sh_proj
    vol_sh_x = sh_vol[1]
    vol_sh_y = sh_vol[0]
    px_height = 2.0
    px_width = 2.0
    z_source = sh_vol[1]
    z_det = z_source
    angles = jnp.linspace(0.0, 2 * jnp.pi, nangles, False)
    proj = jnp.ones(sh_proj, dtype=dtype)

    def run():
        return cone_bp.get_bp(
            proj,
            angles,
            px_width,
            px_height,
            vol_sh_x,
            vol_sh_y,
            vx_size,
            z_source,
            z_det,
        ).block_until_ready()

    _ = run()
    dt = timeit.timeit(run, number=5) / 5.0
    nvoxels = sh_vol[0] * sh_vol[1] * sh_vol[2]
    return {
        "seconds": dt,
        "milliseconds": dt * 1e3,
        "microseconds_per_voxel": (dt / nvoxels) * 1e6,
        "grays_per_second": (nvoxels / 1000.0**3) / dt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jaxtomo cone FP/BP timings for one size."
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Select JAX platform.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index when --device=gpu.",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        util.set_platform("cpu")
    else:
        util.set_platform("gpu")
        util.set_cuda_device(args.gpu_id, verbose=False)

    fp = get_timing_fp(args.size, args.dtype)
    bp = get_timing_bp(args.size, args.dtype)

    result = {
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "device": args.device,
        "size": args.size,
        "dtype": args.dtype,
        "fp": fp,
        "bp": bp,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
