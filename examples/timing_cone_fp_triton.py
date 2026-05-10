import argparse
import os
import timeit


def _parse_gpu(value):
    if value == "None":
        return None
    return eval(value)


def _configure_runtime(gpu, prealloc):
    if gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if isinstance(gpu, int):
            gpu = (gpu,)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu))

    if not prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"


def _time(fun, warmup, repeat):
    for _ in range(warmup):
        fun().block_until_ready()
    return timeit.timeit(lambda: fun().block_until_ready(), number=repeat) / repeat


parser = argparse.ArgumentParser(
    description="Compare JAX and Triton cone forward projection timings.",
)
parser.add_argument("--gpu", default="None", help="ID(s) of GPUs to use (None, int or tuple)")
parser.add_argument("--size", default="256", help="Size of volume and projections to time")
parser.add_argument("--dtype", default="float32", help="dtype of arrays used")
parser.add_argument("--prealloc", default=False, help="Use preallocation of GPU memory", action="store_true")
parser.add_argument("--warmup", default="1", help="Number of warmup runs")
parser.add_argument("--repeat", default="5", help="Number of timed runs")
parser.add_argument("--skip-jax", default=False, help="Skip the plain JAX projector", action="store_true")
parser.add_argument("--skip-triton", default=False, help="Skip the Triton projector", action="store_true")
parser.add_argument("--check", default=False, help="Compare output values before timing", action="store_true")
args = parser.parse_args()

GPU = _parse_gpu(args.gpu)
SIZE = int(args.size)
DTYPE = str(args.dtype)
PREALLOC = args.prealloc
WARMUP = int(args.warmup)
REPEAT = int(args.repeat)
SKIP_JAX = args.skip_jax
SKIP_TRITON = args.skip_triton
CHECK = args.check

_configure_runtime(GPU, PREALLOC)

import jax
import jax.numpy as jnp

from jaxtomo.projectors import cone_fp

if GPU is None:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

print("gpu         :", repr(GPU))
print("prealloc    :", repr(PREALLOC))
print("size        :", repr(SIZE))
print("dtype       :", repr(DTYPE))
print("warmup      :", repr(WARMUP))
print("repeat      :", repr(REPEAT))
print("skip_jax    :", repr(SKIP_JAX))
print("skip_triton :", repr(SKIP_TRITON))
print("check       :", repr(CHECK))
print("devices     :", jax.devices())

HAS_GPU_DEVICE = any(device.platform == "gpu" for device in jax.devices())
if not SKIP_TRITON and not HAS_GPU_DEVICE:
    print("triton      : skipped because no GPU device is active")
    SKIP_TRITON = True

if not SKIP_TRITON:
    from jaxtomo.projectors import cone_fp_triton

vx_size = 1.0
sh_vol = (SIZE,) * 3
sh_proj = (SIZE,) * 3
nangles, nrows, ncols = sh_proj
px_height = 2.0
px_width = 2.0
z_source = sh_vol[1]
z_det = z_source

angles = jnp.linspace(0.0, 2 * jnp.pi, nangles, False)
vol = jnp.ones(sh_vol, dtype=DTYPE)

fp_args = (
    vol,
    angles,
    vx_size,
    ncols,
    px_width,
    nrows,
    px_height,
    z_source,
    z_det,
)

jobs = []
if not SKIP_JAX:
    jobs.append(("JAX", lambda: cone_fp.get_fp(*fp_args)))
if not SKIP_TRITON:
    jobs.append(("Triton", lambda: cone_fp_triton.get_fp(*fp_args)))

if CHECK and len(jobs) == 2:
    ref = jobs[0][1]().block_until_ready()
    out = jobs[1][1]().block_until_ready()
    diff = jnp.abs(ref - out)
    denom = jnp.maximum(1.0, jnp.abs(ref).max())
    print("==== CHECK ====")
    print(f"max abs : {float(diff.max()):.6e}")
    print(f"mean abs: {float(diff.mean()):.6e}")
    print(f"rel max : {float(diff.max() / denom):.6e}")

print("==== FP ====")
nrays = nangles * nrows * ncols
for name, fun in jobs:
    dt = _time(fun, WARMUP, REPEAT)
    dt_ray = dt / nrays
    grays = nrays / 1000.0**3 / dt
    print(
        f"{name:7} {str(sh_vol):15} -> {str(sh_proj):15} : "
        f"{dt * 1e3:5.0f} ms , "
        f"{dt_ray * 1e6:5.2f} us per pixel , "
        f"{grays:2.3f} GRays/s"
    )
