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


def _print_timing(name, sh_in, sh_out, dt, nwork, unit, rate_unit):
    dt_work = dt / nwork
    gwork_s = nwork / 1000.0**3 / dt
    print(
        f"{name:10} {str(sh_in):15} -> {str(sh_out):15} : "
        f"{dt * 1e3:5.0f} ms , "
        f"{dt_work * 1e6:5.2f} us per {unit} , "
        f"{gwork_s:2.3f} {rate_unit}"
    )


parser = argparse.ArgumentParser(
    description="Compare JAX and Triton cone FP and BP timings.",
)
parser.add_argument("--gpu", default="None", help="ID(s) of GPUs to use (None, int or tuple)")
parser.add_argument("--size", default="0", help="Size of volume and projections to time")
parser.add_argument("--dtype", default="float32", help="dtype of arrays used")
parser.add_argument("--prealloc", default=False, help="Use preallocation of GPU memory", action="store_true")
parser.add_argument("--fp", default=False, help="Benchmark forward-projection", action="store_true")
parser.add_argument("--bp", default=False, help="Benchmark back-projection", action="store_true")
parser.add_argument("--warmup", default="1", help="Number of warmup runs")
parser.add_argument("--repeat", default="5", help="Number of timed runs")
parser.add_argument("--skip-jax", default=False, help="Skip the plain JAX projectors", action="store_true")
parser.add_argument("--skip-triton", default=False, help="Skip the Triton projectors", action="store_true")
args = parser.parse_args()

GPU = _parse_gpu(args.gpu)
SIZE = int(args.size)
DTYPE = str(args.dtype)
PREALLOC = args.prealloc
FP = args.fp
BP = args.bp
WARMUP = int(args.warmup)
REPEAT = int(args.repeat)
SKIP_JAX = args.skip_jax
SKIP_TRITON = args.skip_triton

_configure_runtime(GPU, PREALLOC)

import jax
import jax.numpy as jnp

from jaxtomo.projectors import cone_bp, cone_fp

if GPU is None:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

print("gpu         :", repr(GPU))
print("prealloc    :", repr(PREALLOC))
print("fp          :", repr(FP))
print("bp          :", repr(BP))
print("size        :", repr(SIZE))
print("dtype       :", repr(DTYPE))
print("warmup      :", repr(WARMUP))
print("repeat      :", repr(REPEAT))
print("skip_jax    :", repr(SKIP_JAX))
print("skip_triton :", repr(SKIP_TRITON))
print("devices     :", jax.devices())

has_gpu_device = any(device.platform == "gpu" for device in jax.devices())
if not SKIP_TRITON and not has_gpu_device:
    print("triton      : skipped because no GPU device is active")
    SKIP_TRITON = True

if not SKIP_TRITON:
    from jaxtomo.projectors import cone_bp_triton, cone_fp_triton

if SIZE > 0:
    configs = [
        ((SIZE,) * 3, (SIZE,) * 3),
    ]
else:
    configs = [
        ((8, 512, 512), (1024, 512, 512)),
    ]


def get_timing_fp(sh_vol, sh_proj, fp_fun):
    vx_size = 1.0
    nangles, nrows, ncols = sh_proj
    px_height = 2.0
    px_width = 2.0
    z_source = sh_vol[1]
    z_det = z_source
    angles = jnp.linspace(0.0, 2 * jnp.pi, nangles, False)
    vol = jnp.ones(sh_vol, dtype=DTYPE)

    def get_proj():
        return fp_fun(
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

    return _time(get_proj, WARMUP, REPEAT)


def get_timing_bp(sh_vol, sh_proj, bp_fun):
    vx_size = 1.0
    nangles = sh_proj[0]
    vol_sh_x = sh_vol[1]
    vol_sh_y = sh_vol[0]
    px_height = 2.0
    px_width = 2.0
    z_source = sh_vol[1]
    z_det = z_source
    angles = jnp.linspace(0.0, 2 * jnp.pi, nangles, False)
    proj = jnp.ones(sh_proj, dtype=DTYPE)

    def get_vol():
        return bp_fun(
            proj,
            angles,
            px_width,
            px_height,
            vol_sh_x,
            vol_sh_y,
            vx_size,
            z_source,
            z_det,
        )

    return _time(get_vol, WARMUP, REPEAT)


fp_jobs = []
bp_jobs = []
if not SKIP_JAX:
    fp_jobs.append(("JAX", cone_fp.get_fp))
    bp_jobs.append(("JAX", cone_bp.get_bp))
if not SKIP_TRITON:
    fp_jobs.append(("Triton", cone_fp_triton.get_fp))
    bp_jobs.append(("Triton", cone_bp_triton.get_bp))

if FP:
    print("==== FP ====")
    for sh_vol, sh_proj in configs:
        nrays = sh_proj[0] * sh_proj[1] * sh_proj[2]
        for name, fp_fun in fp_jobs:
            dt = get_timing_fp(sh_vol, sh_proj, fp_fun)
            _print_timing(name, sh_vol, sh_proj, dt, nrays, "ray", "GRays/s")

if BP:
    print("==== BP ====")
    for sh_vol, sh_proj in configs:
        nvoxels = sh_vol[0] * sh_vol[1] * sh_vol[2]
        for name, bp_fun in bp_jobs:
            dt = get_timing_bp(sh_vol, sh_proj, bp_fun)
            _print_timing(name, sh_proj, sh_vol, dt, nvoxels, "voxel", "GVoxels/s")
