import os
import jax


def set_cuda_device(n):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n)


def set_platform(platform):
    assert platform in {"cpu", "gpu"}
    jax.config.update("jax_platform_name", platform)
    if platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""



def set_preallocation(is_prealloc):
    if not is_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str("0.90")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"


def multi_vmap(fun, in_axes, out_axes):
    """ vmap over several axes """
    batched_fun = fun
    for inax, outax in zip(in_axes, out_axes):
        batched_fun = jax.vmap(batched_fun, inax, outax)
    return batched_fun
