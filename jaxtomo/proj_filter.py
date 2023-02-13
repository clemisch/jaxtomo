import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


def gen_fbp_weights(ncols):
    nfreq = np.maximum(64, np.power(2, np.int32(np.ceil(np.log2(np.sqrt(2) * ncols + 1))) + 1))
    freqs = np.fft.fftfreq(nfreq)[:int(nfreq / 2 + 1)]
    w = np.pi * np.absolute(freqs)
    w[np.abs(freqs) > .5] = 0.

    return w


@jax.jit
def proj_filter(proj, weights):
    ncols = proj.shape[-1]
    nfreq = np.maximum(64, np.power(2, np.int32(np.ceil(np.log2(np.sqrt(2) * ncols + 1))) + 1))
    pad_width = (nfreq - ncols) // 2
    proj = jnp.pad(proj, pad_width=((0, 0), (0, 0), (pad_width, pad_width)))
    proj_fft = jnp.fft.rfft(proj, axis=-1) * weights[None, None]
    result = jnp.fft.irfft(proj_fft, axis=-1)
    
    return result[..., pad_width:-pad_width]


@partial(jax.pmap, in_axes=(0, None))
def _proj_filter_pmap(proj, weights):
    proj_f = proj_filter(proj, weights)
    return proj_f


@jax.jit
def proj_filter_pmap(proj, weights):
    nangles = proj.shape[0]
    ndevices = jax.device_count()
    assert nangles % ndevices == 0

    proj = proj.reshape(ndevices, -1, *proj.shape[1:])
    proj_f = _proj_filter_pmap(proj, weights)
    proj_f = proj_f.reshape(nangles, *proj_f.shape[-2:])

    return proj_f
