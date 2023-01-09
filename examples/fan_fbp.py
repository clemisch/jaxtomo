from pylab import *
import scipy.ndimage as nd
from skimage.data import shepp_logan_phantom
from skimage.transform import radon
import jax
import jax.numpy as jnp

import sys
sys.path.append("/users/schmid/phd/code/other/jax_projector")

import fan_fp as P_fp
import fan_bp as P_bp
import proj_filter
import util

util.set_cuda_device(2)
util.set_preallocation(False)

###############################################################################
# 2D PARALLEL
###############################################################################

img = shepp_logan_phantom()
N = 256
img = nd.zoom(img, N / img.shape[0], order=1)
vol = img[None].repeat(16, 0)
vol = jax.device_put(vol)

S = -1000
D = 1000
M = (D - S) / abs(S)

angles = linspace(0, 2 * pi, N * 2, False)
projs = P_fp.get_projs_2d(vol, angles, 1., N, M, 10, 1., S, D).block_until_ready()

vol_bp = P_bp.get_bp(projs, angles, M, 1., N, 16, 1., S, D)





w = proj_filter.gen_fbp_weights(N)
projs_f = proj_filter.proj_filter(projs, w)

fbp = P_bp.get_bp(projs_f, angles, M, 1., N, 16, 1., S, D)



T = exp(-projs / 50)
ncounts = 1e4
T_noise = poisson(T * ncounts) / ncounts
mu_noise = -log(T_noise) * 50

fbp_noise = P_bp.get_bp(proj_filter.proj_filter(mu_noise, w), angles, M, 1., N, 16, 1., S, D)
