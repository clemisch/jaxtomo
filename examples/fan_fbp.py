import scipy.ndimage as nd
from skimage.data import shepp_logan_phantom
import jax

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp
from jaxtomo import proj_filter


img = shepp_logan_phantom()
N = 256
img = nd.zoom(img, N / img.shape[0], order=1)
vol = img[None].repeat(16, 0)
vol = jax.device_put(vol)

S = -1000
D = 1000
M = (D - S) / abs(S)

n_angles = N * 2
angles = linspace(0, 2 * pi, n_angles, False)
projs = P_fp.get_fp(vol, angles, 1., N, M, 10, 1., S, D)
vol_bp = P_bp.get_bp(projs, angles, M, 1., N, 16, 1., S, D)

w = proj_filter.gen_fbp_weights(N)
projs_f = proj_filter.proj_filter(projs, w)
fbp = P_bp.get_bp(projs_f, angles, M, 1., N, 16, 1., S, D)
fbp = fbp / n_angles


fig, ax = subplots(1, 3, figsize=(14, 5))
ax[0].imshow(vol[0], vmin=0, vmax=1)
ax[1].imshow(projs[:N, 0])
ax[2].imshow(fbp[7], vmin=0, vmax=1)
ax[0].set_title("Original")
ax[1].set_title("Sinogram (fan beam)")
ax[2].set_title("FBP")
tight_layout()
show()
