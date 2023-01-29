from pylab import *
import scipy.ndimage as nd
from skimage.data import shepp_logan_phantom
import jax

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp
from jaxtomo import proj_filter

# from jaxtomo import util
# util.set_preallocation(False)
# util.set_cuda_device(2)

img = shepp_logan_phantom()
vol_sh_x = 256
nslices = 10
img = nd.zoom(img, vol_sh_x / img.shape[0], order=1)
vol = img[None].repeat(nslices, 0)
vol = jax.device_put(vol)

z_source = 1000
z_det = 1000
M = (z_det + z_source) / abs(z_source)

n_angles = vol_sh_x * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = vol_sh_x
px_width = M
nrows = 10
px_height = 1.
vol_sh_y = nslices

projs = P_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)

w = proj_filter.gen_fbp_weights(vol_sh_x)
projs_f = proj_filter.proj_filter(projs, w)


fbp = P_bp.get_bp(
    projs_f, angles, 
    px_width, px_height, 
    vol_sh_x, vol_sh_y, vx_size, 
    z_source, z_det
)
fbp = fbp / n_angles


fig, ax = subplots(1, 3, figsize=(14, 5))
ax[0].imshow(vol[0], vmin=0, vmax=1)
ax[1].imshow(projs[:vol_sh_x, 0])
ax[2].imshow(fbp[7], vmin=0, vmax=1)
ax[0].set_title("Original")
ax[1].set_title("Sinogram (fan beam)")
ax[2].set_title("FBP")
tight_layout()
show()
