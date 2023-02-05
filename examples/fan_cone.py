from pylab import *
import scipy.ndimage as nd
from skimage.data import shepp_logan_phantom
import jax

from jaxtomo import fan_fp
from jaxtomo import cone_fp

from jaxtomo import util
util.set_preallocation(False)
util.set_cuda_device(2)

fname = "/buffer/schmid/forbild_head.npz"

vol = array(load(fname)["arr_0"])
vol = vol.transpose(1, 0, 2)
vol_sh_x = 256
vol = vol / vol_sh_x
vol = nd.zoom(vol, vol_sh_x / vol.shape[0], order=1)
vol = jax.device_put(vol)
nslices = vol.shape[0]

z_source = 350
z_det = 2000
M = (z_det + z_source) / abs(z_source)

n_angles = vol_sh_x * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = vol_sh_x
px_width = M
nrows = nslices
px_height = 1.
vol_sh_y = nslices

proj_fan = fan_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)


px_height = M

proj_cone = cone_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)


fig, ax = subplots(1, 3, figsize=(14, 5))
ax[0].imshow(vol[nrows//2])
ax[1].imshow(proj_fan[:vol_sh_x, nrows//5 * 2])
ax[2].imshow(proj_cone[:vol_sh_x, nrows//5 * 2])
ax[0].set_title("Original")
ax[1].set_title("Sinogram (fan beam)")
ax[2].set_title("Sinogram (cone beam)")
tight_layout()
show()
