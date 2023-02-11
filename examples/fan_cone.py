from pylab import *

from jaxtomo import fan_fp
from jaxtomo import cone_fp
from jaxtomo import util

util.set_preallocation(False)
util.set_cuda_device(2)

vol_sh_x = 256
nslices = vol_sh_x

vol = zeros((vol_sh_x, vol_sh_x, vol_sh_x), dtype="float32")
vol[
    nslices//4 : nslices//4 + nslices//2,
    vol_sh_x//4 : vol_sh_x//4 + vol_sh_x//2,
    vol_sh_x//4 : vol_sh_x//4 + vol_sh_x//2
] = 1

z_source = vol_sh_x
z_det = z_source * 10
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


fig, ax = subplots(2, 3, figsize=(14, 8))
ax[0, 0].imshow(vol[nrows//2])
ax[0, 1].imshow(proj_fan[:vol_sh_x, nrows//4])
ax[0, 2].imshow(proj_cone[:vol_sh_x, nrows//4])
ax[0, 0].set_title("Original")
ax[0, 1].set_title("Sinogram (fan beam)")
ax[0, 2].set_title("Sinogram (cone beam)")
ax[1, 1].imshow(proj_fan[n_angles//16])
ax[1, 2].imshow(proj_cone[n_angles//16])
tight_layout()
show()
