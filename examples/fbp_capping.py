from pylab import *
import scipy.ndimage as nd
import jax

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp

from jaxtomo import proj_filter
from jaxtomo import util

util.set_preallocation(False)
util.set_cuda_device(2)

fname = "/buffer/schmid/forbild_head.npz"

vol_sh_x = 256
nslices = 16

vol = zeros((nslices, vol_sh_x, vol_sh_x), dtype="float32")
vol[:, 16:-16, 16:-16] = util.roundmask(vol_sh_x - 32, vol_sh_x - 32)


z_source = 500
z_det = 4000
M = (z_det + z_source) / abs(z_source)

n_angles = vol_sh_x * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = vol_sh_x
px_width = M
nrows = nslices
px_height = M
vol_sh_y = nslices

proj = P_fp.get_fp(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)

w = proj_filter.gen_fbp_weights(vol_sh_x)
proj_f = proj_filter.proj_filter(proj, w)

fbp = P_bp.get_bp(
    proj_f, angles, 
    px_width, px_height, 
    vol_sh_x, vol_sh_y, vx_size, 
    z_source, z_det
)
fbp = fbp / n_angles


figure(figsize=(12, 4))
subplot(131)
imshow(vol[nslices//2], vmin=0, vmax=1)
axhline(vol_sh_x//2, color="blue")

subplot(132)
imshow(fbp[nslices//2], vmin=0, vmax=1)
axhline(vol_sh_x//2, color="orange")

subplot(133)
plot(vol[nslices//2, vol_sh_x//2], label="Ground Truth")
plot(fbp[nslices//2, vol_sh_x//2], label="FBP")
legend()
tight_layout()
