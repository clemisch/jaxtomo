from pylab import *
import scipy.ndimage as nd
import jax

from jaxtomo import cone_fp as P_fp
from jaxtomo import cone_bp as P_bp

from jaxtomo import proj_filter
from jaxtomo import util

util.set_preallocation(True)
util.set_cuda_device(0,1)

vol_sh_x = 256
nslices = vol_sh_x

# local
# fname = "/buffer/schmid/forbild_head.npz"
# fname = "/home/clem/forbild_head.npz"
# vol = array(load(fname)["arr_0"])
# vol = vol.transpose(1, 0, 2)
# vol = vol / vol_sh_x
# vol = nd.zoom(vol, vol_sh_x / vol.shape[0], order=1)
# vol = jax.device_put(vol)

# general
vol = zeros((vol_sh_x, vol_sh_x, vol_sh_x), dtype="float32")
vol[
    nslices//4 : nslices//4 + nslices//2,
    vol_sh_x//4 : vol_sh_x//4 + vol_sh_x//2,
    vol_sh_x//4 : vol_sh_x//4 + vol_sh_x//2
] = 1


z_source = vol_sh_x * 2
z_det = z_source * 10
M = (z_det + z_source) / abs(z_source)

n_angles = vol_sh_x * 2
angles = linspace(0, 2 * pi, n_angles, False)

vx_size = 1.
ncols = vol_sh_x
px_width = M
nrows = nslices
px_height = M
vol_sh_y = nslices

proj = P_fp.get_fp_pmap(
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
imshow(vol[nslices//2], vmin=0, vmax=vol.max())
axhline(vol_sh_x//2, color="blue")
title("Ground Truth")

subplot(132)
imshow(fbp[nslices//2], vmin=0, vmax=vol.max())
axhline(vol_sh_x//2, color="orange")
title("Cone FBP")

subplot(133)
plot(vol[nslices//2, vol_sh_x//2], label="Ground Truth")
plot(fbp[nslices//2, vol_sh_x//2], label="FBP")
legend()
title("Comparison")
tight_layout()
