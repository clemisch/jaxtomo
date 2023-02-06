from pylab import *
import scipy.ndimage as nd
import jax

from jaxtomo import cone_fp as P_fp
from jaxtomo import cone_bp as P_bp

from jaxtomo import fan_fp as P_fp
from jaxtomo import fan_bp as P_bp

from jaxtomo import proj_filter
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


vol = zeros_like(vol)
vol[16:-16:, 8:-8, 8:-8] = util.roundmask(vol_sh_x - 16, vol_sh_x - 16)


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
