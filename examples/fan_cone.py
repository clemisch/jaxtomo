from pylab import *

from jaxtomo import parallel_fp
from jaxtomo import fan_fp
from jaxtomo import cone_fp
from jaxtomo import util

util.set_preallocation(True)
util.set_cuda_device(0, 1)

vol_sh_x = 128
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
px_height = M
vol_sh_y = nslices

print("PARALLEL")
proj_par = parallel_fp.get_fp_pmap(
    vol, angles, 
    vx_size, 
    ncols, 1., 
    nrows, 1.,
)

print("FAN")
proj_fan = fan_fp.get_fp_pmap(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, 1.,
    z_source, z_det
)

print("CONE")
proj_cone = cone_fp.get_fp_pmap(
    vol, angles, 
    vx_size, 
    ncols, px_width, 
    nrows, px_height,
    z_source, z_det
)





fig, ax = subplots(2, 3, figsize=(14, 8))

ax[0, 0].imshow(proj_par[:vol_sh_x, nrows//4])
ax[0, 1].imshow(proj_fan[:vol_sh_x, nrows//4])
ax[0, 2].imshow(proj_cone[:vol_sh_x, nrows//4])
ax[1, 0].imshow(proj_par[n_angles//16])
ax[1, 1].imshow(proj_fan[n_angles//16])
ax[1, 2].imshow(proj_cone[n_angles//16])

ax[0, 0].set_title("Parallel beam")
ax[0, 1].set_title("Fan beam")
ax[0, 2].set_title("Cone beam")
ax[0, 0].set_ylabel("Sinogram")
ax[1, 0].set_ylabel("Projection")

for a in ax[0]:
    for spine in a.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(3)
for a in ax[1]:
    for spine in a.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(3)

ax[0, 0].axhline(n_angles//16, color="red")
ax[0, 1].axhline(n_angles//16, color="red")
ax[0, 2].axhline(n_angles//16, color="red")
ax[1, 0].axhline(nrows//4, color="blue")
ax[1, 1].axhline(nrows//4, color="blue")
ax[1, 2].axhline(nrows//4, color="blue")

tight_layout()
show()
