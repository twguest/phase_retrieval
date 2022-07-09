# -*- coding: utf-8 -*-
"""
see: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
for more
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from FELpy.phase_retrieval.correlation import scanned_correlation
from scipy.ndimage import gaussian_filter

image = gaussian_filter(np.random.rand(50,50)*10, 2)
image -= np.mean(image)



rows, cols = image.shape[0], image.shape[1]

src_cols = np.linspace(0, cols, 10)
src_rows = np.linspace(0, rows, 10)



src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]
 

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] 
dst_cols = src[:, 0] - np.sin(np.linspace(0, np.pi/2, src.shape[0]))*4
 
dst = np.vstack([dst_cols, dst_rows]).T


tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0]
out_cols = cols
out = warp(image, tform, output_shape=image.shape, cval = np.mean(image))

 

arr1 = image
arr2 = out 

### define the width of the correlation window
wx = 25
wy = 25

### define the step size
step_size = 1

from felpy.utils.vis_utils import Grids
grid = Grids(global_aspect = 2.5)
grid.create_grid(n=1,m=2)
ax1, ax2 = grid.axes
 

ax1.plot(src[:, 0], src[:, 1], 'xr')
ax1.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')


r = scanned_correlation(arr1,arr2,wx,wy,step_size = step_size, plot = True, sdir = "./gif_tmp/")
