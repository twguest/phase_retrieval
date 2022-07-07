# -*- coding: utf-8 -*-
"""
see: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
for more
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from phase_retrieval.correlation import scanned_correlation

image = np.random.rand(50,50)



rows, cols = image.shape[0], image.shape[1]

src_cols = np.linspace(0, cols, 10)
src_rows = np.linspace(0, rows, 10)



src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(src[:, 0], src[:, 1], '.b')
ax.set_xlim(0,49)
ax.set_ylim(0,49)
plt.show()

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] - np.sin(np.linspace(0, np.pi, src.shape[0]))
dst_cols = src[:, 0]
 
dst = np.vstack([dst_cols, dst_rows]).T


tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0]
out_cols = cols
out = warp(image, tform, output_shape=image.shape, cval = np.mean(image))

fig, ax = plt.subplots()
ax.imshow(out-np.mean(out))
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
ax.set_xlim(0,49)
ax.set_ylim(0,49)
plt.show()

arr1 = image
arr2 = out 

### define the width of the correlation window
wx = 15
wy = 15

### define the step size
step_size = 1

scanned_correlation(arr1,arr2,wx,wy,step_size = step_size, plot = True, sdir = "./gif_tmp/")
