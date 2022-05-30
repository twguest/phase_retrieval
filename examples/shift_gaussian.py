# -*- coding: utf-8 -*-

from scipy.ndimage import gaussian_filter
import numpy as np
 
from felpy.utils.vis_utils import Grids
 
from phase_retrieval.correlation import cross_correlation, cc_shift, shift_image


shift_x = 25
shift_y = -5

arr1 = np.zeros([101,101])
arr1[50,50] = 1
arr1 = gaussian_filter(arr1, 10)

arr2 = np.zeros([151,151])
arr2[75,75] = 1
arr2 = gaussian_filter(arr2, 10)
arr2 = shift_image(arr1, shift_x, shift_y)

c = cross_correlation(arr1, arr2, window = np.ones)


grid = Grids(scale = 2, global_aspect = 3.5)
grid.create_grid(n=1, m = 3, sharey = False, sharex = False)
grid.pad(1)
grid.add_global_colorbar(clabel = "Correlation", vmin = np.min(c), vmax = np.max(c), cmap = 'jet',
                         tick_values = [np.min(c), np.max(c)],
                         tick_labels = ["$c_{min}$","$c_{max}$"], fontsize = 22)

grid.set_fontsize(22)



ax1, ax2, ax3 = grid.axes
ax1.imshow(arr1, cmap = 'bone')
ax2.imshow(arr1, cmap = 'bone')
ax3.imshow(c, cmap = 'jet')

sx, sy = cc_shift(c)

print("Actual Horizontal Shift: {}".format(shift_x))
print("Recorded Horizontal Shift: {}".format(sx))
print("Actual Vertical Shift: {}".format(shift_y))
print("Recorded Vertical Shift: {}".format(sy))