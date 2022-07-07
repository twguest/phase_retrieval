# -*- coding: utf-8 -*-
"""

 

__author__ = "Trey Guest"
__credits__ = ["Trey Guest"]
__license__ = "EuXFEL"
__version__ = "0.2.1"
__maintainer__ = "Trey Guest"
__email__ = "trey.guest@xfel.eu"
__status__ = "Developement"

the purpose of this script is to calculate the shift between two images,
we scan cropped regions of arr2 across the larger arr1
"""

from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from felpy.utils.vis_utils import Grids, add_colorbar
from felpy.utils.fig_combine import combine_figures
from phase_retrieval.correlation import cross_correlation, cc_shift, shift_image, get_boundaries, shift_by_correlation, scanned_correlation
from mpl_toolkits.axes_grid1 import make_axes_locatable





### fontsize 
fs= 14

### perscribed (variable shift)
shift_x = 2
shift_y = 1

### define the width of the correlation window
wx = 15
wy = 15

### define the step size
step_size = 5

### define the arrays
arr1 = np.random.rand(50,50)
arr1 = gaussian_filter(arr1, 1)-np.mean(arr1)
#arr1[int(np.random.uniform(30,50)),int(np.random.uniform(30,50))] = -100

arr2 = shift_image(arr1, shift_x, shift_y)

 
### create results array
 


scanned_correlation(arr1,arr2,wx,wy,step_size = step_size, plot = True, sdir = "./gif_tmp/")
# =============================================================================
# 
# for cx in range(boundaries[0], boundaries[1]):
#     for cy in range(boundaries[2], boundaries[3]):
# =============================================================================

 