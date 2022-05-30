# -*- coding: utf-8 -*-
"""
FELpy.phase_retrieval

__author__ = "Trey Guest"
__credits__ = ["Trey Guest"]
__license__ = "EuXFEL"
__version__ = "0.2.1"
__maintainer__ = "Trey Guest"
__email__ = "trey.guest@xfel.eu"
__status__ = "Developement"
"""

import numpy as np
from scipy.signal import correlate2d
from matplotlib import pyplot as plt
from felpy.utils.vis_utils import Grids
from felpy.utils.analysis_utils import window_2D


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def cross_correlation(arr1, arr2, window = np.ones):
    """
    This is a wrapper for scipy.signal 2D, where we can apply a 2D window.
    This is a base-level function
    
    :param arr1:
    :param arr2:
    :param window:
        
    :returns: correlation function <corr2 x corr1*>*window
    """
    cc = correlate2d(arr2, arr1, mode = 'full')
    return cc*window_2D(N = cc.shape[0], M = cc.shape[1], window = window)

def cc_shift(cc):
    """ 
    determine the shift between two-images from the maximum of their correlation function
    
    :param cc: cross correlation between a pair of arrays
    
    :returns: horizontal and vertical shfits between the correlated images
    """
    sy, sx = np.where(cc == np.max(cc))

    cy, cx = cc.shape[0]//2, cc.shape[1]//2
    return sx-cx, sy-cy

if __name__ == '__main__':
    from scipy.ndimage import gaussian_filter
    
    
    shift_x = 25
    shift_y = -5
    
    arr1 = np.zeros([101,101])
    arr1[50,50] = 1
    arr1 = gaussian_filter(arr1, 10)
    
    arr2 = np.zeros([151,151])
    arr2[75,75] = 1
    arr2 = gaussian_filter(arr2, 10)
    arr2 = shift_image(arr1, shift_x, shift_y)
    
    plt.imshow(arr1)
    
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