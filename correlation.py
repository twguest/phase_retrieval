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
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from felpy.utils.vis_utils import Grids, add_colorbar
from felpy.utils.fig_combine import combine_figures
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from felpy.analysis.scalar.centroid import get_com


def get_boundaries(shape,wx,wy,ss = 1):
    """
    define the boundaries over which the correlation analysis can take place
    
    :param shape: shape tuple of array of analysis
    :param wx: correlation window width
    :param wy: correlation window height
    
    :returns valid: list of upper and lower limits for valid correlation [xmin,xmax,ymin,ymax]
    """
    return[wx//2,shape[0]-wx//2-(ss-1),wy//2,shape[1]-wy//2-(ss-1)]


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

def cross_correlation(arr1, arr2):
    """
    This is a wrapper for scipy.signal 2D, where we can apply a 2D window.
    This is a base-level function
    
    :param arr1:
    :param arr2:
    :param window:
        
    :returns: correlation function <corr2 x corr1*>*window
    """
    cc = correlate2d(arr2, arr1, mode = 'full')
    return cc#*window(N = cc.shape[0], M = cc.shape[1], window = window)

def cc_shift(cc):
    """ 
    determine the shift between two-images from the maximum of their correlation function
    
    :param cc: cross correlation between a pair of arrays
    
    :returns: horizontal and vertical shfits between the correlated images
    """
    sy, sx = np.where(cc == np.max(cc))

    cy, cx = cc.shape[0]//2, cc.shape[1]//2
    return cx-sx,cy-sy


 

def shift_by_correlation(arr1, arr2, cx, cy, wx, wy, step_size = 1, plot = False, ex = None):
        
        ### crop arrays 
        arr2_c = arr2[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2]
        arr1_c = arr1[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2]
        
        ### determine the cross-correlation of the cropped arrays
        c = cross_correlation(arr2_c,arr1_c)
        
        ### calculate shift from the cross-correlation
        sx, sy = cc_shift(c)
        
        glob = None 
        
        if plot: 
            
            if ex is not None:
                ex1, ex2, ex3 = ex
            else:
                ex1 = ex2 = ex3 = ex
        
            fs = 16 ### set fontsize

            ### roi figure
            
            glob = Grids(scale = 2, global_aspect = 3)
            glob.create_mosaic(mosaic = [['a','b','c']], share_x = False, share_y = False, width_ratios=[2,2,2])
            
             
            axes = glob.axes
            
            im1 = axes['a'].imshow(arr1, cmap = 'bone', extent = ex1, vmin = np.min(arr1), vmax = np.max(arr1))
            rect = patches.Rectangle((cy-wy//2, cx-wx//2), width = wx, height = wy, linewidth=1, edgecolor='r', facecolor='none')
            axes['a'].add_patch(rect)
            
            cb = add_colorbar(im1, glob.axes['a'], glob.fig,
                         orientation = 'vertical',
                         clabel = "Intensity",
                         vmin = np.min(arr1),
                         vmax = np.max(arr1),
                         cmap = 'bone',
                         pad = 0.05,
                         fontsize = fs,
                         labelpad = -10)
            
            cb.set_ticks([np.min(arr1),np.max(arr1)])
            cb.set_ticklabels(['$I_{min}$','$I_{max}$'])
             
            import matplotlib.colors
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "green"])            
            
            im2 = axes['b'].imshow(arr1_c, cmap = 'bone', extent = ex2,alpha = 1)
 
            cb = add_colorbar(im2, glob.axes['b'], glob.fig,
                         orientation = 'vertical',
                         clabel = "Intensity",
                         vmin = np.min(arr1_c),
                         vmax = np.max(arr1_c),
                         cmap = 'bone',
                         pad = 0.05,
                         fontsize = fs,
                         labelpad = -10)
            
            cb.set_ticks([np.min(arr1_c),np.max(arr1_c)])
            cb.set_ticklabels(['$I_{min}$','$I_{max}$'])
            
 
             
              
             
            im3 = glob.axes['c'].imshow(c, cmap = 'jet', extent = ex3)
            
            cb = add_colorbar(im3, glob.axes['c'], glob.fig,
                         orientation = 'vertical',
                         clabel = "Covariance",
                         vmin = 0,
                         vmax = np.max(c),
                         cmap = 'jet',
                         pad = 0.05,
                         fontsize = fs,
                         labelpad = -10)
            
            cb.set_ticks([np.min(c),np.max(c)])
            cb.set_ticklabels(['$C_{min}$','$C_{max}$'])
            
            glob.set_fontsize(fs)
            
            glob.pad(1)
            glob.annotate(fontsize = fs)
            plt.show()
                
      
        return [sx, sy], glob
    
def scanned_correlation(arr1, arr2, wx, wy, step_size = 1, plot = False, sdir = None, dpi = 70):
    boundaries = get_boundaries(arr1.shape, wx, wy, ss = step_size)   
    results = np.zeros([boundaries[1], boundaries[3],2])
    itr = 0
    
    for cx in tqdm(range(boundaries[0], boundaries[1], step_size)):
        for cy in range(boundaries[2], boundaries[3], step_size):
    
            [sx, sy], grid = shift_by_correlation(arr1, arr2, cx, cy, wx, wy, step_size = step_size, plot = plot, ex = None)
            

            if plot == True and sdir is not None:
                grid.fig.savefig(sdir + "cplot_{:04}.png".format(itr), dpi = dpi)
                
            itr += 1
            print(sx,sy)
            results[cx, cy, 0] = sx
            results[cx, cy, 1] = sy
            
            del grid 
    
    return results
    
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
    
    c = cross_correlation(arr1, arr2)
    
    
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
    
    cc = scanned_correlation(arr1, arr2, w = 5)
    
    print("Actual Horizontal Shift: {}".format(shift_x))
    print("Recorded Horizontal Shift: {}".format(sx))
    print("Actual Vertical Shift: {}".format(shift_y))
    print("Recorded Vertical Shift: {}".format(sy))