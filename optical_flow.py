# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from math import pi as pi
from math import floor as floor

 
from felpy.utils.vis_utils import Grids
 
from phase_retrieval.correlation import shift_image

from matplotlib import pyplot as plt
def treys_optical_flow(arr1, arr2):
    Nx, Ny = arr1.shape
    
    dkx = (2*np.pi)/(Nx)
    dky = (2*np.pi)/(Ny)
    
    kx, ky = np.meshgrid((np.arange(0, Ny) - floor(Ny / 2) - 1) * dky, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dkx)
    
    A = 1j/arr1
    B = fftshift(fft2(arr1-arr2))
    plt.imshow(ky)
    
    C = (1j*kx)/(kx**2+ky**2)
    C[np.isnan(C)] = 0

    D = (1j*ky)/(kx**2+ky**2)
    D[np.isnan(D)] = 0

    
    dx = (A*ifft2(ifftshift(B*C))).real
    dx[np.isnan(dx)] = 0
    
    dy = (A*ifft2(ifftshift(B*D))).real
    dy[np.isnan(dy)] = 0
# =============================================================================
#     
#     M = fftshift(fft2(dx+1j*dy))
#     N = 1j*kx-ky
#     d = ifft2(fftshift(M/N))
# =============================================================================
    
    return dx,dy
    
def optical_flow(Is,Ir,alpha=0,sigma=0):
    derivative = (Is - Ir * (np.mean(gaussian_filter(Is,sigma=sigma)) / np.mean(gaussian_filter(Ir,sigma=sigma))))
    Nx, Ny = derivative.shape
    # fourier transfomm of the derivative and shift low frequencies to the centre
    ftdI = fftshift(fft2(derivative))
    # calculate frequencies
    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)
    Qx, Qy = np.meshgrid((np.arange(0, Ny) - floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dqx)


    #building filters


    sigmaX = dqx / 1. * np.power(sigma,2)
    sigmaY = dqy / 1. * np.power(sigma,2)
    #sigmaX=sig_scale
    #sigmaY = sig_scale

    g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
    #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
    beta = 1 - g;

    # fourier filters
    ftfiltX = (1j* Qx / ((Qx**2 + Qy**2 + alpha))*beta)
    ftfiltX[np.isnan(ftfiltX)] = 0

    ftfiltY = (1j*Qy/ ((Qx**2 + Qy**2 + alpha))*beta)
    ftfiltY[np.isnan(ftfiltY)] = 0

    # output calculation
    dImX = 1. / Is * ifft2(ifftshift(ftfiltX * ftdI))
    dImY = 1. / Is * ifft2(ifftshift(ftfiltY * ftdI))

    return dImX.real,dImY.real

if __name__ == '__main__':
    shift_x = 1
    shift_y = -5
    
    arr1 = np.zeros([101,101])+np.random.rand(101,101)*1000
    arr1[50,50] = 1
    #arr1 = gaussian_filter(arr1, 2)
    
    arr2 = shift_image(arr1, shift_x, shift_y)
    
    arr1 = arr1[25:75, 25:75]
    arr2 = arr2[25:75, 25:75]
    
    dx,dy = treys_optical_flow(arr1, arr2)
    
    #grid = Grids(scale = 2, global_aspect = 3.5)
    #grid.create_grid(n=2, m = 2, sharey = False, sharex = False)
    #grid.pad(1)
    
# =============================================================================
#     grid.add_global_colorbar(clabel = "Correlation", vmin = np.min(dx), vmax = np.max(dx), cmap = 'jet',
#                              tick_values = [np.min(dx), np.max(dx)],
#                              tick_labels = ["$c_{min}$","$c_{max}$"], fontsize = 22)
#     
# =============================================================================
    #grid.set_fontsize(22)
    
    
    
    #ax1, ax2, ax3, ax4 = grid.axes.flatten()
    
    #ax1.imshow(arr1, cmap = 'bone')
    #ax2.imshow(arr1, cmap = 'bone')
    #ax3.imshow(dx, cmap = 'jet')
    from matplotlib import pyplot as plt
    plt.imshow(arr1)
    plt.show()
    plt.imshow(arr2)
    plt.show()
    plt.imshow(dx)
    plt.show()
    plt.imshow(dy)
    #sx, sy = cc_shift(c)
# =============================================================================
#     
#     print("Actual Horizontal Shift: {}".format(shift_x))
#     print("Recorded Horizontal Shift: {}".format(sx))
#     print("Actual Vertical Shift: {}".format(shift_y))
#     print("Recorded Vertical Shift: {}".format(sy))
# =============================================================================
