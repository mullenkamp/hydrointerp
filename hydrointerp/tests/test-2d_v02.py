# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

from scipy.interpolate import (
    LinearNDInterpolator, RectBivariateSpline,
    RegularGridInterpolator, CloughTocher2DInterpolator)
from scipy.ndimage import map_coordinates

################################################
### Parameters

in_n = 50
out_n = 100

############################################

# Tweak how images are plotted with imshow
mpl.rcParams['image.interpolation'] = 'none' # no interpolation
mpl.rcParams['image.origin'] = 'lower' # origin at lower left corner
mpl.rcParams['image.cmap'] = 'RdBu_r'


def f_2d(x,y):
    '''a function with 2D input to interpolate on [0,1]'''
    return np.exp(-x)*np.cos(x*2*pi)*np.sin(y*2*pi)

# quick check :
f_2d(0,0.25)

def f_3d(x,y,z):
    '''a function with 3D input to interpolate on [0,1]'''
    return np.sin(x*2*pi)*np.sin(y*2*pi)*np.sin(z*2*pi)

# quick check :
f_3d(0.25,0.25,0.25)

Ndata = in_n
xgrid = np.linspace(0,1, Ndata)
ygrid = np.linspace(0,1, Ndata+1) # use a slighly different size to check differences
zgrid = np.linspace(0,1, Ndata+2)

f_2d_grid = f_2d(xgrid.reshape(-1,1), ygrid)

#plt.imshow(f_2d_grid)
#plt.title(u'image of a 2D function ({}² pts)'.format(Ndata));

#f_2d_grid.shape

# Define the grid to interpolate on :
Ninterp = out_n
xinterp = np.linspace(0,1, Ninterp)
yinterp = np.linspace(0,1, Ninterp+1) # use a slighly different size to check differences
zinterp = np.linspace(0,1, 5) # small dimension to avoid size explosion
X2, Y2 = np.meshgrid(xinterp, yinterp)
xy_int = np.column_stack((X2.flatten(), Y2.flatten()))


#############################
### CloughTocher2DInterpolator

# Build data for the interpolator
pionts = np.dstack(np.meshgrid(xgrid.reshape(-1,1), ygrid)).reshape(-1, 2)
values = f_2d_grid.flatten()

# Build
%timeit f_2d_interp = CloughTocher2DInterpolator(points, values)

f_2d_interp = CloughTocher2DInterpolator(points, values)

# Evaluate
%timeit f_2d_interp(xy_int)

cti  = f_2d_interp(xy_int)
np.min(cti)
np.max(cti)

# Display
plt.imshow(f_2d_interp(xinterp.reshape(-1,1), yinterp))
plt.title(u'interpolation of a 2D function ({}² pts)'.format(Ninterp));


##################################
### RectBivariateSpline

%timeit f_2d_interp = RectBivariateSpline(xgrid, ygrid, f_2d_grid, s=0)

f_2d_interp = RectBivariateSpline(xgrid, ygrid, f_2d_grid, s=0)

# Evaluate points output
%timeit f_2d_interp(X2, Y2, grid=False)

# Evaluate grid output
%timeit f_2d_interp(xinterp, yinterp)

rbs  = f_2d_interp(X2, Y2, grid=False)
np.min(rbs)
np.max(rbs)

# Display
plt.imshow(f_2d_interp(X2, Y2, grid=False))
plt.title(u'interpolation of a 2D function ({}² pts)'.format(Ninterp));

#################################
### map_coordinates

# Prepare the coordinates to evaluate the array on :
points_x, points_y = np.broadcast_arrays(xinterp.reshape(-1,1), yinterp)
coord = np.vstack((points_x.flatten()*(len(xgrid)-1), # a weird formula !
                   points_y.flatten()*(len(ygrid)-1)))

%timeit f_2d_interp = map_coordinates(f_2d_grid, coord, order=1)

f_2d_interp = map_coordinates(f_2d_grid, coord, order=3)

# Reshape
f_2d_interp = f_2d_interp.reshape(len(xinterp), len(yinterp))

np.min(f_2d_interp)
np.max(f_2d_interp)

# Display
plt.imshow(f_2d_interp)
plt.title(u'interpolation of a 2D function ({}² pts)'.format(Ninterp));

scipy.interpolate.interpn



a = np.arange(6).reshape(2,3)
it = np.nditer(a, flags=['multi_index'])

while not it.finished:
    print(it[0], it.multi_index),
    it.iternext()


for b in a:
    print(b)






































