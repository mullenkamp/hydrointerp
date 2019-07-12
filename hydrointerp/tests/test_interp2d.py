# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:08:28 2019

@author: MichaelEK
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp import interp2d, Interp
#from hydrointerp.io.raster import save_geotiff

#########################################
### Parameters

py_dir = os.path.realpath(os.path.dirname(__file__))

nc1 = 'nasa_gpm_2017-07-20.nc'
#tif0 = 'nasa_gpm_2017-07-20.tif'


time_name = 'time'
x_name = 'lon'
y_name = 'lat'
data_name = 'precipitationCal'
from_crs = 4326
to_crs = 2193
grid_res = 1000
bbox=None
order=1
extrapolation='constant'
cval=np.nan
digits = 2
min_lat = -48
max_lat = -41
min_lon = 170
max_lon = 178
min_val=0
method='linear'
points = [[1625350, 5504853], [1687819, 5564772], [1632149, 5150014]]
values = [94.46, 45.57, 23.82]
points_df = pd.DataFrame(points, columns=['x', 'y'])

########################################
### Read data

ds = xr.open_dataset(os.path.join(py_dir, nc1))

### Aggregate data to day
da4 = ds[data_name].resample(time='D', closed='right', label='left').sum('time')

### Close the file (by removing the object)
ds.close()

### Save as tif
df5 = da4.to_dataframe().reset_index()

#def test_save_geotiff():
#    save_geotiff(df5, from_crs, 'precipitationCal', 'lon', 'lat', export_path=tif0)
#
#    assert 0 == 0

########################################
### Run interpolations

def test_grid_to_grid1():
    interp1 = interp2d.grid_to_grid(da4.to_dataset(), time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, order, extrapolation, min_val=min_val)
    assert 33000000 > interp1.precipitationCal.sum() > 32800000


def test_points_to_grid1():
    interp2 = interp2d.points_to_grid(df5, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, method, extrapolation, min_val=min_val)
    assert 33000000 > interp2.precipitationCal.sum() > 32800000


def test_grid_to_points1():
    interp3 = interp2d.grid_to_points(da4.to_dataset(), time_name, x_name, y_name, data_name, points_df, from_crs, to_crs, order, min_val=min_val)
    assert 24 > interp3.precipitationCal.sum() > 22


def test_points_to_points1():
    interp4 = interp2d.points_to_points(df5, time_name, x_name, y_name, data_name, points_df, from_crs, to_crs, method, min_val=min_val)
    assert 24 > interp4.precipitationCal.sum() > 22


def test_grid_to_grid2():
    interpc1 = Interp(da4.to_dataset(), time_name, x_name, y_name, data_name, from_crs)
    interp1 = interpc1.grid_to_grid(grid_res, to_crs, bbox, order, extrapolation, min_val=min_val)
    assert 33000000 > interp1.precip.sum() > 32800000


def test_points_to_grid2():
    interpc2 = Interp(point_data=df5, point_time_name=time_name, point_x_name=x_name, point_y_name=y_name, point_data_name=data_name, point_crs=from_crs)
    interp2 = interpc2.points_to_grid(grid_res, to_crs, bbox, method, extrapolation, min_val=min_val)
    assert 33000000 > interp2.precip.sum() > 32800000


def test_grid_to_points2():
    interpc1 = Interp(da4.to_dataset(), time_name, x_name, y_name, data_name, from_crs)
    interp3 = interpc1.grid_to_points(points_df, to_crs, order, min_val=min_val)
    assert 24 > interp3.precip.sum() > 22


def test_points_to_points2():
    interpc2 = Interp(point_data=df5, point_time_name=time_name, point_x_name=x_name, point_y_name=y_name, point_data_name=data_name, point_crs=from_crs)
    interp4 = interpc2.points_to_points(points_df, to_crs, method, min_val=min_val)
    assert 24 > interp4.precip.sum() > 22






