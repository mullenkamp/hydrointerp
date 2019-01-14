# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:08:28 2019

@author: MichaelEK
"""
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp.interp2d import points_to_grid, points_to_points, grid_to_grid, grid_to_points
from hydrointerp.io.raster import save_geotiff

#########################################
### Parameters

nc1 = 'nasa_gpm_2018-01-17.nc'
tif0 = 'nasa_gpm_2018-01-17.tif'
tif1 = 'nasa_gpm_2018-01-17_test1.tif'
tif2 = 'nasa_gpm_2018-01-17_test2.tif'
tif3 = 'nasa_gpm_2018-01-17_test3.tif'
tif4 = 'nasa_gpm_2018-01-17_test4.tif'

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

ds = xr.open_dataset(nc1)

### Aggregate data to day
da4 = ds[data_name].resample(time='D', closed='right', label='left').sum('time')

### Close the file (by removing the object)
del ds

### Save as tif
df5 = da4.to_dataframe().reset_index()

save_geotiff(df5, from_crs, 'precipitationCal', 'lon', 'lat', export_path=tif0)

########################################
### Run interpolations

interp1 = grid_to_grid(da4.to_dataset(), time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, order, extrapolation, min_val=min_val)

save_geotiff(interp1.to_dataframe().reset_index(), to_crs, 'precipitationCal', 'x', 'y', export_path=tif1)

interp2 = points_to_grid(df5, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, method, extrapolation, min_val=min_val)

save_geotiff(interp2.to_dataframe().reset_index(), to_crs, 'precipitationCal', 'x', 'y', export_path=tif2)

interp3 = grid_to_points(da4.to_dataset(), time_name, x_name, y_name, data_name, points_df, from_crs, to_crs, order, min_val=min_val)

interp4 = points_to_points(df5, time_name, x_name, y_name, data_name, points_df, from_crs, to_crs, method, min_val=min_val)








