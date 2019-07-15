# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:08:28 2019

@author: MichaelEK
"""
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp import Interp, datasets
#from hydrointerp.io.raster import save_geotiff

#########################################
### Parameters

#py_dir = os.path.realpath(os.path.dirname(__file__))

nc1 = 'nasa_gpm_2015-06-18'
csv1 = 'ecan_data_2015-06-18'
#tif0 = 'nasa_gpm_2017-07-20.tif'


grid_time_name = 'time'
grid_x_name = 'lon'
grid_y_name = 'lat'
grid_data_name = 'precipitationCal'
grid_crs = 4326

point_time_name = 'date'
point_x_name = 'NZTMX'
point_y_name = 'NZTMY'
point_data_name = 'precip'
point_crs = 2193

to_crs = 2193
grid_res = 10000
bbox=None
order=2
extrapolation='constant'
cval=np.nan
digits = 2
min_lat = -48
max_lat = -41
min_lon = 170
max_lon = 178
min_val=0
method='linear'

########################################
### Read data

ds = xr.open_dataset(datasets.get_path(nc1))
df1 = pd.read_csv(datasets.get_path(csv1), parse_dates=['date'], infer_datetime_format=True)

### Assign nan toplaces wherethe quality index is below 0.4
ds2 = ds[[grid_data_name]].where(ds.precipitationQualityIndex > 0.4)

### Close the file (by removing the object)
ds.close()

### Convert to DataFrame
df5 = ds2[grid_data_name].to_dataframe().reset_index()

### Create example points
points_df = df1.loc[[6, 15, 132], [point_x_name, point_y_name]].copy()
points_df.rename(columns={point_x_name: 'x', point_y_name: 'y'}, inplace=True)

#def test_save_geotiff():
#    save_geotiff(df5, from_crs, 'precipitationCal', 'lon', 'lat', export_path=tif0)
#
#    assert 0 == 0

########################################
### Run interpolations

interpc = Interp(ds2, grid_time_name, grid_x_name, grid_y_name, grid_data_name, grid_crs, point_data=df1, point_time_name=point_time_name, point_x_name=point_x_name, point_y_name=point_y_name, point_data_name=point_data_name, point_crs=point_crs)
interpc.grid_interp_na()


def test_grid_interp_na():
    nan1 = ds2[grid_data_name].isnull().sum()
    nan2 = interpc.grid_data['precip'].isnull().sum()
    assert (nan1 > 0) & (nan2 == 0)


def test_grid_to_grid2():
    interp1 = interpc.grid_to_grid(grid_res, to_crs, bbox, order, extrapolation, min_val=min_val)
    assert 28 > interp1.precip.mean() > 27


def test_points_to_grid2():
    interp2 = interpc.points_to_grid(grid_res, to_crs, bbox, method, extrapolation, min_val=min_val)
    assert 28 > interp2.precip.mean() > 27


def test_grid_to_points2():
    interp3 = interpc.grid_to_points(points_df, to_crs, order, min_val=min_val)
    assert 149 > interp3.precip.mean() > 148


def test_points_to_points2():
    interp4 = interpc.points_to_points(points_df, to_crs, method, min_val=min_val)
    assert 34 > interp4.precip.mean() > 33


def test_adj_grid_from_points():
    interp5 = interpc.adjust_grid_from_points(grid_res, to_crs)
    assert 55 > interp5.precip.mean() > 54


def test_validate_grid_from_points():
    interp6 = interpc.validate_grid_from_points(0.08, grid_res, to_crs)
    assert len(interp6) > 8

