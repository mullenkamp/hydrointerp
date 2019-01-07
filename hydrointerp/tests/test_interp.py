# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:22:22 2018

@author: michaelek
"""
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp.interp2d import interp_to_grid, interp_to_points
from hydrointerp.io.raster import save_geotiff
from hydrointerp.util import grp_ts_agg
from nzmetservice import select_bounds, to_df

pd.options.display.max_columns = 10

#####################################
### Parameters

min_lat = -47
max_lat = -40
min_lon = 166
max_lon = 175
nc1 = r'N:\met_service\forecasts\wrf_hourly_precip_nz4kmN-NCEP_2018090918.nc'
nc2 = r'N:\nasa\precip\gpm_3IMERGHHL\gpm_3IMERGHHL_v07_20190101-20190106.nc4'

from_crs = 4326
to_crs = 2193
grid_res = 1000
interp_fun = 'cubic'

time_col = 'time'
x_col = 'longitude'
y_col = 'latitude'
data_col = 'precip_rate'
interp_fun = 'cubic'
digits = 2

ts_resample_code = '4H'

point_shp = r'N:\met_service\point_test1.shp'
point_site_col = 'site_id'
site_test = 5


####################################
### Import

ms1 = select_bounds(nc1, min_lat, max_lat, min_lon, max_lon)

ms_df = to_df(ms1, True).dropna().reset_index()
ms_df1 = ms_df[(ms_df.time <= '2018-09-24 12:00') & (ms_df.time >= '2018-09-24')]

### Resample in time
ms_df2 = grp_ts_agg(ms_df1, ['longitude', 'latitude'], 'time', ts_resample_code, 'left', 'right')[data_col].sum().reset_index()


### Interp

new_df = interp_to_grid(ms_df2, 'time', 'longitude', 'latitude', 'precip_rate', grid_res, from_crs, to_crs, interp_fun)

new_ds = interp_to_grid(ms_df2, 'time', 'longitude', 'latitude', 'precip_rate', grid_res, from_crs, to_crs, interp_fun, output='xarray')

new_points = interp_to_points(ms_df2, 'time', 'longitude', 'latitude', 'precip_rate', point_shp, point_site_col, from_crs)


###################################
### Testing

time_name = 'time'
x_name = 'lon'
y_name = 'lat'
data_name = 'precipitationCal'

grid1 = xr.open_dataset(nc2)

grid = grid1.copy()
grid1.close()

df1 = grid.to_dataframe().reset_index()

x = da1[x_name].values
y = da1[y_name].values

xy_orig_pts = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

da2 = da1.transpose(x_name, y_name, time_name)
ar1 = da2.values
ar1[:2] = np.nan

np.nan_to_num(ar1, False)

points_x, points_y = np.broadcast_arrays(xinterp.reshape(-1,1), yinterp)
coord = np.vstack((points_x.flatten()*(len(xgrid)-1), points_y.flatten()*(len(ygrid)-1)))


grouped = df1.groupby([x_name, y_name, time_name])[data_name].mean()

grouped = df1.set_index([x_name, y_name, time_name])[data_name]

# create an empty array of NaN of the right dimensions
shape = tuple(len(i) for i in grouped.index.levels)
arr = np.full(shape, np.nan)

# fill it using Numpy's advanced indexing
arr[tuple(grouped.index.labels)] = grouped.values.flat


df = pd.DataFrame({'x': [1, 2, 1, 3, 1, 2, 3, 1, 2],
                   'y': [1, 1, 2, 2, 1, 1, 1, 2, 2],
                   'z': [1, 1, 1, 1, 2, 2, 2, 2, 2],
                   'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

grouped = df.groupby(['z', 'y', 'x'])['value'].mean()







































