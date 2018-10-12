# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:22:22 2018

@author: michaelek
"""
import pandas as pd
from hydrointerp.interp2d import interp_to_grid, interp_to_points
from hydrointerp.io.netcdf import metservice_select, metservice_to_df
from hydrointerp.io.raster import save_geotiff

pd.options.display.max_columns = 10

#####################################
### Parameters

min_lat = -47
max_lat = -40
min_lon = 166
max_lon = 175
nc1 = r'N:\met_service\forecasts\wrf_hourly_precip_nz8kmN-NCEP_2018092312.nc'

from_crs = 4326
to_crs = 2193
grid_res = 1000
interp_fun = 'cubic'

time_col = 'time'
x_col = 'longitude'
y_col = 'latitude'
data_col = 'precip_rate'
interp_fun = 'cubic'
agg_ts_fun = None
period = None
digits = 2

point_shp = r'N:\met_service\point_test1.shp'
point_site_col = 'site_id'
site_test = 5


####################################
### Import

ms1 = metservice_select(nc1, min_lat, max_lat, min_lon, max_lon)

ms_df = metservice_to_df(ms1, True).dropna().reset_index()
ms_df1 = ms_df[(ms_df.time <= '2018-09-24 12:00') & (ms_df.time >= '2018-09-24')]

### Interp/resample

new_df = interp_to_grid(ms_df1, 'time', 'longitude', 'latitude', 'precip_rate', grid_res, from_crs, to_crs, interp_fun, agg_ts_fun='sum', period='2H')

new_points = interp_to_points(ms_df1, 'time', 'longitude', 'latitude', 'precip_rate', point_shp, point_site_col, from_crs, agg_ts_fun='sum', period='2H')























































