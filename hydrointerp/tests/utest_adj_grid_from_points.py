# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:08:58 2019

@author: MichaelEK
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj, transform
from datetime import datetime
from pdsql import mssql
from hydrointerp import Interp
from matplotlib import pyplot

pd.options.display.max_columns = 10


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

########################################
### Parameters

dataset_type = [15]

server = 'edwprod01'
db = 'hydro'


py_dir = os.path.realpath(os.path.dirname(__file__))

nc1 = 'nasa_gpm_2017-07-20.nc'

nc_dir = r'N:\nasa\precip\gpm_3IMERGHH'
nc2 = 'gpm_3IMERGHH_v06_20150601-20150630.nc4'
nc1 = r'N:\nasa\precip\gpm_3IMERGHH\gpm_3IMERGHH_v06_20150101-20150131.nc4'

point_time_name = 'DateTime'
point_x_name = 'NZTMX'
point_y_name = 'NZTMY'
point_data_name = 'Value'
point_crs = 2193

grid_time_name = 'time'
grid_x_name = 'lon'
grid_y_name = 'lat'
grid_data_name = 'precipitationCal'
grid_crs = 4326

grid_res=20000
to_crs=2193
bbox=None
order=3
digits=2
min_val=0

########################################
### Get data

## Nasa data
ds1 = xr.open_dataset(nc1)

grid_data = ds1[[grid_data_name]].resample(time='D', closed='right', label='left').sum('time') / 2

ds1.close()
del ds1

## Met station data
summ1 = mssql.rd_sql(server, db, 'TSDataNumericDailySumm', where_in={'DatasetTypeID': dataset_type}).drop('ModDate', axis=1)
summ1.ToDate = pd.to_datetime(summ1.ToDate)
summ1.FromDate = pd.to_datetime(summ1.FromDate)

dates1 = grid_data.time.to_index()
min_date = dates1.min()
max_date = dates1.max()

summ2 = summ1[(summ1.FromDate <= min_date) & (summ1.ToDate >= max_date)].copy()

ts_data = mssql.rd_sql(server, db, 'TSDataNumericDaily', ['ExtSiteID', 'DateTime', 'Value'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist(), 'DatasetTypeID': dataset_type}, from_date=str(min_date.date()), to_date=str(max_date.date()), date_col='DateTime')
ts_data.DateTime = pd.to_datetime(ts_data.DateTime)

site_data = mssql.rd_sql(server, db, 'ExternalSite', ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist()}).round()

point_data = pd.merge(ts_data, site_data, on='ExtSiteID').drop('ExtSiteID', axis=1)


self = Interp(grid_data, grid_time_name, grid_x_name, grid_y_name, grid_data_name, grid_crs, point_data, point_time_name, point_x_name, point_y_name, point_data_name, point_crs)

point_grid = self.points_to_grid(20000, to_crs)

new_grid = self.adjust_grid_from_points(20000, to_crs)


point_grid.precip.isel(time=0).plot()

new_grid.precip.isel(time=0).plot()

#############################3
### Other testing
### A 'precipitationQualityIndex' of > 0.4 is the ideal setting for GPM data

ds1 = xr.open_dataset(os.path.join(nc_dir, nc2))

ds2 = ds1[[grid_data_name, 'precipitationQualityIndex']].sel(time=slice('2015-06-18', '2015-06-19')).copy()

ds1.close()
del ds1

da1 = ds2[grid_data_name].resample(time='D', closed='right', label='left').sum('time') / 2
da2 = ds2['precipitationQualityIndex'].resample(time='D', closed='right', label='left').mean('time')

ds2b = xr.merge([da1, da2])

ds2b.to_netcdf(r'E:\ecan\git\hydrointerp\hydrointerp\datasets\nasa_gpm_2015-06-18.nc')

ds3 = da1.where(da2 > 0.4).to_dataset()

max_x = site_data.NZTMX.max()
min_x = site_data.NZTMX.min()
max_y = site_data.NZTMY.max()
min_y = site_data.NZTMY.min()

i1 = Interp(ds3, grid_time_name, grid_x_name, grid_y_name, grid_data_name, grid_crs)

i1.grid_interp_na()

ds4 = i1.grid_to_grid(10000, to_crs, bbox=(1300000, 1700000, 5000000, 5370000), min_val=0)

ds3.mean(['lat', 'lon']).round()

ds4.mean(['y', 'x']).round()

ds5 = ds4.sel(time=slice('2015-06-18', '2015-06-19'))

ds5.precip.plot(x='x', y='y', col='time', col_wrap=2)


ds3a = ds3.sel(time=slice('2015-06-18', '2015-06-19'))

ds3a[grid_data_name].plot(x='lon', y='lat', col='time', col_wrap=2)



plt1 = da5.plot(x='lon', y='lat', col='time', col_wrap=3)




grid_data = ds3.sel(time=slice('2015-06-18', '2015-06-19'))

## Met station data
summ1 = mssql.rd_sql(server, db, 'TSDataNumericDailySumm', where_in={'DatasetTypeID': dataset_type}).drop('ModDate', axis=1)
summ1.ToDate = pd.to_datetime(summ1.ToDate)
summ1.FromDate = pd.to_datetime(summ1.FromDate)

dates1 = grid_data.time.to_index()
min_date = dates1.min()
max_date = dates1.max()

summ2 = summ1[(summ1.FromDate <= min_date) & (summ1.ToDate >= max_date) & (summ1.ExtSiteID.astype(int) < 900000)].copy()

ts_data = mssql.rd_sql(server, db, 'TSDataNumericDaily', ['ExtSiteID', 'DateTime', 'Value'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist(), 'DatasetTypeID': dataset_type}, from_date=str(min_date.date()), to_date=str(max_date.date()), date_col='DateTime')
ts_data.DateTime = pd.to_datetime(ts_data.DateTime)

site_data = mssql.rd_sql(server, db, 'ExternalSite', ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': summ2.ExtSiteID.tolist()}).round()

point_data = pd.merge(ts_data, site_data, on='ExtSiteID').drop('ExtSiteID', axis=1)
# point_data.rename(columns={'Value': 'precip', 'DateTime': 'date'}, inplace=True)

point_data.to_csv(r'E:\ecan\git\hydrointerp\hydrointerp\datasets\ecan_data_2015-06-18.csv', index=False)


self = Interp(grid_data, grid_time_name, grid_x_name, grid_y_name, grid_data_name, grid_crs, point_data, point_time_name, point_x_name, point_y_name, point_data_name, point_crs)

self.grid_interp_na()

#point_grid = self.points_to_grid(10000, to_crs)

new_grid = self.adjust_grid_from_points(10000, to_crs)


new_grid.precip.plot(x='x', y='y', col='time', col_wrap=2)


comp_res = self.validate_grid_from_points(0.08, 10000, to_crs)
comp_res

























