# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:08:58 2019

@author: MichaelEK
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from hydrolm import LM
from hilltoppy import web_service as ws
from hydrointerp.interp2d import grid_to_points, points_to_grid, grid_to_grid
from pyproj import Proj, transform
from datetime import datetime

pd.options.display.max_columns = 10


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

########################################
### Parameters

ecan_base_url = 'http://testwateruse.ecan.govt.nz'
hts_name = 'AtmosRecorderEcanDaily.hts'

py_dir = os.path.realpath(os.path.dirname(__file__))

nc1 = 'tests/nasa_gpm_2017-07-20.nc'


nc1 = r'N:\nasa\precip\trmm_3B42\trmm_3B42_v07_20000101-20001231.nc4'
nasa_mtype = 'precipitation'
nasa_lat = 'lat'
nasa_lon = 'lon'
nasa_time = 'time'
nasa_crs = 4326
nz_crs = 2193

mtype = 'Precipitation'

from_date = '2000-01-01'
to_date = '2000-12-31'

min_n_days = 5 * 365
min_missing_days = 14

########################################
### Get data

## Nasa data
ds1 = xr.open_dataset(nc1)

da1 = ds1[nasa_mtype].resample(time='D', closed='right', label='left').sum('time') * 3

del ds1

## Met stations
sites = ws.site_list(ecan_base_url, hts_name, True)
sites.rename(columns={'SiteName': 'Site'}, inplace=True)

mtypes = []
for s in sites['Site'].values:
    mtypes.append(ws.measurement_list(ecan_base_url, hts_name, s, mtype))

mtypes_df = pd.concat(mtypes).reset_index().drop('DataType', axis=1)

mtypes_df['n_days'] = (mtypes_df['To'] - mtypes_df['From']).dt.days

mtypes_df1 = mtypes_df[mtypes_df['n_days'] >= min_n_days].copy()

sites1 = pd.merge(sites, mtypes_df1, on='Site')

ts_data = []
for index, row in sites1.iterrows():
    ts_data.append(ws.get_data(ecan_base_url, hts_name, row['Site'], row['Measurement']))

ts_data_df = pd.concat(ts_data)
ts_data_df.index = ts_data_df.index.droplevel('Measurement')

ts_data_df2 = ts_data_df['Value'].unstack(0)


#################################################
### Process station data

period_data1 = ts_data_df2[from_date:to_date].copy()
missing_days = period_data1.isnull().sum()
good_ones = missing_days[missing_days <= min_missing_days].reset_index()['Site']

good_sites1 = period_data1.loc[:, period_data1.columns.isin(good_ones)].interpolate()

period_data1.loc[:, period_data1.columns.isin(good_ones)] = good_sites1

ts_data1 = period_data1.stack().reorder_levels([1, 0]).sort_index()
ts_data1.name = 'precip'

ts_data2 = pd.merge(sites, ts_data1.reset_index(), on='Site').drop('Site', axis=1).rename(columns={'Easting': 'x', 'Northing': 'y', 'DateTime': 'time'})

##############################################
### Interpolations

points1 = sites1.rename(columns={'Easting': 'x', 'Northing': 'y'})[['x', 'y']].copy()

nasa_df1 = grid_to_points(da1.to_dataset(), nasa_time, nasa_lon, nasa_lat, nasa_mtype, points1, nasa_crs, nz_crs, order=1, min_val=0)

nasa_df2 = nasa_df1.dropna().reset_index().copy()
nasa_df2['x'] = nasa_df2['x'].round().astype(int)
nasa_df2['y'] = nasa_df2['y'].round().astype(int)

both1 = pd.merge(ts_data2, nasa_df2, on=['x', 'y', 'time'])

both1['diff'] = both1['precip'] - both1[nasa_mtype]

points2 = both1[['x', 'y']].drop_duplicates()

from_crs1 = Proj(convert_crs(nz_crs, pass_str=True), preserve_units=True)
to_crs1 = Proj(convert_crs(nasa_crs, pass_str=True), preserve_units=True)
points_wgs = np.array([transform(from_crs1, to_crs1, x, y) for x, y in points2.values]).round(3)
min_p = points_wgs.min(0)
max_p = points_wgs.max(0)

min_lon = find_nearest(da1.lon, min_p[0])
max_lon = find_nearest(da1.lon, max_p[0])
min_lat = find_nearest(da1.lat, min_p[1])
max_lat = find_nearest(da1.lat, max_p[1])

dxy = np.median(np.diff(da1[nasa_lon]))


diff_grid = points_to_grid(both1, 'time', 'x', 'y', 'diff', dxy, nz_crs, nasa_crs, bbox=(min_lon, max_lon, min_lat, max_lat))
diff_grid = diff_grid.rename({'x': nasa_lon, 'y': nasa_lat})

da2 = da1.sel({nasa_lon: da1[nasa_lon].isin(diff_grid[nasa_lon]), nasa_lat: da1[nasa_lat].isin(diff_grid[nasa_lat])})

da3 = da1 + diff_grid['diff']
da3.name = 'precipitation_corr'

ar3 = da3.values
ar3[ar3 < 0] = 0

da3.data = ar3

da3.isel(time=2).plot(x='lon', y='lat')

ds4 = grid_to_grid(da1.to_dataset(), nasa_time, nasa_lon, nasa_lat, nasa_mtype, dxy, nasa_crs, bbox=(min_lon, max_lon, min_lat, max_lat), order=1, min_val=0)

ds4[nasa_mtype].isel(time=2).plot(x='x', y='y')

site_grid = points_to_grid(both1, 'time', 'x', 'y', 'precip', dxy, nz_crs, nasa_crs, bbox=(min_lon, max_lon, min_lat, max_lat))

site_grid['precip'].isel(time=2).plot(x='x', y='y')





