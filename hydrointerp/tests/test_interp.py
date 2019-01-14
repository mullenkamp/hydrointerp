# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:22:22 2018

@author: michaelek
"""
from time import time
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp.io.raster import save_geotiff
from hydrointerp.util import grp_ts_agg
from nzmetservice import select_bounds, to_df
from scipy import ndimage

pd.options.display.max_columns = 10

#####################################
### Parameters

min_lat = -47
max_lat = -40
min_lon = 166
max_lon = 175
nc1 = r'N:\met_service\forecasts\wrf_hourly_precip_nz4kmN-NCEP_2018090918.nc'
nc2 = r'N:\nasa\precip\gpm_3IMERGHHL\gpm_3IMERGHHL_v07_20190101-20190108.nc4'


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

point_path = r'N:\met_service\point_test1.shp'
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

nc2 = r'N:\nasa\precip\gpm_3IMERGHHL_v07_20190101-20190106.nc4'
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
point_data = r'N:\met_service\point_test1.shp'

grid1 = xr.open_dataset(nc2)

grid = grid1.where((grid1[y_name] >= min_lat) & (grid1[y_name] <= max_lat) & (grid1[x_name] >= min_lon) & (grid1[x_name] <= max_lon), drop=True).copy()
grid1.close()

grid2d = grid.isel(time=280)[data_name]

grid2d.plot.pcolormesh(x='lon', y='lat')

start1 = time()
interp1 = grid_to_grid(grid, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, order, extrapolation, cval, digits, min_val)
end1 = time()
end1 - start1

output2d = interp1.isel(time=280)[data_name]

output2d.plot.pcolormesh(x='x', y='y')

df = grid.to_dataframe().reset_index()

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


a = np.arange(12.).reshape((4, 3))




dx, dy = 0.4, 0.4
xmax, ymax = 2, 4
x = np.arange(-xmax, xmax, dx)
y = np.arange(-ymax, ymax, dy)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(2*X)**2 - (Y/2)**2)


interp2b = grid_to_points(grid, time_name, x_name, y_name, data_name, point_data, from_crs, to_crs, order, extrapolation, fill_val, digits, min_val)

interp2

d2 = df[df.time == pd.Timestamp('2019-01-06T20:29:59.999000')].copy()

interp2c = points_to_grid(d2, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs, bbox, method, fill_val, digits, min_val)

output2d = interp2c.isel(time=0)[data_name]

output2d.plot.pcolormesh(x='x', y='y')




grid1 = xr.open_dataset(nc1)
grid = grid1.copy()
grid1.close()


nc2 = r'N:\nasa\precip\gpm_3IMERGHH\gpm_3IMERGHH_v07_20170101-20171231.nc4'
nc2 = r'N:\nasa\precip\gpm_3IMERGHH\gpm_3IMERGHH_v07_20180101-20180701.nc4'

grid1 = xr.open_dataset(nc2)
da = grid1[data_name].copy()
grid1.close()

da1 = da * 0.5

da2 = da1.resample(time='D', closed='right', label='left').sum('time')

max_day = da2.groupby('time').sum().to_dataframe().sort_values('precipitationCal')

da3 = da2.loc['2018-01-15':'2018-01-22']

#grid2d.plot.pcolormesh(x='lon', y='lat')

plt1 = da3.plot(x='lon', y='lat', col='time', col_wrap=4)

save1 = r'E:\ecan\git\hydrointerp\hydrointerp\datasets\nasa_gpm_2017-07-20.nc'
save2 = r'E:\ecan\git\hydrointerp\hydrointerp\datasets\nasa_gpm_2017-07-20.tif'

da4 = da1.loc['2017-07-20':'2017-07-21 00:00']
da5 = da4.resample(time='H', closed='right', label='right').sum('time')
da5.attrs = grid1[data_name].attrs

da5.to_dataset().to_netcdf(save1)

#da5 = xr.open_dataset(save1)

df5 = da5.resample(time='D', closed='right', label='left').sum('time').to_dataframe().reset_index()

save_geotiff(df5, 4326, 'precipitationCal', 'lon', 'lat', export_path=save2)


a = np.arange(50, step=2).reshape((5,5))
a[2,2] = 2

ndimage.gaussian_filter(a, sigma=1, order=0)
ndimage.gaussian_gradient_magnitude(a, sigma=1)

ndimage.median_filter(a, size=3)
ndimage.percentile_filter(a, percentile=100, size=2)
ndimage.uniform_filter(a, size=3)



date0 = '2018-01-17'

date1 = pd.Timestamp(date0)
date2 = date1 + pd.DateOffset(days=1)

da4 = da1.loc[date1:date2]
da5 = da4.resample(time='D', closed='right', label='left').sum('time')
da6 = da5.copy()
da6['time'] = da6['time'].to_series() + pd.DateOffset(hours=1)

#da6.data = ndimage.percentile_filter(da5, percentile=100, size=2)
da6.data = ndimage.median_filter(da5, size=2)

da7 = da5.combine_first(da6)

plt1 = da7.plot(x='lon', y='lat', col='time', col_wrap=2)

save2 = r'E:\ecan\git\hydrointerp\hydrointerp\datasets\nasa_gpm_2018-01-18.tif'
save_geotiff(da6.to_dataframe().reset_index(), 4326, 'precipitationCal', 'lon', 'lat', export_path=save2)


from nasadap import Nasa

product = '3IMERGDF'
product = '3IMERGHH'
#product2c = '3IMERGHHE'
product = '3B42'
username = 'Dryden'
password = 'NasaData4me'
from_date = '2018-04-15'
to_date = '2018-04-17'
mission = 'gpm'
mission = 'trmm'
dataset_type = ['precipitationCal', 'randomError', 'precipitationQualityIndex']
dataset_type = ['precipitation']
min_lat=-49
max_lat=-33
min_lon=165
max_lon=180
cache_dir = r'\\fs02\GroundWaterMetData$\nasa\cache\nz'

ge = Nasa(username, password, mission, cache_dir)
ds2 = ge.get_data(product, dataset_type, from_date, to_date, min_lat, max_lat, min_lon, max_lon, 60)
ds3 = ds2.copy().load()

ds3['time'] = ds2.time.to_index() + pd.DateOffset(hours=12)
del ds2

da5 = ds3['precipitationCal'].resample(time='D', closed='right', label='left').sum('time').isel(time=slice(1, 4))
ea1 = ds3['randomError'].resample(time='D', closed='right', label='left').mean('time').isel(time=slice(1, 4))
qa1 = ds3['precipitationQualityIndex'].resample(time='D', closed='right', label='left').mean('time').isel(time=slice(1, 4))

ea2 = ea1/da5

plt1 = da5.plot(x='lon', y='lat', col='time', col_wrap=3)

da6 = da5.where(qa1 > 0.4).fillna(400)
da6.name = 'precip'

plt2 = da6.plot(x='lon', y='lat', col='time', col_wrap=3)


bad_spot1 = [168.75, -44.45]

day1 = '2018-01-17'
qa1.sel(time='2018-01-17', lat=-44.45, lon=168.75)
qa1.loc['2018-01-17'].sel(lon=168.75)

ea1.loc['2018-01-17'].sel(lon=168.75)

(qa1 < 0.26).sum(['lat', 'lon'])

qa2 = da5.where(qa1 > 0.26)
qa2.name = 'precip'

df1 = qa2.to_dataframe().reset_index()

### A 'precipitationQualityIndex' of > 0.4 is the ideal setting for GPM data


da5 = ds3['precipitation'].resample(time='D', closed='right', label='left').sum('time').isel(time=slice(1, 4))

plt1 = da5.plot(x='lon', y='lat', col='time', col_wrap=3)



da5 = ds3['precipitationCal'].resample(time='D', closed='right', label='left').sum('time').isel(time=slice(1, 4))
qa1 = ds3['precipitationQualityIndex'].resample(time='D', closed='right', label='left').mean('time').isel(time=slice(1, 4))

da6 = da5.where(qa1 > 0.4)
da6.name = 'precip'

ds7 = grid_interp_na(da6.to_dataset(), time_name, x_name, y_name, 'precip', min_val=0)

#da7 = ds7['precipitationCal'].resample(time='D', closed='right', label='left').sum('time').isel(time=slice(1, 4))

plt1 = da5.plot(x='lon', y='lat', col='time', col_wrap=3)
plt2 = ds7['precip'].plot(x='lon', y='lat', col='time', col_wrap=3)
























