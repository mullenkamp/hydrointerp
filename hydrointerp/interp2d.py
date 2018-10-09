# -*- coding: utf-8 -*-
"""
Raster and spatial interpolation functions.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata, Rbf
from pycrsx.utils import convert_crs
from util import pd_grouby_fun
from pyproj import Proj


def interp_to_grid(df, time_col, x_col, y_col, data_col, grid_res, from_crs=None, to_crs=None, interp_fun='cubic', agg_ts_fun=None, period=None, digits=2):
    """
    Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). The values can be aggregated in time, but only as a sum or mean. Returns a DataFrame of gridded interpolated results at each time stamp.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing four columns as shown in the below parameters.
    time_col: str
        The time column name.
    x_col: str
        The x column name.
    y_col: str
        The y column name.
    data_col: str
        The data column name.
    grid_res: int
        The resulting grid resolution in meters (or the unit of the final projection).
    from_crs: int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs: int or str or None
        The projection for the output data similar to from_crs.
    interp_fun: str
        The scipy griddata interpolation function to be applied (see `https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html`_).
    agg_ts_fun: str or None
        The pandas time series resampling function to resample the data in time (either 'mean' or 'sum'). If None, then no time resampling.
    period: str or None
        The pandas time series code to resample the data in time (i.e. '2H' for two hours).
    digits: int
        the number of digits to round.

    Returns
    -------
    DataFrame
    """

    df1 = df.copy()

    if (to_crs == 4326) or ((from_crs == 4326) and (to_crs is None)):
        xy_digits = 5
    else:
        xy_digits = 0

    ### Resample the time series data
    if isinstance(agg_ts_fun, str):
        df1a = df1.set_index(time_col)
        fun1 = pd_grouby_fun(df1a, agg_ts_fun)
        df2 = fun1(df1a.groupby([pd.Grouper(freq=period), pd.Grouper(y_col), pd.Grouper(x_col)])[data_col]).reset_index()
    else:
        df2 = df1

    time = df2[time_col].sort_values().unique()

    ### convert to new projection and prepare X/Y data
    if from_crs is None:
        x = df2.loc[df2[time_col] == time[0], x_col].values
        y = df2.loc[df2[time_col] == time[0], y_col].values
    else:
        data1 = df2.loc[df2[time_col] == time[0]]
        from_crs1 = convert_crs(from_crs, pass_str=True)
        to_crs1 = convert_crs(to_crs, pass_str=True)
        geometry = [Point(xy) for xy in zip(data1[x_col], data1[y_col])]
        gpd1 = gpd.GeoDataFrame(data1.index, geometry=geometry, crs=from_crs1)
        gpd2 = gpd1.to_crs(crs=to_crs1)
        x = gpd2.geometry.apply(lambda p: p.x).values
        y = gpd2.geometry.apply(lambda p: p.y).values

    xy = np.column_stack((x, y))

    max_x = np.round(x.max(), xy_digits)
    min_x = np.round(x.min(), xy_digits)

    max_y = np.round(y.max(), xy_digits)
    min_y = np.round(y.min(), xy_digits)

    new_x = np.arange(min_x, max_x, grid_res)
    new_y = np.arange(min_y, max_y, grid_res)
    x_int, y_int = np.meshgrid(new_x, new_y)

    ### Create new df
    x_int2 = x_int.flatten()
    y_int2 = y_int.flatten()
    xy_int = np.column_stack((x_int2, y_int2))
    time_df = np.repeat(time, len(x_int2))
    x_df = np.tile(x_int2, len(time))
    y_df = np.tile(y_int2, len(time))
    new_df = pd.DataFrame({'time': time_df, 'x': x_df, 'y': y_df, data_col: np.repeat(0, len(time) * len(x_int2))})

    ### Grid interp
    new_lst = []
    for t in pd.to_datetime(time):
        print(t)
        set1 = df2.loc[df2[time_col] == t, data_col]
        new_z = griddata(xy, set1.values, xy_int, method=interp_fun).round(digits)
        new_z[new_z <= 0] = 0
        new_lst.extend(new_z.tolist())
    new_df.loc[:, data_col] = new_lst

    ### Export results
    return new_df[new_df[data_col].notnull()]


def interp_to_points(df, time_col, x_col, y_col, data_col, point_shp, point_site_col, from_crs, to_crs=None, interp_fun='cubic', agg_ts_fun=None, period=None, digits=2):
    """
    Function to take a dataframe of z values and interate through and resample both in time and space. Returns a DataFrame in the shape of the points from the point_shp.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing four columns as shown in the below parameters.
    time_col: str
        The time column name.
    x_col: str
        The x column name.
    y_col: str
        The y column name.
    data_col: str
        The data column name.
    point_shp: str or GeoDataFrame
        Path to shapefile of points to be interpolated or a GeoPandas GeoDataFrame.
    point_site_col: str
        The column name of the site names/numbers of the point_shp.
    grid_res: int
        The resulting grid resolution in meters (or the unit of the final projection).
    from_crs: int or str
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs: int or str
        The projection for the output data similar to from_crs.
    interp_fun: str
        The scipy griddata interpolation function to be applied (see `https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html`_).
    agg_ts_fun: str or None
        The pandas time series resampling function to resample the data in time (either 'mean' or 'sum'). If None, then no time resampling.
    period: str or None
        The pandas time series code to resample the data in time (i.e. '2H' for two hours).
    digits: int
        the number of digits to round.

    Returns
    -------
    DataFrame
    """

    ### Read in points
    if isinstance(point_shp, str) & isinstance(point_site_col, str):
        points = gpd.read_file(point_shp)[[point_site_col, 'geometry']]
        to_crs1 = points.crs
    elif isinstance(point_shp, gpd.GeoDataFrame) & isinstance(point_site_col, str):
        points = point_shp[[point_site_col, 'geometry']]
        to_crs1 = points.crs
    else:
        raise ValueError('point_shp must be a str path to a shapefile or a GeoDataFrame and point_site_col must be a str.')

    ### Create the grids
    df1 = df.copy()

    ### Resample the time series data
    if isinstance(agg_ts_fun, str):
        df1a = df1.set_index(time_col)
        fun1 = pd_grouby_fun(df1a, agg_ts_fun)
        df2 = fun1(df1a.groupby([pd.Grouper(freq=period), pd.Grouper(y_col), pd.Grouper(x_col)])[data_col]).reset_index()
    else:
        df2 = df1

    time = df2[time_col].sort_values().unique()

    ### Convert input data to crs of points shp and create input xy
    data1 = df2.loc[df2[time_col] == time[0]]
    from_crs1 = convert_crs(from_crs, pass_str=True)

    if to_crs is not None:
        to_crs1 = convert_crs(to_crs, pass_str=True)
        points = points.to_crs(to_crs1)
    geometry = [Point(xy) for xy in zip(data1[x_col], data1[y_col])]
    gpd1 = gpd.GeoDataFrame(data1.index, geometry=geometry, crs=from_crs1)
    gpd2 = gpd1.to_crs(crs=to_crs1)
    x = gpd2.geometry.apply(lambda p: p.x).values
    y = gpd2.geometry.apply(lambda p: p.y).values

    xy = np.column_stack((x, y))

    ### Prepare the x and y of the points geodataframe output
    x_int = points.geometry.apply(lambda p: p.x).values
    y_int = points.geometry.apply(lambda p: p.y).values
    sites = points[point_site_col]

    xy_int = np.column_stack((x_int, y_int))

    ### Create new df
    sites_ar = np.tile(sites, len(time))
    time_ar = np.repeat(time, len(xy_int))
    x_ar = np.tile(x_int, len(time))
    y_ar = np.tile(y_int, len(time))
    new_df = pd.DataFrame({'site': sites_ar, 'time': time_ar, 'x': x_ar, 'y': y_ar, data_col: np.repeat(0, len(time) * len(xy_int))})

    new_lst = []
    for t in pd.to_datetime(time):
        print(t)
        set1 = df2.loc[df2[time_col] == t, data_col]
        new_z = griddata(xy, set1.values, xy_int, method=interp_fun).round(digits)
        new_z[new_z <= 0] = 0
        new_lst.extend(new_z.tolist())
    new_df.loc[:, data_col] = new_lst

    ### Export results
    return new_df[new_df[data_col].notnull()]


def grid_resample(x, y, z, x_int, y_int, digits=3, method='multiquadric'):
    """
    Function to interpolate and resample a set of x, y, z values using the scipy Rbf function.
    """

    interp1 = Rbf(x, y, z, function=method)
    z_int = interp1(x_int, y_int).round(digits)
    z_int[z_int < 0] = 0

    z_int2 = z_int.flatten()
    return z_int2
