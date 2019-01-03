# -*- coding: utf-8 -*-
"""
Raster and spatial interpolation functions.
"""
import pandas as pd
import numpy as np
import fiona
import xarray as xr
from scipy.interpolate import griddata, Rbf
from pycrsx.utils import convert_crs
from pyproj import Proj, transform


def interp_to_grid(df, time_col, x_col, y_col, data_col, grid_res, from_crs=None, to_crs=None, interp_fun='cubic', digits=2, output='pandas'):
    """
    Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). The values can be aggregated in time, but but are not interpolated. Returns a DataFrame of gridded interpolated results at each time stamp or an xarray Dataset with the 3 dimensions.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing four columns as shown in the below parameters.
    time_col : str
        The time column name.
    x_col : str
        The x column name.
    y_col : str
        The y column name.
    data_col : str
        The data column name.
    grid_res : int
        The resulting grid resolution in meters (or the unit of the final projection).
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    interp_fun : str
        The scipy griddata interpolation function to be applied (see `https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html`_).
    digits : int
        The number of digits to round the results.
    output : str
        If output = 'pandas' then the function will return a pandas DataFrame. If output = 'xarray' then the function will return an xarray Dataset.

    Returns
    -------
    DataFrame or Dataset (see output)
    """

    df2 = df.copy()

    if (to_crs == 4326) or ((from_crs == 4326) and (to_crs is None)):
        xy_digits = 5
    else:
        xy_digits = 0

    time = pd.to_datetime(df2[time_col].sort_values().unique())

    ### convert to new projection and prepare X/Y data
    if (from_crs is not None) & (to_crs is not None):
        from_crs1 = Proj(convert_crs(from_crs, pass_str=True), preserve_units=True)
        to_crs1 = Proj(convert_crs(to_crs, pass_str=True), preserve_units=True)
        xy1 = list(zip(df2[x_col], df2[y_col]))
        xy_new1 = list(zip(*[transform(from_crs1, to_crs1, x, y) for x, y in xy1]))
        df2[x_col] = xy_new1[0]
        df2[y_col] = xy_new1[1]

    df2.sort_values(time_col, inplace=True)

    maxes = df2[[x_col, y_col]].max().round(xy_digits)
    mins = df2[[x_col, y_col]].min().round(xy_digits)

    max_x = maxes.loc[x_col]
    min_x = mins.loc[x_col]

    max_y = maxes.loc[y_col]
    min_y = mins.loc[y_col]

    new_x = np.arange(min_x, max_x, grid_res)
    new_y = np.arange(min_y, max_y, grid_res)
    x_int, y_int = np.meshgrid(new_x, new_y)

    x_int2 = x_int.flatten()
    y_int2 = y_int.flatten()
    xy_int = np.column_stack((x_int2, y_int2))

    ### Grid interp
    new_lst = []
    for t in time:
        print(t)
        set1 = df2.loc[df2.time == t]
        xy = set1[[x_col, y_col]].values
        new_z = griddata(xy, set1[data_col].values, xy_int, method=interp_fun).round(digits)
        new_z[new_z <= 0] = 0
        new_lst.extend(new_z.tolist())

    if output == 'xarray':
        ar1 = np.array(new_lst).reshape(len(new_x), len(new_y), len(time))
        new1 = xr.DataArray(ar1, coords=[new_x, new_y, time], dims=['x', 'y', 'time'], name=data_col).to_dataset()
    elif output == 'pandas':
        time_df = np.repeat(time, len(x_int2))
        x_df = np.tile(x_int2, len(time))
        y_df = np.tile(y_int2, len(time))
        new1 = pd.DataFrame({'time': time_df, 'x': x_df, 'y': y_df, data_col: new_lst}).dropna().set_index(['time', 'x', 'y'])

    ### Export results
    return new1


def interp_to_points(df, time_col, x_col, y_col, data_col, point_path, point_site_col, from_crs, to_crs=None, interp_fun='cubic', digits=2):
    """
    Function to take a dataframe of z values and interate through and resample both in time and space. Returns a DataFrame in the shape of the points from the point_shp.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing four columns as shown in the below parameters.
    time_col : str
        The time column name.
    x_col : str
        The x column name.
    y_col : str
        The y column name.
    data_col : str
        The data column name.
    point_path : str
        Path to geometry file of points to be interpolated (e.g. shapefile). Can be anything that fiona/gdal can open.
    point_site_col : str
        The column name of the site names/numbers of the point_shp.
    grid_res : int
        The resulting grid resolution in meters (or the unit of the final projection).
    from_crs : int or str
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str
        The projection for the output data similar to from_crs.
    interp_fun : str
        The scipy griddata interpolation function to be applied (see `https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html`_).
    digits : int
        The number of digits to round the results.

    Returns
    -------
    DataFrame
    """

    ### Read in points
    if isinstance(point_path, str) & isinstance(point_site_col, str):
        with fiona.open(point_path) as f1:
            point_crs = Proj(f1.crs, preserve_units=True)
            points = {p['properties'][point_site_col]: p['geometry']['coordinates'] for p in f1 if p['geometry']['type'] == 'Point'}
    else:
        raise ValueError('point_path must be a str path to a geometry file (e.g. shapefile) and point_site_col must be a str.')

    ### Create the grids
    df2 = df.copy()

    time = pd.to_datetime(df2[time_col].sort_values().unique())

    ### Convert input data to crs of points shp and create input xy
    if to_crs is not None:
        to_crs1 = Proj(convert_crs(to_crs, pass_str=True), preserve_units=True)
        points = {p: transform(point_crs, to_crs1, points[p][0], points[p][1]) for p in points}
    else:
        to_crs1 = point_crs
    from_crs1 = Proj(convert_crs(from_crs, pass_str=True), preserve_units=True)
    xy1 = list(zip(df2[x_col], df2[y_col]))
    xy_new1 = list(zip(*[transform(from_crs1, to_crs1, x, y) for x, y in xy1]))
    df2[x_col] = xy_new1[0]
    df2[y_col] = xy_new1[1]

    ### Prepare the x and y of the points geodataframe output
    sites = list(points.keys())
    xy_int = np.array(list(points.values()))

    new_lst = []
    for t in time:
        print(t)
        set1 = df2.loc[df2.time == t]
        xy = set1[[x_col, y_col]].values
        new_z = griddata(xy, set1[data_col].values, xy_int, method=interp_fun).round(digits)
        new_z[new_z <= 0] = 0
        new_lst.extend(new_z.tolist())

    ### Create new df
    sites_ar = np.tile(sites, len(time))
    time_ar = np.repeat(time, len(xy_int))
    x_ar = np.tile(xy_int.T[0], len(time))
    y_ar = np.tile(xy_int.T[1], len(time))
    new_df = pd.DataFrame({'site': sites_ar, 'time': time_ar, 'x': x_ar, 'y': y_ar, data_col: new_lst}).set_index(['time', 'x', 'y'])

    ### Export results
    return new_df


def grid_resample(x, y, z, x_int, y_int, digits=3, method='multiquadric'):
    """
    Function to interpolate and resample a set of x, y, z values using the scipy Rbf function.
    """

    interp1 = Rbf(x, y, z, function=method)
    z_int = interp1(x_int, y_int).round(digits)
    z_int[z_int < 0] = 0

    z_int2 = z_int.flatten()
    return z_int2
