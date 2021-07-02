# -*- coding: utf-8 -*-
"""
Raster and spatial interpolation functions.
"""
import pandas as pd
import numpy as np
try:
    import fiona
    _fiona = True
except:
    _fiona = False
import xarray as xr
from scipy.interpolate import griddata, RectBivariateSpline
from pyproj import Proj, CRS, Transformer
from scipy.ndimage import map_coordinates
from hydrointerp.util import grid_xy_to_map_coords, point_xy_to_map_coords
#from util import grid_xy_to_map_coords, point_xy_to_map_coords

#################################################
### Helper functions


def _process_grid_input(input_data, time_name, x_name, y_name, data_name, zero_pad=True):
    if isinstance(input_data, pd.DataFrame):
        grouped = input_data.set_index([time_name, y_name, x_name])[data_name].sort_index()
        # grouped = input_data.set_index([time_name, x_name, y_name])[data_name].sort_index()
        input_data = grouped.to_xarray().to_dataset()
    if isinstance(input_data, xr.Dataset):
        da1 = input_data[data_name]
        da2 = da1.transpose(time_name, y_name, x_name).sortby([time_name, y_name, x_name])
        # da2 = da1.transpose(time_name, x_name, y_name).sortby([time_name, x_name, y_name])
        arr1 = da2.values
        if zero_pad:
            np.nan_to_num(arr1, False)
        x = da1[x_name].values
        y = da1[y_name].values
        time1 = da1[time_name].values
        # xy_orig_pts = np.dstack(np.meshgrid(y, x)).reshape(-1, 2)
        xy_orig_pts = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

    return arr1, xy_orig_pts, time1

################################################
### Core functions


def grid_to_grid(grid, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, digits=2, min_val=None):
    """
    Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). Returns an xarray Dataset with the 3 dimensions. Uses the scipy interpolation function called map_coordinates.

    Parameters
    ----------
    grid : DataFrame or Dataset
        A pandas DataFrame or an xarray Dataset. It's recommended to use an xarray Dataset for the input grid as it ensures that the user knows that it is truly regular. Regardless, the input will be regularised.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    grid_res : int or float
        The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    bbox : tuple of int or float
        The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
    order : int
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
    extrapolation : str
        The equivalent of 'mode' in the map_coordinates function. Options are: 'constant', 'nearest', 'reflect', 'mirror', and 'wrap'. Most reseaonable options for this function will be either 'constant' or 'nearest'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_ for more details.
    fill_val : int or float
        If 'constant' if passed to the extrapolation parameter, fill_val assigns the value outside of the boundary. Defaults to numpy.nan.
    digits : int
        The number of digits to round the output.
    min_val : int, float, or None
        The minimum value for the results. All results below min_val will be assigned min_val.

    Returns
    -------
    xarray Dataset
    """
    print('Preparing input and output')
    if from_crs == 4326:
        input_digits = 4
    else:
        input_digits = 0
    if (to_crs == 4326) | ((from_crs == 4326) & (isinstance(to_crs, (str, int, dict)))):
        output_digits = 4
    else:
        output_digits = 0

    ### Prepare input data
    arr1, xy_orig_pts, time1 = _process_grid_input(grid, time_name, x_name, y_name, data_name)

    input_coords, dxy, x_min, y_min = grid_xy_to_map_coords(xy_orig_pts, input_digits)

    ### convert to new projection and prepare X/Y data
    if isinstance(from_crs, (str, int, dict)) & isinstance(to_crs, (str, int, dict)):
        from_crs1 = Proj(CRS.from_user_input(from_crs))
        to_crs1 = Proj(CRS.from_user_input(to_crs))
        trans1 = Transformer.from_proj(from_crs1, to_crs1, always_xy=True)
        xy_new = np.array(trans1.transform(*xy_orig_pts.T)).round(output_digits)
        # out_y_min, out_x_min = xy_new.min(1)
        # out_y_max, out_x_max = xy_new.max(1)
        out_x_min, out_y_min = xy_new.min(1)
        out_x_max, out_y_max = xy_new.max(1)
    else:
        out_x_max, out_y_max = xy_orig_pts.max(0)
        out_x_max, out_y_max = xy_orig_pts.max(0)
        out_x_min = x_min
        out_y_min = y_min

    ### Prepare output data
    if isinstance(bbox, tuple):
        out_x_min, out_x_max, out_y_min, out_y_max = bbox

    new_x = np.arange(out_x_min, out_x_max, grid_res)
    new_y = np.arange(out_y_min, out_y_max, grid_res)

    # xy_out = np.dstack(np.meshgrid(new_y, new_x)).reshape(-1, 2)
    xy_out = np.dstack(np.meshgrid(new_x, new_y)).reshape(-1, 2)

    if isinstance(to_crs, (str, int, dict)):
        trans2 = Transformer.from_proj(to_crs1, from_crs1, always_xy=True)
        xy_new_index = np.array(trans2.transform(*xy_out.T)).T
    else:
        xy_new_index = xy_out

    output_coords = point_xy_to_map_coords(xy_new_index, dxy, x_min, y_min, float)

    ### Run interpolation (need to add mutliprossessing)
    arr2 = np.zeros((len(time1), output_coords.shape[1]), arr1.dtype)

    print('Running interpolations...')
    ## An example for using RectBivariateSpline as the equivalent to map_coordinates output (about half as fast):
#    arr2a = arr2.copy()
#
#    x_out = xy_new_index.T[1]
#    y_out = xy_new_index.T[0]
#
#    for d in np.arange(len(arr1)):
#        arr2a[d] = RectBivariateSpline(y, x, arr1[d], kx=3, ky=3).ev(y_out, x_out)

    for d in np.arange(len(arr1)):
         map_coordinates(arr1[d], output_coords, arr2[d], order=order, mode=extrapolation, cval=fill_val, prefilter=True)

    ### Reshape and package data
    print('Packaging up the output')
    arr2 = arr2.reshape((len(time1), len(new_y), len(new_x))).round(digits)

    if isinstance(min_val, (int, float)):
        arr2[arr2 < min_val] = min_val

    new_ds = xr.DataArray(arr2, coords=[time1, new_y, new_x], dims=['time', 'y', 'x'], name=data_name).to_dataset()

    return new_ds


def grid_to_points(grid, time_name, x_name, y_name, data_name, point_data, from_crs, to_crs=None, order=3, digits=2, min_val=None):
    """
    Function to take a dataframe of point value inputs (df) and interpolate to other points (point_data). Uses the scipy interpolation function called map_coordinate.

    Parameters
    ----------
    grid : DataFrame or Dataset
        A pandas DataFrame or an xarray Dataset. It's recommended to use an xarray Dataset for the input grid as it ensures that the user knows that it is truly regular. Regardless, the input will be regularised.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    point_data : str or DataFrame
        Path to geometry file of points to be interpolated (e.g. shapefile). Can be any file type that fiona/gdal can open. It can also be a DataFrame with 'x' and 'y' columns and the crs must be the same as to_crs.
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    order : int
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
    digits : int
        The number of digits to round the output.
    min_val : int, float, or None
        The minimum value for the results. All results below min_val will be assigned min_val.

    Returns
    -------
    DataFrame
    """
    print('Preparing input and output')
    if from_crs == 4326:
        input_digits = 4
    else:
        input_digits = 0

    ### Read in points
    if isinstance(point_data, str):
        if _fiona:
            with fiona.open(point_data) as f1:
                point_crs = Proj(f1.crs)
                points = np.array([tuple(p['geometry']['coordinates']) for p in f1 if p['geometry']['type'] == 'Point'])
        else:
            raise ImportError('Please install fiona for importing GIS files')
    elif isinstance(point_data, pd.DataFrame):
        point_crs = CRS.from_user_input(to_crs)
        points = point_data[['x', 'y']].values
    else:
        raise ValueError('point_path must be a str path to a geometry file (e.g. shapefile) or a DataFrame with the same x_name and y_name columns')

    ### Prepare input grid
    arr1, xy_orig_pts, time1 = _process_grid_input(grid, time_name, x_name, y_name, data_name)

    input_coords, dxy, x_min, y_min = grid_xy_to_map_coords(xy_orig_pts, input_digits)

    ### Convert points to to_crs, from_crs, and array index
    from_crs1 = CRS.from_user_input(from_crs)
    if to_crs is not None:
        to_crs1 = CRS.from_user_input(to_crs)
        trans1 = Transformer.from_crs(point_crs, to_crs1)
        points = np.array(trans1.transform(*points.T)).T
    else:
        to_crs1 = point_crs
    trans2 = Transformer.from_crs(to_crs1, from_crs1)
    points_from_crs = np.array(trans2.transform(*points.T)).T
    points_coords = point_xy_to_map_coords(points_from_crs, dxy, x_min, y_min, float)

    ### Run interpolations
    print('Running interpolations...')
    arr2 = np.zeros((len(time1), points_coords.shape[1]), arr1.dtype)

    for d in np.arange(len(arr1)):
         map_coordinates(arr1[d], points_coords, arr2[d], order=order, cval=np.nan, prefilter=True)

    if isinstance(min_val, (int, float)):
        arr2[arr2 < min_val] = min_val

    ### Reshape and package data
    print('Packaging up the output')
    arr3 = arr2.flatten().round(digits)

    time_ar = np.repeat(time1, len(points))
    y_ar = np.tile(points.T[0], len(time1))
    x_ar = np.tile(points.T[1], len(time1))
    new_df = pd.DataFrame({'time': time_ar, 'x': x_ar, 'y': y_ar, data_name: arr3}).set_index(['time', 'x', 'y'])

    return new_df


def points_to_grid(df, time_name, x_name, y_name, data_name, grid_res, from_crs, to_crs=None, bbox=None, method='linear', extrapolation='contstant', fill_val=np.nan, digits=2, min_val=None):
    """
    Function to take a dataframe of point value inputs (df) and interpolate to a grid. Uses the scipy griddata function for interpolation.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing four columns as shown in the below parameters.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    grid_res : int or float
        The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    bbox : tuple of int or float
        The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
    method : str
        The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
    extrapolation : str
        Either 'constant' or 'nearest'.
    fill_val : int or float
        If 'constant' if passed to the extrapolation parameter, fill_val assigns the value outside of the boundary. Defaults to numpy.nan.
    digits : int
        The number of digits to round the output.
    min_val : int, float, or None
        The minimum value for the results. All results below min_val will be assigned min_val.

    Returns
    -------
    xarray Dataset
    """
    print('Prepare input and output data')
    if (to_crs == 4326) | ((from_crs == 4326) & (to_crs is None)):
        output_digits = 4
    else:
        output_digits = 0

    ### Prepare input data
    df2 = df.copy()

    time1 = pd.to_datetime(df2[time_name].sort_values().unique())

    ### Convert input data to crs of points shp and create input xy
    from_crs1 = Proj(CRS.from_user_input(from_crs))
    xy1 = np.array(list(zip(df2[x_name], df2[y_name]))).T
    if isinstance(to_crs, (str, int, dict)):
        to_crs1 = Proj(CRS.from_user_input(to_crs))
        trans1 = Transformer.from_proj(from_crs1, to_crs1)
        xy1 = np.array(trans1.transform(*xy1))
        df2[x_name] = xy1[0]
        df2[y_name] = xy1[1]

    ### Prepare output data
    if isinstance(bbox, tuple):
        out_x_min, out_x_max, out_y_min, out_y_max = bbox
    else:
        out_x_min, out_y_min = xy1.min(1).round(output_digits)
        out_x_max, out_y_max = xy1.max(1).round(output_digits)

    new_x = np.arange(out_x_min, out_x_max, grid_res)
    new_y = np.arange(out_y_min, out_y_max, grid_res)

    xy_out = np.dstack(np.meshgrid(new_x, new_y)).reshape(-1, 2)

    ### Run interpolations
    print('Run interpolations...')
    arr2 = np.zeros((len(time1), xy_out.shape[0]), df2[data_name].dtype)
    index1 = {time1[i]: i for i in np.arange(len(time1))}

#    start1 = time()
    for name, group in df2.groupby(time_name):
        print(name)
        i = index1[name]
        xy = group[[x_name, y_name]].values
        arr2[i] = griddata(xy, group[data_name].values, xy_out, method=method, fill_value=fill_val).round(digits)
        if extrapolation == 'nearest':
            nan_index = np.isnan(arr2[i])
            nan_xy = xy_out[nan_index]
            nonnan_values = arr2[i][~nan_index]
            nonnan_xy = xy_out[~nan_index]
            arr2[i][nan_index] = griddata(nonnan_xy, nonnan_values, nan_xy, method='nearest').round(digits)
#    end1 = time()
#    setb = end1 - start1

    ### Reshape and package data
    print('Packaging up the output')
    arr2 = arr2.reshape((len(time1), len(new_y), len(new_x))).round(digits)

    if isinstance(min_val, (int, float)):
        arr2[arr2 < min_val] = min_val

    new_ds = xr.DataArray(arr2, coords=[time1, new_y, new_x], dims=['time', 'y', 'x'], name=data_name).to_dataset()

    return new_ds


def points_to_points(df, time_name, x_name, y_name, data_name, point_data, from_crs, to_crs=None, method='linear', digits=2, min_val=None):
    """
    Function to take a dataframe of point value inputs (df) and interpolate to other points (point_data). Uses the scipy griddata function for interpolation.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing four columns as shown in the below parameters.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    point_data : str or DataFrame
        Path to geometry file of points to be interpolated (e.g. shapefile). Can be any file type that fiona/gdal can open. It can also be a DataFrame with 'x' and 'y' columns and the crs must be the same as to_crs.
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    method : str
        The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
    fill_val : int or float
        fill_val assigns the value outside of the boundary. Defaults to numpy.nan.
    digits : int
        The number of digits to round the output.
    min_val : int, float, or None
        The minimum value for the results. All results below min_val will be assigned min_val.

    Returns
    -------
    DataFrame
    """
    ### Read in points
    if isinstance(point_data, str):
        if _fiona:
            with fiona.open(point_data) as f1:
                point_crs = Proj(f1.crs)
                points = np.array([p['geometry']['coordinates'] for p in f1 if p['geometry']['type'] == 'Point'])
        else:
            raise ImportError('Please install fiona for importing GIS files')
    elif isinstance(point_data, pd.DataFrame):
        point_crs = Proj(CRS.from_user_input(to_crs))
        points = point_data[['x', 'y']].values
    else:
        raise ValueError('point_path must be a str path to a geometry file (e.g. shapefile) or a DataFrame with the same x_name and y_name columns')

    df2 = df.copy()

    time1 = pd.to_datetime(df2[time_name].sort_values().unique())

    ### Convert input data to crs of points shp and create input xy
    if to_crs is not None:
        to_crs1 = Proj(CRS.from_user_input(to_crs))
        trans1 = Transformer.from_proj(point_crs, to_crs1)
        points = np.array(trans1.transform(*points.T))
    else:
        to_crs1 = point_crs
    from_crs1 = Proj(CRS.from_user_input(from_crs))
    xy1 = df2[[y_name, x_name]].values
    trans2 = Transformer.from_proj(from_crs1, to_crs1)
    xy_new1 = np.array(trans2.transform(*xy1.T))
    df2[x_name] = xy_new1[0]
    df2[y_name] = xy_new1[1]

    ### Run interpolations
    points1 = np.array((points[0], points[1])).T
    new_lst = []
    for name, group in df2.groupby(time_name):
        print(name)
        xy = group[[x_name, y_name]].values
        new_z = griddata(xy, group[data_name].values, points1, method=method, fill_value=np.nan).round(digits)
        if isinstance(min_val, (int, float)):
            new_z[new_z < min_val] = min_val
        new_lst.extend(new_z.tolist())

    ### Create new df
    time_ar = np.repeat(time1, len(points1))
    x_ar = np.tile(points1.T[0], len(time1))
    y_ar = np.tile(points1.T[1], len(time1))
    new_df = pd.DataFrame({'time': time_ar, 'x': x_ar, 'y': y_ar, data_name: new_lst}).set_index(['time', 'x', 'y'])

    ### Export results
    return new_df


def grid_interp_na(grid, time_name, x_name, y_name, data_name, method='linear', min_val=None):
    """
    Function to fill in nan in a grid to make it complete.

    Parameters
    ----------
    grid : DataFrame or Dataset
        A pandas DataFrame or an xarray Dataset. It's recommended to use an xarray Dataset for the input grid as it ensures that the user knows that it is truly regular. Regardless, the input will be regularised.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    method : str
        The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
    min_val : int, float, or None
        The minimum value for the results. All results below min_val will be assigned min_val.

    Returns
    -------
    Xarray Dataset
    """
    ### Prepare input data
    arr1, xy_orig_pts, time1 = _process_grid_input(grid, time_name, x_name, y_name, data_name, False)
    arr2 = arr1.reshape(len(time1), xy_orig_pts.shape[0])

    ### Run through interpolations
    for t in arr2:
        isnan1 = np.isnan(t)
        t[isnan1] = griddata(xy_orig_pts[~isnan1], t[~isnan1], xy_orig_pts[isnan1], method=method, fill_value=np.nan)

    ### Reshape and package data
    arr2 = arr2.reshape(arr1.shape)

    if isinstance(min_val, (int, float)):
        arr2[arr2 < min_val] = min_val

    new_ds = xr.DataArray(arr2, coords=[time1, grid[y_name].values, grid[x_name].values], dims=['time', y_name, x_name], name=data_name).to_dataset()

    return new_ds
