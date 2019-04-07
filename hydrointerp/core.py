# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:43:35 2019

@author: michaelek
"""
import numpy as np
import pandas as pd
import xarray as xr
from hydrointerp import interp2d



###################################
### base class


class Interp(object):
    """
    Base Interp class to prepare the input data for the interpolation functions.

    Parameters
    ----------
    data : DataFrame or Dataset
        A pandas DataFrame containing four columns as shown in the below parameters or an Xarray Dataset.
    time_name : str
        If grid is a DataFrame, then time_name is the time column name. If grid is a Dataset, then time_name is the time coordinate name.
    x_name : str
        If grid is a DataFrame, then x_name is the x column name. If grid is a Dataset, then x_name is the x coordinate name.
    y_name : str
        If grid is a DataFrame, then y_name is the y column name. If grid is a Dataset, then y_name is the y coordinate name.
    data_name : str
        If grid is a DataFrame, then data_name is the data column name. If grid is a Dataset, then data_name is the data variable name.
    from_crs : int or str or None
        The projection info for the input data if the result should be reprojected to the to_crs projection (either a proj4 str or epsg int).

        """
    def __init__(self, data, time_name, x_name, y_name, data_name, from_crs):

        ## Assign variables
        self.data = data.copy()
        self.time_name = time_name
        self.x_name = x_name
        self.y_name = y_name
        self.data_name = data_name
        self.from_crs = from_crs

        ## Check data type
        if isinstance(data, pd.DataFrame):
            self.input_type = pd.DataFrame
            self.points_to_grid = self._points_to_grid
            self.points_to_points = self._points_to_points
        elif isinstance(data, xr.Dataset):
            self.input_type = xr.Dataset
            self.grid_to_grid = self._grid_to_grid
            self.grid_to_points = self._grid_to_points
        else:
            raise ValueError('data must be either a DataFrame or Dataset')

    def _grid_to_grid(self, grid_res, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, digits=2, min_val=None):
        """
        Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). Returns an xarray Dataset with the 3 dimensions. Uses the scipy interpolation function called `map_coordinates <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_.

        Parameters
        ----------
        grid_res : int or float
            The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        bbox : tuple of int or float
            The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
        extrapolation : str
            The equivalent of 'mode' in the map_coordinates function. Options are: 'constant', 'nearest', 'reflect', 'mirror', and 'wrap'. Most reseaonable options for this function will be either 'constant' or 'nearest'. See `scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_ for more details.
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
        out1 = interp2d.grid_to_grid(self.data, self.time_name, self.x_name, self.y_name, self.data_name, grid_res, self.from_crs, to_crs, bbox, order, extrapolation, fill_val, digits, min_val)
        return out1

    def _grid_to_points(self, point_data, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, digits=2, min_val=None):
        """
        Function to take a dataframe of point value inputs (df) and interpolate to other points (point_data). Uses the `scipy griddata function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for interpolation.

        Parameters
        ----------
        point_data : str or DataFrame
        Path to geometry file of points to be interpolated (e.g. shapefile). Can be any file type that fiona/gdal can open. It can also be a DataFrame with 'x' and 'y' columns and the crs must be the same as to_crs.
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        bbox : tuple of int or float
            The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
        extrapolation : str
            The equivalent of 'mode' in the map_coordinates function. Options are: 'constant', 'nearest', 'reflect', 'mirror', and 'wrap'. Most reseaonable options for this function will be either 'constant' or 'nearest'. See `scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_ for more details.
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
        out1 = interp2d.grid_to_points(self.data, self.time_name, self.x_name, self.y_name, self.data_name, point_data, self.from_crs, to_crs, order, digits, min_val)
        return out1

    def _points_to_grid(self, grid_res, to_crs=None, bbox=None, method='linear', extrapolation='contstant', fill_val=np.nan, digits=2, min_val=None):
        """
        Function to take a dataframe of point value inputs (df) and interpolate to a grid. Uses the `scipy griddata function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for interpolation.

    Parameters
    ----------
    grid_res : int or float
        The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
    to_crs : int or str or None
        The projection for the output data similar to from_crs.
    bbox : tuple of int or float
        The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
    method : str
        The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
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
        out1 = interp2d.points_to_grid(self.data, self.time_name, self.x_name, self.y_name, self.data_name, grid_res, self.from_crs, to_crs, bbox, method, extrapolation, fill_val, digits, min_val)
        return out1

    def _points_to_points(self, point_data, to_crs=None, method='linear', digits=2, min_val=None):
        """
        Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). Returns an xarray Dataset with the 3 dimensions. Uses the scipy interpolation function called `map_coordinates <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_.

        Parameters
        ----------
        grid_res : int or float
            The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        bbox : tuple of int or float
            The bounding box for the output interpolation in the to_crs projection. None will return a similar grid extent as the input. The tuple should contain four ints or floats in the following order: (x_min, x_max, y_min, y_max)
        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
        extrapolation : str
            The equivalent of 'mode' in the map_coordinates function. Options are: 'constant', 'nearest', 'reflect', 'mirror', and 'wrap'. Most reseaonable options for this function will be either 'constant' or 'nearest'. See `scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_ for more details.
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
        out1 = interp2d.points_to_points(self.data, self.time_name, self.x_name, self.y_name, self.data_name, point_data, self.from_crs, to_crs, method, digits, min_val)
        return out1






