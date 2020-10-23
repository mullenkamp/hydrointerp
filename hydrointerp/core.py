# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:43:35 2019

@author: michaelek
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj, CRS, Transformer
from hydrointerp import interp2d
#import interp2d
from hydrointerp import util
# import util
import random

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

        Returns
        -------
        Interp object
        """
    def __init__(self, grid_data=None, grid_time_name=None, grid_x_name=None, grid_y_name=None, grid_data_name=None, grid_crs=None, point_data=None, point_time_name=None, point_x_name=None, point_y_name=None, point_data_name=None, point_crs=None):

        ## Assign variables
        self._grid_crs = grid_crs
        self._point_crs = point_crs

        ## Check data type
        if point_data is not None:
            if isinstance(point_data, pd.DataFrame):
                self.point_data = point_data.rename(columns={point_x_name: 'x', point_y_name: 'y', point_time_name: 'time', point_data_name: 'precip'}).copy()
                if self.point_data.dtypes['time'].name != 'datetime64[ns]':
                    self.point_data['time'] = pd.to_datetime(self.point_data['time'])
                self.points_to_grid = self._points_to_grid
                self.points_to_points = self._points_to_points
            else:
                raise TypeError('point_data must be a DataFrame')
        if grid_data is not None:
            if isinstance(grid_data, xr.Dataset):
                self.grid_data = grid_data.rename({grid_x_name: 'x', grid_y_name: 'y', grid_time_name: 'time', grid_data_name: 'precip'}).copy()
                if self.grid_data.coords['time'].dtype.name != 'datetime64[ns]':
                    self.grid_data['time'] = pd.to_datetime(self.grid_data['time'].to_index())
                self.grid_interp_na = self._grid_interp_na
                self.grid_to_grid = self._grid_to_grid
                self.grid_to_points = self._grid_to_points
            else:
                raise TypeError('grid_data must be a Dataset')
        if (point_data is not None) and (grid_data is not None):
            self.adjust_grid_from_points = self._adjust_grid_from_points
            self.validate_grid_from_points = self._validate_grid_from_points
        if (point_data is None) and (grid_data is None):
            raise ValueError('point_data must be a DataFrame and/or grid_data must be a Dataset')
        pass


    def _grid_to_grid(self, grid_res, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, digits=2, min_val=None):
        """
        Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). Returns an xarray Dataset with the 3 dimensions. Uses the scipy interpolation function called map_coordinates.

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
        out1 = interp2d.grid_to_grid(self.grid_data, 'time', 'x', 'y', 'precip', grid_res, self._grid_crs, to_crs, bbox, order, extrapolation, fill_val, digits, min_val)
        return out1

    def _grid_to_points(self, point_data, to_crs=None, bbox=None, order=3, digits=2, min_val=None):
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
        digits : int
            The number of digits to round the output.
        min_val : int, float, or None
            The minimum value for the results. All results below min_val will be assigned min_val.

        Returns
        -------
        DataFrame
        """
        out1 = interp2d.grid_to_points(self.grid_data, 'time', 'x', 'y', 'precip', point_data, self._grid_crs, to_crs, order, digits, min_val)
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
        out1 = interp2d.points_to_grid(self.point_data, 'time', 'x', 'y', 'precip', grid_res, self._point_crs, to_crs, bbox, method, extrapolation, fill_val, digits, min_val)
        return out1

    def _points_to_points(self, point_data, to_crs=None, method='linear', digits=2, min_val=None):
        """
        Function to interpolate regularly or irregularly spaced values over many time stamps. Each time stamp of spatial values are interpolated independently (2D interpolation as opposed to 3D interpolation). Returns an xarray Dataset with the 3 dimensions. Uses the scipy interpolation function called map_coordinates.

        Parameters
        ----------
        point_data : str or DataFrame
            Path to geometry file of points to be interpolated (e.g. shapefile). Can be any file type that fiona/gdal can open. It can also be a DataFrame with 'x' and 'y' columns and the crs must be the same as to_crs.
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        method : str
            The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'.
        digits : int
            The number of digits to round the output.
        min_val : int, float, or None
            The minimum value for the results. All results below min_val will be assigned min_val.

        Returns
        -------
        DataFrame
        """
        out1 = interp2d.points_to_points(self.point_data, 'time', 'x', 'y', 'precip', point_data, self._point_crs, to_crs, method, digits, min_val)
        return out1


    def _grid_interp_na(self, method='linear', min_val=None, inplace=True):
        """
        Function to fill nan's in the grid_data to make it complete. Necessary for grid_to_* functions.

        Parameters
        ----------
        method : str
            The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
        min_val : int, float, or None
            The minimum value for the results. All results below min_val will be assigned min_val.
        inplace : bool
            Should the new grid overwrite the grid_data from initialisation?

        Returns
        -------
        Xarray Dataset
        """
        new_grid = interp2d.grid_interp_na(self.grid_data, 'time', 'x', 'y', 'precip', method=method, min_val=min_val)

        if inplace:
            self.grid_data = new_grid
        else:
            return new_grid


    def _adjust_grid_from_points(self, grid_res=None, to_crs=None, order=2, method='linear', digits=2, min_val=0):
        """
        Method to adjust a grid by forcing it through point data.

        Parameters
        ----------
        grid_res : int, float, or None
            The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
        method : str
            The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
        digits : int
            The number of digits to round the output.
        min_val : int, float, or None
            The minimum value for the results. All results below min_val will be assigned min_val.

        Returns
        -------
        Dataset
        """
        ## Resample the grid if needed
        if isinstance(grid_res, (int, float)):
            grid1 = self.grid_to_grid(grid_res, to_crs=to_crs, order=order, digits=digits, min_val=min_val)
        else:
            grid1 = self.grid_data

        ## grid to points
        point_data = self.point_data.copy()
        sites1 = point_data[['x', 'y']].drop_duplicates().copy()

        pts1 = self.grid_to_points(sites1, to_crs=to_crs, order=order, digits=digits, min_val=min_val)

        ## Subtract the original points from the new points
        pts2 = pts1.dropna().reset_index().copy()
        pts2['x'] = pts2['x'].round().astype(int)
        pts2['y'] = pts2['y'].round().astype(int)

        # Convert crs if needed
        if (to_crs is not None) & (to_crs != self._point_crs):
            to_crs1 = Proj(CRS.from_user_input(to_crs))
            trans1 = Transformer.from_proj(self._point_crs, to_crs1)
            points = np.array(trans1.transform(*point_data[['x', 'y']].values.T))
            point_data['x'] = points[0]
            point_data['y'] = points[1]

        both1 = pd.merge(point_data, pts2.rename(columns={'precip': 'grid_precip'}), on=['x', 'y', 'time'])

        both1['ratio'] = both1['precip']/both1['grid_precip']
        both1.loc[both1['ratio'].isnull(), 'ratio'] = 0
        self.ratio_precip = both1

        ## Create the bounding box for the new grid
        min_p = sites1.min(0)
        max_p = sites1.max(0)

        min_lon = util.find_nearest(grid1.x, min_p[0])
        max_lon = util.find_nearest(grid1.x, max_p[0])
        min_lat = util.find_nearest(grid1.y, min_p[1])
        max_lat = util.find_nearest(grid1.y, max_p[1])

        ## Points to grid
        ratio_grid = interp2d.points_to_grid(both1[['time', 'x', 'y', 'ratio']], 'time', 'x', 'y', 'ratio', grid_res, to_crs, None, (min_lon, max_lon, min_lat, max_lat), method, 'nearest')
        val = ratio_grid.ratio.values
        val[val == np.inf] = 0

        ## Add original grid by diff grid
        grid2 = xr.merge([grid1, ratio_grid], join='inner')
        grid3 = grid2['precip'] * grid2['ratio']
        grid3.name = 'precip'
        if isinstance(min_val, (int, float)):
            grid3 = xr.where(grid3 < min_val, 0, grid3).copy()

        ## Return
        return grid3.to_dataset()


    def _validate_grid_from_points(self, test_perc=0.1, grid_res=None, to_crs=None, order=2, method='linear', digits=2, min_val=0):
        """
        Method to validate a grid created by the adjust_grid_from_points method.

        Parameters
        ----------
        grid_res : int, float, or None
            The resulting grid resolution in the unit of the final projection (usually meters or decimal degrees).
        to_crs : int or str or None
            The projection for the output data similar to from_crs.
        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5. An order of 1 is linear interpolation.
        method : str
            The scipy griddata interpolation method to be applied. Options are 'nearest', 'linear', and 'cubic'. See `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_ for more details.
        digits : int
            The number of digits to round the output.
        min_val : int, float, or None
            The minimum value for the results. All results below min_val will be assigned min_val.

        Returns
        -------
        DataFrame
        """
        ## Prepare input data
        point_data = self.point_data.copy()
        sites1 = point_data[['x', 'y']].drop_duplicates().reset_index(drop=True).copy()
        rem_sites = random.sample(range(len(sites1)), int(len(sites1)*test_perc))
        new_sites1 = sites1.loc[~sites1.index.isin(rem_sites)]
        new_point_data = pd.merge(point_data, new_sites1, on=['x', 'y'])

        ## Create grid from grid and points
        self.point_data = new_point_data
        grid_grid = self.adjust_grid_from_points(grid_res, to_crs, order, method, digits, min_val)
        self.point_data = point_data

        ## Create points from new points
        rem_sites1 = sites1.loc[sites1.index.isin(rem_sites)]
        rem_point_data = pd.merge(point_data, rem_sites1, on=['x', 'y'])

        points_points = interp2d.points_to_points(new_point_data, 'time', 'x', 'y', 'precip', rem_sites1, to_crs, to_crs, method, digits, min_val).rename(columns={'precip': 'point_precip'})

        ## Extract and compare
        grid_points = interp2d.grid_to_points(grid_grid, 'time', 'x', 'y', 'precip', rem_sites1, to_crs, to_crs, 2, digits, min_val).rename(columns={'precip': 'grid_precip'})

        comp_data1 = pd.merge(rem_point_data, points_points.reset_index(), on=['time', 'x', 'y'])
        comp_data2 = pd.merge(comp_data1, grid_points.reset_index(), on=['time', 'x', 'y']).set_index(['time', 'x', 'y']).sort_index()

        return comp_data2
