# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
#from pycrs import parse
#
##########################################
#### Dictionaries for convert_crs
#
#proj4_netcdf_var = {'x_0': 'false_easting', 'y_0': 'false_northing', 'f': 'inverse_flattening',
#                    'lat_0': 'latitude_of_projection_origin',
#                    'lon_0': ('longitude_of_central_meridian', 'longitude_of_projection_origin'),
#                    'pm': 'longitude_of_prime_meridian',
#                    'k_0': ('scale_factor_at_central_meridian', 'scale_factor_at_projection_origin'),
#                    'a': 'semi_major_axis', 'b': 'semi_minor_axis', 'lat_1': 'standard_parallel',
#                    'proj': 'transform_name'}
#
#proj4_netcdf_name = {'aea': 'albers_conical_equal_area', 'tmerc': 'transverse_mercator',
#                     'aeqd': 'azimuthal_equidistant', 'laea': 'lambert_azimuthal_equal_area',
#                     'lcc': 'lambert_conformal_conic', 'cea': 'lambert_cylindrical_equal_area',
#                     'longlat': 'latitude_longitude', 'merc': 'mercator', 'ortho': 'orthographic',
#                     'ups': 'polar_stereographic', 'stere': 'stereographic', 'geos': 'vertical_perspective'}
#
#########################################
#### Functions


def grid_xy_to_map_coords(xy, digits=2, dtype=int, copy=True):
    """
    Convert an array of gridded x and y values to map_coordinates appropriate values.

    Parameters
    ----------
    xy : ndarray
        of shape (l, 2) where l is the number of location pairs. The first value of each pair is x and the second is y.
    digits : int
        The resolution of the data in digits.
    copy : bool
        Should it make a copy or modify the array in place?

    Returns
    -------
    ndarray
        Transposed shape from the input xy to an index array.
    dxy
        The grid interval.
    x_min
        The x value at index 0.
    y_min
        The y value at index 0.
    """
    if copy:
        xy1 = (xy.T * 10**digits).copy()
    else:
       xy1 = (xy.T * 10**digits)
    x_min, y_min = xy1.min(1)
    dxy = int(np.median(np.diff(xy1[0])))
    xy1[0] = ((xy1[0] - x_min)/dxy)
    xy1[1] = ((xy1[1] - y_min)/dxy)

    if dtype == int:
        xy1 = np.rint(xy1).astype(int)

    return xy1, dxy/10**digits, x_min/10**digits, y_min/10**digits


def point_xy_to_map_coords(xy, dxy, x_min, y_min, dtype=int, copy=True):
    """
    Convert an array of irregular x and y values to map_coordinates appropriate values.

    Parameters
    ----------
    xy : ndarray
        of shape (l, 2) where l is the number of location pairs. The first value of each pair is x and the second is y.
    dxy
        The grid interval.
    x_min
        The x value at index 0.
    y_min
        The y value at index 0.
    copy : bool
        Should it make a copy or modify the array in place?

    Returns
    -------
    ndarray
        Transposed shape from the input xy to an index array.
    """
    if copy:
        xy1 = xy.T.copy()
    else:
        xy1 = xy.T

    x = ((xy1[0] - x_min)/dxy)
    y = ((xy1[1] - y_min)/dxy)
    xy1[0] = y
    xy1[1] = x

    if dtype == int:
        xy1 = np.rint(xy1).astype(int)

    return xy1


def map_coords_to_xy(coords, dxy, x_min, y_min, copy=True):
    """
    The reverse of point_xy_to_map_coords.
    """
    if copy:
        coords1 = coords.copy().astype(float)
    else:
        coords1 = coords.astype(float)
    coords1[0] = coords1[0]*dxy + x_min
    coords1[1] = coords1[1]*dxy + y_min

    return coords1.T


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#def convert_crs(from_crs, crs_type='proj4', pass_str=False):
#    """
#    Convenience function to convert one crs format to another.
#
#    Parameters
#    ----------
#    from_crs: int or str
#        The crs as either an epsg number or a str in a common crs format (e.g. proj4 or wkt).
#    crs_type: str
#        Output format type of the crs ('proj4', 'wkt', 'proj4_dict', or 'netcdf_dict').
#    pass_str: str
#        If input is a str, should it be passed though without conversion?
#
#    Returns
#    -------
#    str or dict
#    """
#
#    ### Load in crs
#    if all([pass_str, isinstance(from_crs, str)]):
#        crs2 = from_crs
#    else:
#        if isinstance(from_crs, int):
#            crs1 = parse.from_epsg_code(from_crs)
#        elif isinstance(from_crs, str):
#            crs1 = parse.from_unknown_text(from_crs)
#        else:
#            raise  ValueError('from_crs must be an int or str')
#
#        ### Convert to standard formats
#        if crs_type == 'proj4':
#            crs2 = crs1.to_proj4()
#        elif crs_type == 'wkt':
#            crs2 = crs1.to_ogc_wkt()
#        elif crs_type in ['proj4_dict', 'netcdf_dict']:
#            crs1a = crs1.to_proj4()
#            crs1b = crs1a.replace('+', '').split()[:-1]
#            crs1c = dict(i.split('=') for i in crs1b)
#            crs2 = dict((i, float(crs1c[i])) for i in crs1c)
#        else:
#            raise ValueError('Select one of "proj4", "wkt", "proj4_dict", or "netcdf_dict"')
#        if crs_type == 'netcdf_dict':
#            crs3 = {}
#            for i in crs2:
#                if i in proj4_netcdf_var.keys():
#                    t1 = proj4_netcdf_var[i]
#                    if isinstance(t1, tuple):
#                        crs3.update({j: crs2[i] for j in t1})
#                    else:
#                        crs3.update({proj4_netcdf_var[i]: crs2[i]})
#            if crs3['transform_name'] in proj4_netcdf_name.keys():
#                gmn = proj4_netcdf_name[crs3['transform_name']]
#                crs3.update({'transform_name': gmn})
#            else:
#                raise ValueError('No appropriate netcdf projection.')
#            crs2 = crs3
#
#    return crs2
