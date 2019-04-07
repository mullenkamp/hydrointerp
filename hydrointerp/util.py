# -*- coding: utf-8 -*-
"""
Utility functions.
"""
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
#
#
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
