# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import pandas as pd


def tsreg(ts, freq=None, interp=None, maxgap=None, **kwargs):
    """
    Function to regularize a time series DataFrame.
    The first three indeces must be regular for freq=None!!!

    Parameters
    ----------
    ts : DataFrame
        DataFrame with a time series index.
    freq : str
        Either specify the known frequency of the data or use None and
    determine the frequency from the first three indices.
    interp : str or None
        Either None if no interpolation should be performed or a string of the interpolation method.
    **kwargs
        kwargs passed to interpolate.
    """

    if freq is None:
        freq = pd.infer_freq(ts.index[:3])
    ts1 = ts.asfreq(freq)
    if isinstance(interp, str):
        ts1 = ts1.interpolate(interp, limit=maxgap, **kwargs)

    return ts1


def pd_grouby_fun(df, fun_name):
    """
    Function to make a function specifically to be used on pandas groupby objects from a string code of the associated function.
    """
    if type(df) == pd.Series:
        fun1 = pd.core.groupby.SeriesGroupBy.__dict__[fun_name]
    elif type(df) == pd.DataFrame:
        fun1 = pd.core.groupby.GroupBy.__dict__[fun_name]
    else:
        raise ValueError('df should be either a Series or DataFrame.')
    return fun1


def grp_ts_agg(df, grp_col, ts_col, freq_code, closed='left', label='left', discrete=False):
    """
    Simple function to aggregate time series with dataframes with a single column of sites and a column of times.

    Parameters
    ----------
    df : DataFrame
        Dataframe with a datetime column.
    grp_col : str or list of str
        Column name(s) to group by (in addition to the datetime column).
    ts_col : str
        The column name of the datetime column.
    freq_code : str
        The pandas frequency code for the aggregation (e.g. 'M', 'A-JUN').
    closed : str
        Closed end of interval. 'left' or 'right'.
    label :str
        Interval boundary to use for labeling. 'left' or 'right'
    discrete : bool
        Is the data discrete? Will use proper resampling using linear interpolation.

    Returns
    -------
    Pandas resample object
    """

    df1 = df.copy()
    if df[ts_col].dtype.type != np.datetime64:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
        except:
            raise TypeError('The ts_col must be a pandas DateTime column or a string that can be easily coerced to one.')
    df1.set_index(ts_col, inplace=True)
    if isinstance(grp_col, str):
        grp_col = [grp_col]
    else:
        grp_col = grp_col[:]
    if discrete:
        val_cols = [c for c in df1.columns if c not in grp_col]
        df1[val_cols] = (df1[val_cols] + df1[val_cols].shift(-1))/2
    grp_col.extend([pd.Grouper(freq=freq_code, closed=closed, label=label)])
    df_grp = df1.groupby(grp_col)

    return df_grp



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
    y_min, x_min = xy1.min(1)
    dxy = int(np.median(np.diff(xy1[0])))
    xy1[0] = ((xy1[0] - y_min)/dxy)
    xy1[1] = ((xy1[1] - x_min)/dxy)

    if dtype == int:
        xy1 = np.rint(xy1).astype(int)

    return (xy1, dxy/10**digits, x_min/10**digits, y_min/10**digits)


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
    xy1[0] = ((xy1[0] - y_min)/dxy)
    xy1[1] = ((xy1[1] - x_min)/dxy)

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

















































