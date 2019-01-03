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
        Column name that contains the sites.
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
