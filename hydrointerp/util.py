# -*- coding: utf-8 -*-
"""
Utility functions.
"""
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

