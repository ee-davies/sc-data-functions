import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
import glob
import urllib.request
import os.path
import pickle


"""
ULYSSES SERVER DATA PATH
"""

ulysses_path='/Volumes/External/data/ulysses/'
kernels_path='/Volumes/External/data/kernels/'


"""
ULYSSES BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val): #filter across whole df
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def filter_bad_col(df, col, bad_val): #filter by individual columns
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    df[col][mask] = np.nan
    return df


