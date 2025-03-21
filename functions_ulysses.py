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


"""
ULYSSES MAG DATA
# obtained via https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/mag_cdaweb/vhm_1min/
# cdf files available in 1 min, 1 sec, m1
"""


#DOWNLOAD FUNCTIONS


def download_ulyssesmag_1min(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'mag/l2/1min'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'uy_1min_vhm_{date_str}_v01'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/mag_cdaweb/vhm_1min/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#Load single file from specific path using pycdf from spacepy
def get_ulyssesmag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'B_MAG'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['B_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        #df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_ulyssesmag_range(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'uy_1min_vhm_{date_str}_v01'
        fn = f'{path}/{data_item_id}.cdf'
        _df = get_ulyssesmag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
ULYSSES PLASMA DATA
# L2 plasma moments from SWOOPS instrument
"""


#DOWNLOAD FUNCTIONS

#all plasma files are yearly i.e. 19920101, except 19901118
def download_ulyssesplas(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'plas/l2'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        if year == 1990:
            date_str = f'{year}{start.month:02}{start.day:02}'
        else:
            date_str = f'{year}0101'
        data_item_id = f'uy_proton-moments_swoops_{date_str}_v01'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            if year == 1990:
                start += timedelta(days=1)
            else:
                start += timedelta(days=365.25)
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/plasma/swoops_cdaweb/proton-moments_swoops/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                if year == 1990:
                    start += timedelta(days=1)
                else:
                    start += timedelta(days=365.25)
            except Exception as e:
                print('ERROR', e, data_item_id)
                if year == 1990:
                    start += timedelta(days=1)
                else:
                    start += timedelta(days=365.25)


#Load single file from specific path using pycdf from spacepy
#plasma files also include mag data and heliocentricDistance and lat if needed
#need to assess temperature
def get_ulyssesplas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'V_MAG', 'VR', 'VT', 'VN', 'dens'], ['time', 'vt', 'vx', 'vy', 'vz', 'np'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        # t_par = cdf['Tpar'][:]
        # t_per = cdf['Tper'][:]
        # tp = np.sqrt(t_par**2 + t_per**2)
        # df['tp'] = tp
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df