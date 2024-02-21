import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
# import os
import glob
import urllib.request
import os.path


"""
SOLAR ORBITER SERVER DATA PATH
"""

solo_path='/Volumes/External/data/solo/'
kernels_path='/Volumes/External/data/kernels/'


"""
SOLO BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
SOLO MAG DATA
# Potentially different SolO MAG file names: internal low latency, formagonly, and formagonly 1 min.
e.g.
- Internal files: solo_L2_mag-rtn-ll-internal_20230225_V00.cdf
- For MAG only files: solo_L2_mag-rtn-normal-formagonly_20200415_V01.cdf
- For MAG only 1 minute res files: solo_L2_mag-rtn-normal-1-minute-formagonly_20200419_V01.cdf
# All in RTN coords
# Should all follow same variable names within cdf
"""


#DOWNLOAD FUNCTIONS for 1min or 1sec data


def download_solomag_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1min'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_mag-rtn-normal-1-minute_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_solomag_1sec(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1sec'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_mag-rtn-normal_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#LOAD FUNCTIONS for MAG data 


#Load single file from specific path
def get_solomag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['EPOCH'], ['timestamp'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['B_RTN'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_solomag_range_formagonly_internal(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/ll'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-ll-internal_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_solomag_range_formagonly(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/formagonly/full'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal-formagonly_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_solomag_range_formagonly_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/formagonly/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal-1-minute-formagonly_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_solomag_range_1sec(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1sec'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal_{date_str}*')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_solomag_range_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/solo_L2_mag-rtn-normal-1-minute_{date_str}.cdf'
        _df = get_solomag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


"""
SOLO PLASMA DATA
# Level 2 science SWA PLAS grnd moment data
"""


#DOWNLOAD FUNCTION for swa/plas data
def download_soloplas(start_timestamp, end_timestamp, path=f'{solo_path}'+'swa/plas/l2'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_swa-pas-grnd-mom_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#Load single file from specific path
def get_soloplas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'N', 'T'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        vx, vy, vz = cdf['V_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_soloplas_range(start_timestamp, end_timestamp, path=f'{solo_path}'+'swa/plas/l2'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_swa-pas-grnd-mom_{date_str}*')
        _df = get_soloplas(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df