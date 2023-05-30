import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
# import spiceypy
# import os
import glob
import urllib.request
import os.path


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def download_pspmag_1min(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/mag/1_min"):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_fld_l2_mag_rtn_1min_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_pspmag_full(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/mag/full"):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            data_item_id = f'psp_fld_l2_mag_rtn_{date_str}{t:02}_v02'
            if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
                print(f'{data_item_id}.cdf has already been downloaded.')
            else:
                try:
                    data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/{year}/{data_item_id}.cdf'
                    urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                    print(f'Successfully downloaded {data_item_id}.cdf')
                except Exception as e:
                    print('ERROR', e, data_item_id)
        start += timedelta(days=1)        
        

def get_pspmag_1min(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN_1min'], ['timestamp'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN_1min'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspmag_full(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN'], ['timestamp'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspmag_range_1min(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/mag/1_min"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_fld_l2_mag_rtn_1min_{date_str}_v02.cdf'
        _df = get_pspmag_1min(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_pspmag_range_full(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/mag/full"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            fn = f'{path}/psp_fld_l2_mag_rtn_{date_str}{t:02}_v02.cdf'
            _df = get_pspmag_full(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def download_pspplas(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spc/l3i"):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_swp_spc_l3i_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def get_pspspc_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_moment', 'wp_moment'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_moment_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_fit', 'wp_fit'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_fit_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit1(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np1_fit', 'wp1_fit'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp1_fit_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspi_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'DENS', 'TEMP'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['VEL_RTN_SUN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_range_mom(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spc/l3i"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'temperature', -1E30)
    filter_bad_data(df, 'density', -1E30)
    filter_bad_data(df, 'v_bulk', -1E30)
    filter_bad_data(df, 'v_x', -1E30)
    filter_bad_data(df, 'v_y', -1E30)
    filter_bad_data(df, 'v_z', -1E30)
    return df


def get_pspspc_range_fit(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spc/l3i"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_fit(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'temperature', -1E30)
    filter_bad_data(df, 'density', -1E30)
    filter_bad_data(df, 'v_bulk', -1E30)
    filter_bad_data(df, 'v_x', -1E30)
    filter_bad_data(df, 'v_y', -1E30)
    filter_bad_data(df, 'v_z', -1E30)
    return df


def get_pspspi_range_mom(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spi/sf00/mom"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spi_sf00_l3_mom_{date_str}_v04.cdf'
        _df = get_pspspi_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df