import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
# import os
import glob
import urllib.request
import os.path
import pickle


"""
WIND  SERVER DATA PATH
"""

wind_path='/Volumes/External/data/wind/'




def format_path(fp):
    """Formatting required for CDF package."""
    return fp.replace('/', '\\')


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
WIND DATA PATH
"""

wind_path='/Volumes/External/data/wind/'


def get_windmag(fp):
    """raw = gse"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:,0] for cdf_col, df_col in zip(['Epoch', 'BF1'], ['timestamp', 'b_tot'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSE'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df = filter_bad_data(df, 'b_tot', -9.99e+30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


# def get_windmag_range(start_timestamp, end_timestamp, wind_path):
#     """Pass two datetime objects and grab .STS files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date() + timedelta(days=1)
#     while start < end:
#         year = start.year
#         date_str = f'{year}{start.month:02}{start.day:02}'
#         if year < 2020:
#             fn = f'wi_h0_mfi_{date_str}_v05.cdf'
#         else:
#             fn = f'wi_h0_mfi_{date_str}_v04.cdf'
#         _df = get_windmag(f'{path}/{year}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=1)
#     return df


def get_windmag_rtn(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'BF1'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BRTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df = filter_bad_data(df, 'bt', -9.99e+30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_windmag_rtn_range(start_timestamp, end_timestamp, path=wind_path+'mfi/rtn'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'wi_h3-rtn_mfi_{date_str}_v05.cdf'
        _df = get_windmag_rtn(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


# def get_wind3dp_pm(fp):
#     cdf = pycdf.CDF(fp)
#     data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'P_DENS', 'P_TEMP'], ['timestamp', 'density', 'v_therm'])}
#     df = pd.DataFrame.from_dict(data)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     vx, vy, vz = cdf['P_VELS'][:].T #proton velocity vector is in GSE coordinates
#     df['v_x'] = vx
#     df['v_y'] = vy
#     df['v_z'] = vz
#     df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
#     # for col in cols_new[1:]:
#     #     df[col] = df[col].astype('float32')
#     return df


# def get_wind3dp_pm_range(start_timestamp, end_timestamp, path):
#     """Pass two datetime objects and grab .cdf files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start <= end:
#         year = start.year
#         month = start.month
#         day = start.day
#         fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v05.cdf'
#         #print(f'{path}/{fn}')
#         _df = get_wind3dp_pm(f'{path}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=1)
#     return df


def get_windswe(fp):
    try:
        cdf = pycdf.CDF(fp)
        cols_raw = ['Epoch', 'Proton_V_nonlin', 'Proton_VX_nonlin', 'Proton_VY_nonlin',
                    'Proton_VZ_nonlin', 'Proton_W_nonlin', 'Proton_Np_nonlin']
        cols_new = ['time', 'vt', 'vx', 'vy', 'vz', 'tp', 'np']
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        for col in cols_new[1:]:
            df[col] = df[col].astype('float32')
        df = filter_bad_data(df, 'vt', 9.99e+04)
        df = filter_bad_data(df, 'tp', 9.99e+04) #called tp but actually is v_therm
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_windswe_range(start_timestamp, end_timestamp, path=wind_path+'swe'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        year = start.year
        month = start.month
        day = start.day
        fn = f'wi_h1_swe_{year}{month:02}{day:02}_v01.cdf'
        _df = get_windswe(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


#### WIND POSITIONS


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def get_windorbit_hec(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        x, y, z = cdf['HEC_POS'][:].T
        df['x'] = x
        df['y'] = y
        df['z'] = z
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_windorbit_hec_range(start_timestamp, end_timestamp, path=wind_path+'orbit/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df=None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(path+f'wi_or_pre_{date_str}_*')[0]
            _df = get_windorbit_hec(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    r, theta, phi = cart2sphere(df['x'],df['y'],df['z'])
    df['r'] = r
    df['lat'] = theta
    df['lon'] = phi
    return df


# def transform_data(df, instrument, coord_system):
#     #TODO: apply logic to handle b and v data transformations here
#     if instrument == 'mag':
#         prefix = 'b'
#     elif instrument == 'swe' or instrument == '3dp_pm':
#         prefix = 'v'
#     if coord_system == 'rtn_approx':
#         df[f'{prefix}_x'] = -1 * df[f'{prefix}_x']
#         df[f'{prefix}_y'] = -1 * df[f'{prefix}_y']
#     else:
#         raise ValueError('transform does not exist yet, rtn_approx does exist')
#     return df