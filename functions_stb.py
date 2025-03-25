import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spiceypy
import os.path
import glob
import urllib.request
from urllib.request import urlopen
import json
import astrospice
from sunpy.coordinates import HeliocentricInertial, HeliographicStonyhurst
from bs4 import BeautifulSoup
import cdflib
import pickle
from spacepy import pycdf


"""
STEREO-B DATA PATH
"""


stereob_path='/Volumes/External/data/stereob/'
kernels_path='/Volumes/External/data/kernels/'


"""
STEREO-B BAD DATA FILTER
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
STEREO-B MAG AND PLAS DATA 
# Option to load in merged mag and plas data files
# Can also load separate MAG and PLAS beacon data files for real-time use
"""


def get_stereomag(fp):
    cdf = pycdf.CDF(fp)
    data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'BTOTAL'], ['Timestamp', 'B_TOT'])}
    df = pd.DataFrame.from_dict(data)
    bx, by, bz = cdf['BFIELDRTN'][:].T
    df['B_R'] = bx
    df['B_T'] = by
    df['B_N'] = bz
    return filter_bad_data(df, 'B_TOT', -9.99e+29)


# def get_stereoplas(fp):
#     cdf = pycdf.CDF(fp)
#     cols_raw = ['Epoch', 'Vp_RTN', 'Vr_Over_V_RTN', 'Vt_Over_V_RTN', 'Vn_Over_V_RTN', 'Tp', 'Np']
#     cols_new = ['Timestamp', 'v_bulk', 'v_x', 'v_y', 'v_z', 'v_therm', 'density']
#     data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
#     df = pd.DataFrame.from_dict(data)
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     for col in cols_new[1:]:
#         df[col] = df[col].astype('float32')
#     return filter_bad_data(df, 'v_bulk', -9.99e+04)


def get_stereoplas(fp):
    cdf = pycdf.CDF(fp)
    cols_raw = ['epoch', 'proton_bulk_speed', 'proton_Vr_RTN', 'proton_Vt_RTN', 'proton_Vn_RTN', 'proton_temperature', 'proton_number_density']
    cols_new = ['Timestamp', 'v_bulk', 'v_x', 'v_y', 'v_z', 'v_therm', 'density']
    data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
    df = pd.DataFrame.from_dict(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in cols_new[1:]:
        df[col] = df[col].astype('float32')
    return filter_bad_data(df, 'v_bulk', -9.99e+04)


def get_stereobmag_range(start_timestamp, end_timestamp, path=stereob_path):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start < end:
        year = start.year
        fn = f'stb_l2_magplasma_1m_{year}0101_v01.cdf'
        _df = get_stereomag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=366)
    return df



# def get_stereobplas_range(start_timestamp, end_timestamp, path=r'D:/STEREO_B'):
#     """Pass two datetime objects and grab .STS files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start < end:
#         year = start.year
#         fn = f'stb_l2_magplasma_1m_{year}0101_v01.cdf'
#         _df = get_stereoplas(f'{path}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=366)
#     return df


def get_stereobplas_range(start_timestamp, end_timestamp, path=stereob_path):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'stb_l2_pla_1dmax_1min_{date_str}_v09.cdf'
        _df = get_stereoplas(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


"""
STEREO B POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
kernels from https://soho.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/depm/behind/ 
and https://soho.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/epm/behind/ for predicted orbit kernel
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def stereob_furnish():
    """Main"""
    stereob_path = kernels_path+'stereob/'
    generic_path = kernels_path+'generic/'
    stereob_kernels = os.listdir(stereob_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in stereob_kernels:
        spiceypy.furnsh(os.path.join(stereob_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_stb_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        stereob_furnish()
    pos = spiceypy.spkpos("STEREO BEHIND", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_stb_positions(time_series):
    positions = []
    for t in time_series:
        position = stereob_furnish(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_stb_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_stb_pos(t)
        positions.append(position)
        t += timedelta(days=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions


def get_stb_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_stb_pos(t)
        positions.append(position)
        t += timedelta(hours=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions


def get_stb_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_stb_pos(t)
        positions.append(position)
        t += timedelta(minutes=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions