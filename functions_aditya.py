import numpy as np
import pandas as pd
from datetime import timedelta
import spiceypy
from spacepy import pycdf
# import os
import glob
import os.path
import netCDF4 as nc

from functions_general import load_path


"""
ADITYA L1 SERVER DATA PATH
"""

aditya_path=load_path(path_name='aditya_path')
print(f"Aditya path loaded: {aditya_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
FILTER BAD DATA
"""


def filter_bad_col(df, col, bad_val): #filter by individual columns
    if bad_val < 0:
        mask_vals = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask_vals = df[col] > bad_val  # boolean mask for all bad values
    df[col].mask(mask_vals, inplace=True)
    return df


"""
ADITYA DOWNLOAD DATA
#Can't currently download automatically as requires log in
MAG: MAG data available from 20240701
"""


#MAG DATA: https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=mag
# FORMAT example: /al1/protected/downloadData/mag/level2/2025/08/13/L2_AL1_MAG_20250813_V00.nc?mag

# def download_adityamag(start_timestamp, end_timestamp, path=f'{aditya_path}'+'mag/'):
#     start = start_timestamp.date()
#     end = end_timestamp.date() + timedelta(days=1)
#     while start < end:
#         year = start.year
#         date_str = f'{year}{start.month:02}{start.day:02}'
#         data_item_id = f'L2_AL1_MAG_{date_str}_V00'
#         if os.path.isfile(f"{path}/{data_item_id}.nc") == True:
#             print(f'{data_item_id}.nc has already been downloaded.')
#             start += timedelta(days=1)
#         else:
#             try:
#                 data_url = f'https://pradan1.issdc.gov.in/al1/protected/downloadData/mag/level2/{year}/{start.month:02}/{start.day:02}/{data_item_id}.nc?mag'
#                 urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.nc")
#                 print(f'Successfully downloaded {data_item_id}.nc')
#                 start += timedelta(days=1)
#             except Exception as e:
#                 print('ERROR', e, data_item_id)
#                 start += timedelta(days=1)


"""
ADITYA MAG DATA
MAG: MAG data available from 20240701
"""

def get_adityamag_gse(fp):
    """raw = gse"""
    try:
        ncdf = nc.Dataset(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'Bx_gse', 'By_gse', 'Bz_gse'], ['time', 'bx', 'by', 'bz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time']*1E9)
        df = filter_bad_col(df, 'bx', -9000)
        df = filter_bad_col(df, 'by', -9000)
        df = filter_bad_col(df, 'bz', -9000)
        bt = np.linalg.norm([df.bx,df.by,df.bz], axis=0)
        df['bt'] = bt
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_adityamag_gse_range(start_timestamp, end_timestamp, path=aditya_path+'mag'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'L2_AL1_MAG_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.nc')[0]
        except Exception as e:
            path_fn = None
        _df = get_adityamag_gse(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_adityamag_gsm(fp):
    """raw = gse"""
    try:
        ncdf = nc.Dataset(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'Bx_gsm', 'By_gsm', 'Bz_gsm'], ['time', 'bx', 'by', 'bz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time']*1E9)
        bt = np.linalg.norm([df.bx,df.by,df.bz], axis=0)
        df['bt'] = bt
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_adityamag_gsm_range(start_timestamp, end_timestamp, path=aditya_path+'mag'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'L2_AL1_MAG_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.nc')[0]
        except Exception as e:
            path_fn = None
        _df = get_adityamag_gsm(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_adityamag_range(start_timestamp, end_timestamp, coord_sys:str):
    if coord_sys == 'GSE':
        df = get_adityamag_gse_range(start_timestamp, end_timestamp)
    elif coord_sys == 'GSM':
        df = get_adityamag_gsm_range(start_timestamp, end_timestamp)
    return df


"""
ADITYA PLAS DATA
ASPEX-SWIS: Data available from 20240507, but solidly from 20240801
"""

def get_adityaplas(fp):
    """raw = likely GSE"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_for_cdf_mod', 'proton_density', 'proton_thermal', 'proton_bulk_speed', 'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity'], ['time', 'np', 'tp', 'vt', 'vx', 'vy', 'vz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        df = filter_bad_col(df, 'np', -1E30)
        df = filter_bad_col(df, 'tp', -1E30)
        df = filter_bad_col(df, 'vt', -1E30)
        df = filter_bad_col(df, 'vx', -1E30)
        df = filter_bad_col(df, 'vy', -1E30)
        df = filter_bad_col(df, 'vz', -1E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_adityaplas_range(start_timestamp, end_timestamp, path=aditya_path+'plas'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'AL1_ASW91_L2_BLK_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_adityaplas(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
ADITYA POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=spice
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def aditya_furnish():
    """Main"""
    aditya_path = kernels_path+'aditya/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(aditya_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(aditya_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_aditya_pos(t, coord_sys='ECLIPJ2000'): 
    if spiceypy.ktotal('ALL') < 1:
        aditya_furnish()
    if coord_sys == 'GSE':
        try:
            pos = spiceypy.spkpos("ADITYA", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "EARTH")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
    else:
        try:
            pos = spiceypy.spkpos("ADITYA", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "SUN")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
        

def get_aditya_positions(time_series, coord_sys):
    positions = []
    for t in time_series:
        position = get_aditya_pos(t, coord_sys)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_aditya_pos_from_mag(fp, coord_sys='GSE'): #GSE and GSM available
    if coord_sys == 'GSE':
        coord_sys = 'gse'
    elif coord_sys == 'GSM':
        coord_sys = 'gsm'
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', f'x_{coord_sys}', f'y_{coord_sys}', f'z_{coord_sys}'], ['time', 'x', 'y', 'z'])}
        df = pd.DataFrame.from_dict(data)
        r, lat, lon = cart2sphere(df.x,df.y,df.z)
        df['r'] = r
        df['lat'] = lat
        df['lon'] = lon
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df