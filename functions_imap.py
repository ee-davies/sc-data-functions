import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
# import os
import urllib.request
import os.path
import pickle
import glob

from functions_general import load_path


"""
IMAP SERVER DATA PATH
"""

imap_path=load_path(path_name='imap_path')
print(f"IMAP path loaded: {imap_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
IMAP DATA: FROM ARCHIVE
"""


def get_imapmag(fp, coord_sys:str):
    """raw = gse,gsm,rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['mag_epoch', 'mag_B_magnitude'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf[f'mag_B_{coord_sys}'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_imapplas(fp):
    """raw = magnitudes"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['swapi_epoch', 'swapi_pseudo_proton_speed', 'swapi_pseudo_proton_density', 'swapi_pseudo_proton_temperature'], ['time', 'vt', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_imapmag_range(start_timestamp, end_timestamp, coord_sys:str, path=f'{imap_path}'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}ialirt/archive/imap_ialirt_l1_realtime_{date_str}*.cdf')[0]
            _df = get_imapmag(fn, coord_sys)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
                print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


def get_imapplas_range(start_timestamp, end_timestamp, path=f'{imap_path}'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}ialirt/archive/imap_ialirt_l1_realtime_{date_str}*.cdf')[0]
            _df = get_imapplas(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
                print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


"""
COMBINED IMAP MAG AND PLAS
"""


def get_imapmagplas_range(start_timestamp, end_timestamp, coord_sys:str):
    df_mag = get_imapmag_range(start_timestamp, end_timestamp, coord_sys)
    if df_mag is None:
        print(f'IMAP MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        
    df_plas = get_imapplas_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'IMAP SWAPI data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'np':[], 'tp':[]})
        plas_rdf = df_plas
    else:
        plas_rdf = df_plas.set_index('time').resample('1min').mean().reset_index(drop=False)
        plas_rdf.set_index(pd.to_datetime(plas_rdf['time']), inplace=True)
        if mag_rdf.shape[0] != 0:
            plas_rdf = plas_rdf.drop(columns=['time'])

    magplas_rdf = pd.concat([mag_rdf, plas_rdf], axis=1)
    magplas_rdf = magplas_rdf.drop(columns=['time'])
    magplas_rdf['time'] = magplas_rdf.index
    magplas_rdf = magplas_rdf.reset_index(drop=True)

    return magplas_rdf


"""
IMAP POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Coord_systems available: ECLIPJ2000, HEEQ, HEE, GSE
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def imap_furnish():
    """Main"""
    imap_path = kernels_path+'imap/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(imap_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(imap_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_imap_pos(t, coord_sys='ECLIPJ2000'): 
    if spiceypy.ktotal('ALL') < 1:
        imap_furnish()
    if coord_sys == 'GSE':
        try:
            pos = spiceypy.spkpos("IMAP", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "EARTH")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
    else:
        try:
            pos = spiceypy.spkpos("IMAP", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "SUN")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
        

def get_imap_positions(time_series, coord_sys):
    positions = []
    for t in time_series:
        position = get_imap_pos(t, coord_sys)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_imap_positions_daily(start, end, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_imap_pos(t, coord_sys)
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


def get_imap_positions_hourly(start, end, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_imap_pos(t, coord_sys)
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


def get_imap_positions_minute(start, end, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_imap_pos(t, coord_sys)
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

