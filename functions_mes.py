import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
import os.path
import glob


"""
MESSENGER DATA PATH
"""


mes_path='/Volumes/External/data/mes/'
kernels_path='/Volumes/External/data/kernels/'


def get_mesmag(fp):
    cdf = pycdf.CDF(fp)
    data = {
        df_col: cdf[cdf_col][:]
        for cdf_col, df_col in zip(['Epoch','B_radial', 'B_tangential', 'B_normal'],
                                   ['timestamp', 'b_x', 'b_y', 'b_z'])
    }
    df = pd.DataFrame.from_dict(data)
    df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    return df


def get_mesmag_range(start_timestamp, end_timestamp, path):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'messenger_mag_rtn_{date_str}_v01.cdf'
        _df = get_mesmag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df



"""
MESSENGER POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def mes_furnish():
    """Main"""
    mes_path = kernels_path+'mes/'
    generic_path = kernels_path+'generic/'
    mes_kernels = os.listdir(mes_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in mes_kernels:
        spiceypy.furnsh(os.path.join(mes_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


# def furnish():
#     """Main"""
#     base = r"kernels\mes\*"
#     kernels = glob.glob(base)
#     for kernel in kernels:
#         spiceypy.furnsh(kernel)


def get_mes_pos(t, prefurnished=False):
    if not prefurnished: 
        if spiceypy.ktotal('ALL') < 1:
            mes_furnish()
    try:
        pos = spiceypy.spkpos("MESSENGER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [None, None, None, None]


def get_mes_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_mes_pos(t)
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


def get_mes_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_mes_pos(t)
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


def get_mes_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_mes_pos(t)
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



def get_mes_positions(start, end):
    if spiceypy.ktotal('ALL') < 1:
        mes_furnish()
    t = start
    positions = []
    while t < end:
        mes_pos = spiceypy.spkpos("MESSENGER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(mes_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(mes_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(mes_pos[1], mes_pos[0]) * 360 / 2 / np.pi
        positions.append([t, mes_pos, r_au, lat, lon])
        t += timedelta(days=1)
    return positions


def transform_data():
    pass