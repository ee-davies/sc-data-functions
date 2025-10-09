import numpy as np
import pandas as pd
from datetime import timedelta
import spiceypy
import os.path

from functions_general import load_path


"""
VENUS EXPRESS DATA PATH
"""

vex_path=load_path(path_name='vex_path')
print(f"VEX path loaded: {vex_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
VEX POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def vex_furnish():
    """Main"""
    vex_path = kernels_path+'vex/'
    generic_path = kernels_path+'generic/'
    vex_kernels = os.listdir(vex_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in vex_kernels:
        spiceypy.furnsh(os.path.join(vex_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_vex_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        vex_furnish()
    try:
        pos = spiceypy.spkpos("VENUS EXPRESS", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [t, None, None, None, None, None, None]


def get_vex_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_vex_pos(t)
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


def get_vex_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_vex_pos(t)
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


def get_vex_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_vex_pos(t)
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


"""
OLD FUNCTIONS (NEED TO MODIFY)
"""

def get_vexmag(fp):
    """Reformats a .TAB file to .csv equivalent."""
    if not fp.endswith('.TAB'):
        raise Exception('Wrong filetype passed, must end with .TAB...')
    cols = ['Timestamp', 'B_R', 'B_T', 'B_N', 'B_TOT', 'X_POS', 'Y_POS', 'Z_POS', 'R_POS']
    i = 0  # instantiate
    i_stop = 500  # initial
    check_table = True
    data = []
    try:
        with open(fp, 'r') as f:
            lines = f.readlines()
            while i < i_stop:
                if check_table:
                    if lines[i].startswith('^TABLE'):
                        i_stop = int(lines[i].split('=')[-1].strip()) - 1
                        check_table = False
                i += 1
            for line in lines[i:]:
                data.append(line.split())
        df = pd.DataFrame(data, columns=cols)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        for col in cols[1:]:
            df[col] = df[col].astype('float32')
        df['B_R'] = -1 * df['B_R']
        df['B_T'] = -1 * df['B_T']
        df = filter_bad_data(df, 'B_TOT', 9.99e+04)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_vexmag_range(start_timestamp, end_timestamp, path=r'D:/VEX'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        doy = start.strftime('%j')
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'MAG_{date_str}_DOY{doy}_S004_V1.TAB'
        _df = get_vexmag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df