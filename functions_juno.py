import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
import os.path
import glob

def format_path(fp):
    """Formatting required for CDF package."""
    return fp.replace('/', '\\')


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'Timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
JUNO DATA PATH
"""

juno_path='/Volumes/External/data/juno/'
kernels_path='/Volumes/External/data/kernels/'


"""
JUNO MAG DATA
# Cruise phase FGM data, https://pds-ppi.igpp.ucla.edu/search/view/?f=yes&id=pds://PPI/JNO-SS-3-FGM-CAL-V1.0
# 1 min resolution, SE coordinates (equivalent to RTN)
# .sts files, not .cdf
"""


def get_junomag(fp):
    """Get data and return pd.DataFrame."""
    cols = ['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond',
            'Decimal Day', 'bx', 'by', 'bz', 'Range', 'POS_X', 'POS_Y', 'POS_Z']
    try:
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if sum(c.isalpha() for c in line) == 0:
                    break

        df = pd.read_csv(fp, skiprows=i, sep=r'\s+', names=cols)
        df['time'] = df[['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond']]\
            .apply(lambda x: datetime.strptime(' '.join(str(y) for y in x),
                                               r'%Y %j %H %M %S %f'), axis=1)
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    df.drop(columns = ['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond', 'Decimal Day', 'Range', 'POS_X', 'POS_Y', 'POS_Z'], inplace=True)
    return df


def get_junomag_range(start_timestamp, end_timestamp, path=juno_path+'fgm/1min'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start < end:
        year = start.year
        doy = start.strftime('%j')
        fn = f'fgm_jno_l3_{year}{doy}se_r60s_v01.sts'
        _df = get_junomag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
JUNO POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def juno_furnish():
    """Main"""
    juno_path = kernels_path+'juno/'
    generic_path = kernels_path+'generic/'
    juno_kernels = os.listdir(juno_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in juno_kernels:
        spiceypy.furnsh(os.path.join(juno_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))
    

def get_juno_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        juno_furnish()
    try:
        pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [None, None, None, None]   


def get_juno_positions(time_series):
    positions = []
    for t in time_series:
        position = get_juno_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_juno_positions_daily(start, end):
    t = start
    positions = []
    while t < end:
        position = get_juno_pos(t)
        positions.append(position)
        t += timedelta(days=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_juno_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    if spiceypy.ktotal('ALL') < 1:
        juno_furnish()
    transform = spiceypy.pxform(base_frame, to_frame, spiceypy.datetime2et(epoch))
    return transform


def transform_data(df, to_frame):
    pass
