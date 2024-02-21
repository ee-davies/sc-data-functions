import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
# import os
import glob

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


def furnish():
    """Main"""
    base = r"kernels\mes\*"
    kernels = glob.glob(base)
    for kernel in kernels:
        spiceypy.furnsh(kernel)


def get_mes_position(timestamp, prefurnished=False):
    if not prefurnished: 
        if spiceypy.ktotal('ALL') < 1:
            furnish()
    try:
        mes_pos = spiceypy.spkpos("MESSENGER", spiceypy.datetime2et(timestamp), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(mes_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(mes_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(mes_pos[1], mes_pos[0]) * 360 / 2 / np.pi
        return [timestamp, mes_pos, r_au, lat, lon]
    except Exception as e:
        print(e)
        return [None, None, None, None]


def get_mes_positions(start, end):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
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