from datetime import datetime, timedelta
import numpy as np
import spiceypy
import os
import pandas as pd

from .functions_general import load_path


"""
BEPICOLOMBO DATA PATH
"""


bepi_path=load_path(path_name='bepi_path')
print(f"Bepi path loaded: {bepi_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


def format_path(fp):
    """Formatting required for CDF package."""
    return fp.replace('/', '\\')


"""
BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'Timestamp']
    df.loc[mask, cols] = np.nan
    return df


def get_bepimag(fp):
    """Fetch BEPI data from fp, returns DataFrame."""
    cols = ['date_time', '?', 'pos_x', 'pos_y', 'pos_z',
            'b_x', 'b_y', 'b_z', '?_x','?_y', '?_z',]
    df = pd.read_csv(fp, names=cols)
    df['timestamp'] = pd.to_datetime(df['date_time'], format=r'%Y-%m-%dT%H:%M:%S.%fZ')
    df['b_tot'] = df[['b_x', 'b_y', 'b_z']].apply(lambda x: np.linalg.norm(x), axis=1)
    df['pos_AU'] = df[['pos_x', 'pos_y', 'pos_z']].apply(lambda x: np.linalg.norm(x), axis=1)*6.6846E-9
    # return filter_bad_data(df, 'B_TOT', 9.99e+04)
    return df


def get_bepimag_range(start_timestamp, end_timestamp, path):
    """Pass two datetime objects and grab .tab files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        # C:\Users\emmad\Documents\Bepi-Colombo\202203\data\mag_der_sc_ob_a001_e2k_00000_20220308.tab
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'mag_der_sc_ob_a001_e2k_00000_{date_str}.tab'
        _df = get_bepimag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


"""
BEPI POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def bepi_furnish():
    """Main"""
    bepi_path = kernels_path+'bepi/'
    generic_path = kernels_path+'generic/'
    bepi_kernels = os.listdir(bepi_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in bepi_kernels:
        spiceypy.furnsh(os.path.join(bepi_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_bepi_pos(t, prefurnished=False):
    """Return timestamp, position array (in km), r_au, lat, lon."""
    if not prefurnished: 
        if spiceypy.ktotal('ALL') < 1:
            bepi_furnish()
    try:
        pos = spiceypy.spkpos("BEPICOLOMBO MPO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [t, None, None, None, None, None, None]


def get_bepi_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_bepi_pos(t)
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


def get_bepi_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_bepi_pos(t)
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


def get_bepi_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_bepi_pos(t)
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


# def get_bepi_positions(start, end):
#     if spiceypy.ktotal('ALL') < 1:
#         furnish()
#     t = start
#     positions = []
#     while t < end:
#         bepi_pos = spiceypy.spkpos("BEPICOLOMBO MPO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
#         r = np.linalg.norm(bepi_pos)
#         r_au = r/1.495978707E8
#         lat = np.arcsin(bepi_pos[2]/ r) * 360 / 2 / np.pi
#         lon = np.arctan2(bepi_pos[1], bepi_pos[0]) * 360 / 2 / np.pi
#         positions.append([t, bepi_pos, r_au, lat, lon])
#         t += timedelta(hours=1)
#     return positions






def get_bepi_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    if spiceypy.ktotal('ALL') < 1:
        bepi_furnish()
    transform = spiceypy.pxform(base_frame, to_frame, spiceypy.datetime2et(epoch))
    return transform


def transform_data(df, to_frame="rtn"):
    frame_id_map = {
        "rtn": "BC_MPO_RTN",
        # etc
    }
    b_out = []
    for i, t in enumerate(df['timestamp']):
        transform = get_bepi_transform(t, "ECLIPJ2000", frame_id_map[to_frame])
        mag_vector = np.array([[df['b_x'].iloc[i]], [df['b_y'].iloc[i]], [df['b_z'].iloc[i]]])
        transformation = np.matmul(transform, mag_vector)
        b_out.append(np.transpose(transformation)[0].tolist())
    b_out_df = pd.DataFrame(b_out, columns = ['b_x', 'b_y', 'b_z'])
    time_df = df['timestamp'].reset_index(drop=True)
    transformed_df = pd.concat([time_df, b_out_df], axis=1)
    transformed_df['b_tot'] = transformed_df[['b_x', 'b_y', 'b_z']].apply(lambda x: np.linalg.norm(x), axis=1)
    return transformed_df
