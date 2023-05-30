from datetime import datetime, timedelta
import numpy as np
import spiceypy
import os
import pandas as pd
from spacepy import pycdf


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


def furnish():
    """Main"""
    base = "kernels/bepi"
    kernels = ["naif0012.tls", "pck00010.tpc", "de434s.bsp", "heliospheric_v004u.tf", 
               "bc_mpo_fcp_00094_20181020_20251101_v01.bsp", "bc_sci_v06.tf"]
    for kernel in kernels:
        spiceypy.furnsh(os.path.join(base, kernel))  


def get_bepi_position(timestamp, prefurnished=False):
    """Return timestamp, position array (in km), r_au, lat, lon."""
    if not prefurnished: 
        if spiceypy.ktotal('ALL') < 1:
            furnish()
    try:
        bepi_pos = spiceypy.spkpos("BEPICOLOMBO MPO", spiceypy.datetime2et(timestamp), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(bepi_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(bepi_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(bepi_pos[1], bepi_pos[0]) * 360 / 2 / np.pi
        return [timestamp, bepi_pos, r_au, lat, lon]
    except Exception as e:
        print(e)
        return [None, None, None, None]


def get_bepi_positions(start, end):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    t = start
    positions = []
    while t < end:
        bepi_pos = spiceypy.spkpos("BEPICOLOMBO MPO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(bepi_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(bepi_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(bepi_pos[1], bepi_pos[0]) * 360 / 2 / np.pi
        positions.append([t, bepi_pos, r_au, lat, lon])
        t += timedelta(hours=1)
    return positions


def get_bepi_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    if spiceypy.ktotal('ALL') < 1:
        furnish()
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
