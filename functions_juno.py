import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
# import os
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


def get_junomag(fp):
    """Get data and return pd.DataFrame."""
    cols = ['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond',
            'Decimal Day', 'b_x', 'b_y', 'b_z', 'Range', 'POS_X', 'POS_Y', 'POS_Z']
    try:
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if sum(c.isalpha() for c in line) == 0:
                    break

        df = pd.read_csv(fp, skiprows=i, sep=r'\s+', names=cols)
        df['timestamp'] = df[['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond']]\
            .apply(lambda x: datetime.strptime(' '.join(str(y) for y in x),
                                               r'%Y %j %H %M %S %f'), axis=1)
        df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_junomag_range(start_timestamp, end_timestamp, path):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start < end:
        year = start.year
        doy = start.strftime('%j')
        fn = f'fgm_jno_l3_{year}_{doy}_r60s_v01.sts'
        _df = get_junomag(f'{path}/{year}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def furnish():
    """Main"""
    base = r"kernels\juno\*"
    # kernels = ["naif0012.tls", "pck00010.tpc", "juno_struct_v02.bsp", "de434s.bsp", 
    #            "JNO_SCLKSCET.00047.tsc", "juno_v09.tf", "jup310.bsp", "heliospheric_v004u.tf", 
    #            "spk_rec_110805_111026_120302.bsp", "spk_rec_111026_120308_120726.bsp", "spk_rec_120308_120825_121109.bsp",
    #            "spk_rec_120825_130515_130708.bsp", "spk_rec_130515_131005_131031.bsp", "spk_rec_130515_131005_151210.bsp",
    #            "spk_rec_131005_131014_131101.bsp", "spk_rec_131014_131114_140222.bsp", "spk_rec_131114_140918_141208.bsp",
    #            "spk_rec_140903_151003_160118.bsp", "spk_rec_151003_160312_160418.bsp", "spk_rec_160312_160522_160614.bsp",
    #            "spk_rec_160522_160729_160909.bsp", "spk_rec_160729_160923_161027.bsp"]
    # for kernel in kernels:
        # spiceypy.furnsh(os.path.join(base, kernel))
    kernels = glob.glob(base)
    for kernel in kernels:
        spiceypy.furnsh(kernel)
    


def get_juno_position(timestamp, prefurnished=False):
    if not prefurnished: 
        if spiceypy.ktotal('ALL') < 1:
            furnish()
    try:
        juno_pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(timestamp), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(juno_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(juno_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(juno_pos[1], juno_pos[0]) * 360 / 2 / np.pi
        return [timestamp, juno_pos, r_au, lat, lon]
    except Exception as e:
        print(e)
        return [None, None, None, None]


def get_juno_positions(start, end):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    t = start
    positions = []
    while t < end:
        juno_pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(juno_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(juno_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(juno_pos[1], juno_pos[0]) * 360 / 2 / np.pi
        positions.append([t, juno_pos, r_au, lat, lon])
        t += timedelta(days=1)
    return positions


def get_juno_positions_loop(col):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    positions = []
    for i in range(0,col.shape[0]):
        t = col.iloc[i]
        juno_pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r = np.linalg.norm(juno_pos)
        r_au = r/1.495978707E8
        lat = np.arcsin(juno_pos[2]/ r) * 360 / 2 / np.pi
        lon = np.arctan2(juno_pos[1], juno_pos[0]) * 360 / 2 / np.pi
        positions.append([t, juno_pos, r_au, lat, lon])
        i += 1
    return positions

def get_juno_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    transform = spiceypy.pxform(base_frame, to_frame, spiceypy.datetime2et(epoch))
    return transform

def transform_data(df, to_frame):
    pass
