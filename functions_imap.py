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
import json

import ialirt_data_access
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
IMAP BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val): #filter across whole df
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def filter_bad_col(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    df[col][mask] = np.nan
    return df


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
        df = filter_bad_col(df, 'np', -99999)
        df = filter_bad_col(df, 'vt', -99999)
        df = filter_bad_col(df, 'tp', -99999)
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
IMAP DATA: REAL-TIME DATA 
#4 hours is largest data service can call 
"""


def get_imapmag_realtime_hourly(start_timestamp, coord_sys:str):
    time_utc_start = start_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    end_timestamp = start_timestamp+timedelta(hours=1)
    time_utc_end = end_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    json_data = ialirt_data_access.data_product_query(instrument="mag", time_utc_start=time_utc_start, time_utc_end=time_utc_end)
    df = pd.DataFrame(json_data['data'])
    data = {df_new_col: df[df_col][:] for df_col, df_new_col in zip(['time_utc', 'mag_B_magnitude'], ['time', 'bt'])}
    new_df = pd.DataFrame.from_dict(data)
    new_df.time = pd.to_datetime(new_df.time)
    new_df[['bx', 'by', 'bz']] = pd.DataFrame(df[f'mag_B_{coord_sys}'].tolist())
    return new_df


def get_imapplas_realtime_hourly(start_timestamp):
    time_utc_start = start_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    end_timestamp = start_timestamp+timedelta(hours=1)
    time_utc_end = end_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    json_data = ialirt_data_access.data_product_query(instrument="swapi", time_utc_start=time_utc_start, time_utc_end=time_utc_end)
    df = pd.DataFrame(json_data['data'])
    data = {df_new_col: df[df_col][:] for df_col, df_new_col in zip(['time_utc', 'swapi_pseudo_proton_speed', 'swapi_pseudo_proton_density', 'swapi_pseudo_proton_temperature'], ['time', 'vt', 'np', 'tp'])}
    new_df = pd.DataFrame.from_dict(data)
    new_df.time = pd.to_datetime(new_df.time)
    new_df['vt'] = new_df['vt'].astype('float64')
    new_df['np'] = new_df['np'].astype('float64')
    new_df['tp'] = new_df['tp'].astype('float64')
    new_df = filter_bad_col(new_df, 'vt', -99999)
    new_df = filter_bad_col(new_df, 'np', -99999)
    new_df = filter_bad_col(new_df, 'tp', -99999)
    return new_df


#Only use for short durations
def get_imap_realtime_shortrange(start_timestamp, end_timestamp, instrument:str, coord_sys='RTN'):
    df = None
    start = start_timestamp
    end = end_timestamp
    while start < end:
        try:
            if instrument == 'mag':
                _df = get_imapmag_realtime_hourly(start, coord_sys)
            elif instrument == 'plas':
                _df = get_imapplas_realtime_hourly(start)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, "DataFrame is empty") 
        start += timedelta(hours=1)
    return df


def get_imap_realtime_day(start_timestamp, instrument:str, coord_sys='RTN', save_file=True, path=f'{imap_path}'):
    end_timestamp = start_timestamp+timedelta(days=1)
    df = get_imap_realtime_shortrange(start_timestamp, end_timestamp, instrument, coord_sys)
    rdf = df.set_index('time').resample('1min').mean(numeric_only=True).reset_index(drop=False)
    if save_file is True:
        savedate = f'{start_timestamp.year}{start_timestamp.month:02}{start_timestamp.day:02}'
        if instrument == 'mag':
            rdf.to_pickle(f"{path}ialirt/realtime/mag/imap_realtime_mag_{coord_sys}_{savedate}.pkl")
            print(f"Saved {coord_sys} mag data for {savedate}")
        elif instrument == 'plas':
            rdf.to_pickle(f"{path}ialirt/realtime/plas/imap_realtime_plas_{savedate}.pkl")
            print(f"Saved plasma data for {savedate}")
    return rdf


def save_imap_realtime_daily_1min(start_timestamp, end_timestamp, instrument:str, coord_sys='RTN'):
    start = start_timestamp
    end = end_timestamp+timedelta(days=1)
    while start < end:
        df = get_imap_realtime_day(start, instrument, coord_sys, save_file=True)
        start += timedelta(days=1)
    return print('Finished saving files.')


def get_imap_realtime_range_1min(start_timestamp, end_timestamp, instrument:str, coord_sys='RTN', path=f'{imap_path}'):
    df = None
    start = start_timestamp
    end = end_timestamp
    while start < end:
        savedate = f'{start.year}{start.month:02}{start.day:02}'
        try:
            if instrument == 'mag':
                _df = pd.read_pickle(f"{path}ialirt/realtime/mag/imap_realtime_mag_{coord_sys}_{savedate}.pkl")
            elif instrument == 'plas':
                _df = pd.read_pickle(f"{path}ialirt/realtime/plas/imap_realtime_plas_{savedate}.pkl")
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, "DataFrame is empty") 
        start += timedelta(days=1)
    return df


def read_json_to_dataframe(filepath, instrument=None, coord_sys=None):
    """
    Read IMAP JSON file into a pandas DataFrame.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
    instrument : str, optional
        Filter data by instrument (e.g., 'mag', 'hit'). 
        If None, returns all data.
    coord_sys : str, optional
        Coordinate system for mag data ('RTN', 'GSE', or 'GSM').
        Only applicable when instrument='mag'.
        Returns 'time', 'bt', and unpacked bx, by, bz columns for the specified coordinate system.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the data from the JSON file's 'data' field,
        optionally filtered by instrument and coordinate system.
        For mag data, mag_epoch is converted to datetime and renamed to 'time'.
        For mag data, mag_B_magnitude is renamed to 'bt'.
        For mag data with coord_sys specified, the mag_B_* arrays are unpacked into bx, by, bz columns.
        For mag data without coord_sys, all coordinate systems are unpacked with prefixes (RTN_bx, GSE_bx, etc.).
    """
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    # Extract the 'data' array from the JSON structure
    df = pd.DataFrame(json_data['data'])
    
    # Filter by instrument if specified
    if instrument is not None:
        df = df[df['instrument'] == instrument]
        
        # If mag instrument, return specific columns and unpack arrays
        if instrument == 'mag':
            if coord_sys is not None:
                # Return mag_epoch, mag_B_magnitude, and the specified coordinate system column
                mag_columns = ['mag_epoch', 'mag_B_magnitude', f'mag_B_{coord_sys}']
                # Only select columns that exist in the dataframe
                available_columns = [col for col in mag_columns if col in df.columns]
                df = df[available_columns].copy()
                
                # Convert mag_epoch to datetime (nanoseconds since J2000: 2000-01-01 12:00:00)
                if 'mag_epoch' in df.columns:
                    j2000 = datetime(2000, 1, 1, 12, 0, 0)
                    df['time'] = df['mag_epoch'].apply(
                        lambda x: j2000 + timedelta(microseconds=x/1000) if pd.notna(x) else pd.NaT
                    )
                    df = df.drop(columns=['mag_epoch'])
                
                # Rename mag_B_magnitude to bt
                if 'mag_B_magnitude' in df.columns:
                    df = df.rename(columns={'mag_B_magnitude': 'bt'})
                
                # Unpack the mag_B array into bx, by, bz columns (no prefix when single coord_sys)
                b_col = f'mag_B_{coord_sys}'
                if b_col in df.columns:
                    df['bx'] = df[b_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) >= 3 else None)
                    df['by'] = df[b_col].apply(lambda x: x[1] if isinstance(x, list) and len(x) >= 3 else None)
                    df['bz'] = df[b_col].apply(lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else None)
                    # Drop the original array column
                    df = df.drop(columns=[b_col])
                
                # Reorder columns to have time first, then bt, then bx, by, bz
                if 'time' in df.columns:
                    cols = ['time']
                    if 'bt' in df.columns:
                        cols.append('bt')
                    cols.extend([col for col in df.columns if col not in cols])
                    df = df[cols]
            else:
                # Return all mag columns if no coord_sys specified
                mag_columns = ['mag_epoch', 'mag_B_magnitude', 'mag_B_RTN', 'mag_B_GSE', 'mag_B_GSM']
                # Only select columns that exist in the dataframe
                available_columns = [col for col in mag_columns if col in df.columns]
                df = df[available_columns].copy()
                
                # Convert mag_epoch to datetime (nanoseconds since J2000: 2000-01-01 12:00:00)
                if 'mag_epoch' in df.columns:
                    j2000 = datetime(2000, 1, 1, 12, 0, 0)
                    df['time'] = df['mag_epoch'].apply(
                        lambda x: j2000 + timedelta(microseconds=x/1000) if pd.notna(x) else pd.NaT
                    )
                    df = df.drop(columns=['mag_epoch'])
                
                # Rename mag_B_magnitude to bt
                if 'mag_B_magnitude' in df.columns:
                    df = df.rename(columns={'mag_B_magnitude': 'bt'})
                
                # Unpack all coordinate system arrays (with prefixes since multiple coord systems)
                for coord in ['RTN', 'GSE', 'GSM']:
                    b_col = f'mag_B_{coord}'
                    if b_col in df.columns:
                        df[f'{coord}_bx'] = df[b_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) >= 3 else None)
                        df[f'{coord}_by'] = df[b_col].apply(lambda x: x[1] if isinstance(x, list) and len(x) >= 3 else None)
                        df[f'{coord}_bz'] = df[b_col].apply(lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else None)
                        # Drop the original array column
                        df = df.drop(columns=[b_col])
                
                # Reorder columns to have time first, then bt
                if 'time' in df.columns:
                    cols = ['time']
                    if 'bt' in df.columns:
                        cols.append('bt')
                    cols.extend([col for col in df.columns if col not in cols])
                    df = df[cols]
    
    return df


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

