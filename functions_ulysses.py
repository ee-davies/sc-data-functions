import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import spiceypy
import glob
import urllib.request
import os.path
import pickle

from functions_general import load_path


"""
ULYSSES SERVER DATA PATH
"""

ulysses_path=load_path(path_name='ulysses_path')
print(f"Ulysses path loaded: {ulysses_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
ULYSSES BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val): #filter across whole df
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def filter_bad_col(df, col, bad_val): #filter by individual columns
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    df[col][mask] = np.nan
    return df


"""
ULYSSES MAG DATA
# obtained via https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/mag_cdaweb/vhm_1min/
# cdf files available in 1 min, 1 sec, m1
"""


#DOWNLOAD FUNCTIONS


def download_ulyssesmag_1min(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'mag/l2/1min'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'uy_1min_vhm_{date_str}_v01'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/mag_cdaweb/vhm_1min/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#Load single file from specific path using pycdf from spacepy
def get_ulyssesmag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'B_MAG'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['B_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        #df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_ulyssesmag_range(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'uy_1min_vhm_{date_str}_v01'
        fn = f'{path}/{data_item_id}.cdf'
        _df = get_ulyssesmag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
ULYSSES PLASMA DATA
# L2 plasma moments from SWOOPS instrument
"""


#DOWNLOAD FUNCTIONS

#all plasma files are yearly i.e. 19920101, except 19901118
def download_ulyssesplas(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'plas/l2'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        if year == 1990:
            date_str = f'{year}{start.month:02}{start.day:02}'
        else:
            date_str = f'{year}0101'
        data_item_id = f'uy_proton-moments_swoops_{date_str}_v01'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            if year == 1990:
                start += timedelta(days=1)
            else:
                start += timedelta(days=365.25)
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/plasma/swoops_cdaweb/proton-moments_swoops/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                if year == 1990:
                    start += timedelta(days=1)
                else:
                    start += timedelta(days=365.25)
            except Exception as e:
                print('ERROR', e, data_item_id)
                if year == 1990:
                    start += timedelta(days=1)
                else:
                    start += timedelta(days=365.25)


#Load single file from specific path using pycdf from spacepy
#plasma files also include mag data and heliocentricDistance and lat if needed
#need to assess proton temperature is correct
def get_ulyssesplas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'V_MAG', 'VR', 'VT', 'VN', 'dens'], ['time', 'vt', 'vx', 'vy', 'vz', 'np'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        t_par = cdf['Tpar'][:]
        t_per = cdf['Tper'][:]
        tp = np.sqrt(t_par**2 + t_per**2)
        df['tp'] = tp
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_ulyssesplas_range(start_timestamp, end_timestamp, path=f'{ulysses_path}'+'plas/l2'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        fn = glob.glob(f'{path}/uy_proton-moments_swoops_{year}*.cdf')
        _df = get_ulyssesplas(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=365.25)
    time_mask = (df['time'] > start_timestamp) & (df['time'] < end_timestamp)
    df_timerange = df[time_mask]
    return df_timerange


"""
ULYSSES POSITION DATA
# https://naif.jpl.nasa.gov/pub/naif/ULYSSES/kernels/spk/ #apparently may have discontinuities
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def ulysses_furnish():
    """Main"""
    ulysses_path = kernels_path+'ulysses/'
    generic_path = kernels_path+'generic/'
    ulysses_kernels = os.listdir(ulysses_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in ulysses_kernels:
        spiceypy.furnsh(os.path.join(ulysses_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_ulysses_pos(t): #doesn't automatically furnish, fix
    if spiceypy.ktotal('ALL') < 1:
        ulysses_furnish()
    try:
        pos = spiceypy.spkpos("ULYSSES", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [t, None, None, None, None, None, None]


def get_ulysses_positions(time_series):
    positions = []
    for t in time_series:
        position = get_ulysses_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_ulysses_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_ulysses_pos(t)
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


def get_ulysses_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_ulysses_pos(t)
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


def get_ulysses_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_ulysses_pos(t)
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
OUTPUT COMBINED PICKLE FILE
including MAG, PLAS, and POSITION data
"""


def create_ulysses_pkl(start_timestamp, end_timestamp=datetime.now(timezone.utc), level='l2', res='1min', output_path=ulysses_path):
    
    # #download solo mag and plasma data up to now 
    # download_solomag_1min(start_timestamp)
    # download_soloplas(start_timestamp)

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_ulyssesmag_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'Ulysses VHM/FGM data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        
    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_ulyssesplas_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'Ulysses SWOOPS data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
        plas_rdf = df_plas
    else:
        plas_rdf = df_plas.set_index('time').resample('1min').mean().reset_index(drop=False)
        plas_rdf.set_index(pd.to_datetime(plas_rdf['time']), inplace=True)
        if mag_rdf.shape[0] != 0:
            plas_rdf = plas_rdf.drop(columns=['time'])

    #need to combine mag and plasma dfs to get complete set of timestamps for position calculation
    magplas_rdf = pd.concat([mag_rdf, plas_rdf], axis=1)
    #some timestamps may be NaT so after joining, drop time column and reinstate from combined index col
    magplas_rdf = magplas_rdf.drop(columns=['time'])
    magplas_rdf['time'] = magplas_rdf.index
     
    #get solo positions for corresponding timestamps
    ulysses_furnish()
    ulysses_pos = get_ulysses_positions(magplas_rdf['time'])
    ulysses_pos.set_index(pd.to_datetime(ulysses_pos['time']), inplace=True)
    ulysses_pos = ulysses_pos.drop(columns=['time'])
    spiceypy.kclear()

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, ulysses_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    ulysses=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    ulysses = ulysses.view(np.recarray) 

    ulysses.time=dt_lst
    ulysses.bx=comb_df['bx']
    ulysses.by=comb_df['by']
    ulysses.bz=comb_df['bz']
    ulysses.bt=comb_df['bt']
    ulysses.vx=comb_df['vx']
    ulysses.vy=comb_df['vy']
    ulysses.vz=comb_df['vz']
    ulysses.vt=comb_df['vt']
    ulysses.np=comb_df['np']
    ulysses.tp=comb_df['tp']
    ulysses.x=comb_df['x']
    ulysses.y=comb_df['y']
    ulysses.z=comb_df['z']
    ulysses.r=comb_df['r']
    ulysses.lat=comb_df['lat']
    ulysses.lon=comb_df['lon']
    
    #dump to pickle file
    header='Ulysses L2 science data incl. magnetic field (VHM/FGM), plasma (SWOOPS), and heliospheric positions.' + \
    'Timerange: '+ulysses.time[0].strftime("%Y-%b-%d %H:%M")+' to '+ulysses.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by ulysses.time, ulysses.bx, ulysses.vt, ulysses.r etc. '+\
    'Total number of data points: '+str(ulysses.size)+'. '+\
    'Units are btxyz [nT, RTN], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([ulysses,header], open(output_path+f'ulysses_rtn.p', "wb"))