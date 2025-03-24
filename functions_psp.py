import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
# import os
import glob
import urllib.request
import os.path
import pickle

import astrospice
from sunpy.coordinates import HeliocentricInertial, HeliographicStonyhurst


"""
PARKER SOLAR PROBE SERVER DATA PATH
"""

psp_path='/Volumes/External/data/psp/'
kernels_path='/Volumes/External/data/kernels/'


"""
PSP BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
PSP MAG DATA
# 1 min and full resolution files from https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2
# 1min files naming convention: psp_fld_l2_mag_rtn_1min_{date_str}_v02.cdf
# full res files are split into daily files, 6 hourly intervals, so naming convention: psp_fld_l2_mag_rtn_{date_str}{time}_v02.cdf
"""


#DOWNLOAD FUNCTIONS for 1min or full res data


def download_pspmag_1min(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/1min'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_fld_l2_mag_rtn_1min_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_pspmag_full(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/full'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            data_item_id = f'psp_fld_l2_mag_rtn_{date_str}{t:02}_v02'
            if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
                print(f'{data_item_id}.cdf has already been downloaded.')
            else:
                try:
                    data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/{year}/{data_item_id}.cdf'
                    urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                    print(f'Successfully downloaded {data_item_id}.cdf')
                except Exception as e:
                    print('ERROR', e, data_item_id)
        start += timedelta(days=1)        
        

#LOAD FUNCTIONS for MAG data 


def get_pspmag_1min(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN_1min'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN_1min'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspmag_full(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps


def get_pspmag_range_1min(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_fld_l2_mag_rtn_1min_{date_str}_v02.cdf'
        _df = get_pspmag_1min(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_pspmag_range_full(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/full'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            fn = f'{path}/psp_fld_l2_mag_rtn_{date_str}{t:02}_v02.cdf'
            _df = get_pspmag_full(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
PSP PLAS DATA
"""


#DOWNLOAD FUNCTIONS for plas data


def download_pspplas_spc(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spc/l3i'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_swp_spc_l3i_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_pspplas_spi(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spi/l3'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_swp_spi_sf00_l3_mom_{date_str}_v04'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#LOAD FUNCTIONS for plasma (spc and spi) data


def get_pspspc_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_moment', 'wp_moment'], ['time', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_moment_RTN'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_fit', 'wp_fit'], ['time', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_fit_RTN'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit1(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np1_fit', 'wp1_fit'], ['time', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp1_fit_RTN'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspi_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'DENS', 'TEMP'], ['time', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['VEL_RTN_SUN'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


# LOAD RANGES of plasma data


def get_pspspc_range_mom(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spc/l3i'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'tp', -1E30)
    filter_bad_data(df, 'np', -1E30)
    filter_bad_data(df, 'vt', -1E30)
    filter_bad_data(df, 'vx', -1E30)
    filter_bad_data(df, 'vy', -1E30)
    filter_bad_data(df, 'vz', -1E30)
    return df


def get_pspspc_range_fit(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spc/l3i'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_fit(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'tp', -1E30)
    filter_bad_data(df, 'np', -1E30)
    filter_bad_data(df, 'vt', -1E30)
    filter_bad_data(df, 'vx', -1E30)
    filter_bad_data(df, 'vy', -1E30)
    filter_bad_data(df, 'vz', -1E30)
    return df


def get_pspspi_range_mom(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spi/l3'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spi_sf00_l3_mom_{date_str}_v04.cdf'
        _df = get_pspspi_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
PSP SPACECRAFT POSITIONS
#Calls directly from spiceypy kernels
#Set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def sphere2cart(r, lat, lon):
    x = r*np.cos(lat*(np.pi/180))*np.cos(lon*(np.pi/180))
    y = r*np.cos(lat*(np.pi/180))*np.sin(lon*(np.pi/180))
    z = r*np.sin(lat*(np.pi/180))
    r_au = r/1.495978707E8
    return x.value, y.value, z.value, r_au.value


#kernels obtained from https://cdaweb.gsfc.nasa.gov/pub/data/psp/ephemeris/spice/
def psp_furnish():
    """Main"""
    psp_path = kernels_path+'psp/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(psp_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(psp_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_psp_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        psp_furnish()
    pos = spiceypy.spkpos("PARKER SOLAR PROBE", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_psp_positions(time_series):
    positions = []
    for t in time_series:
        position = get_psp_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_psp_positions_daily(start, end):
    t = start
    positions = []
    while t < end:
        position = get_psp_pos(t)
        positions.append(position)
        t += timedelta(days=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_psp_positions_hourly(start, end):
    t = start
    positions = []
    while t < end:
        position = get_psp_pos(t)
        positions.append(position)
        t += timedelta(hours=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_psp_positions_minute(start, end):
    t = start
    positions = []
    while t < end:
        position = get_psp_pos(t)
        positions.append(position)
        t += timedelta(minutes=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


# def get_psp_positions(time_series):
#     kernels_psp = astrospice.registry.get_kernels('psp', 'predict')
#     frame = HeliographicStonyhurst()
#     coords_psp = astrospice.generate_coords('Solar probe plus', time_series)
#     coords_psp = coords_psp.transform_to(frame)
#     x, y, z, r_au = sphere2cart(coords_psp.radius, coords_psp.lat, coords_psp.lon)
#     lat = coords_psp.lat.value
#     lon = coords_psp.lon.value
#     t = [element.to_pydatetime() for element in list(time_series)]
#     positions = np.array([t, x, y, z, r_au, lat, lon])
#     df_positions = pd.DataFrame(positions.T, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
#     return df_positions


"""
OUTPUT COMBINED PICKLE FILE
including MAG, PLAS, and POSITION data
"""


def create_psp_pkl(start_timestamp, end_timestamp, output_path=psp_path):

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_pspmag_range_1min(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'PSP FIELDS data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)

    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_pspspi_range_mom(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'PSP SPI/MOM data is empty for this timerange')
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
    psp_pos = get_psp_positions(magplas_rdf['time'])
    psp_pos.set_index(pd.to_datetime(psp_pos['time']), inplace=True)
    psp_pos = psp_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, psp_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    psp=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    psp = psp.view(np.recarray) 

    psp.time=dt_lst
    psp.bx=comb_df['bx']
    psp.by=comb_df['by']
    psp.bz=comb_df['bz']
    psp.bt=comb_df['bt']
    psp.vx=comb_df['vx']
    psp.vy=comb_df['vy']
    psp.vz=comb_df['vz']
    psp.vt=comb_df['vt']
    psp.np=comb_df['np']
    psp.tp=comb_df['tp']
    psp.x=comb_df['x']
    psp.y=comb_df['y']
    psp.z=comb_df['z']
    psp.r=comb_df['r']
    psp.lat=comb_df['lat']
    psp.lon=comb_df['lon']
    
    #dump to pickle file
    
    header='Science level 2 solar wind magnetic field (FIELDS) and plasma data (SWEAP/SPI/MOM) from Parker Solar Probe, ' + \
    'obtained from https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min and https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/  '+ \
    'Timerange: '+psp.time[0].strftime("%Y-%b-%d %H:%M")+' to '+psp.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 5 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by psp.time, psp.bx, psp.vt etc. '+\
    'Total number of data points: '+str(psp.size)+'. '+\
    'Units are btxyz [nT, RTN], vtxy  [km s^-1], np[cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with [...] by E. Davies (twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([psp,header], open(psp_path+'psp_rtn.p', "wb"))