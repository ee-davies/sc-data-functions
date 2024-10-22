import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import spiceypy
import os
import urllib.request
import os.path
import json
from scipy.io import netcdf
import glob
import pickle
import position_frame_transforms as pos_transform


"""
NOAA/DSCOVR DATA PATH
"""


dscovr_path='/Volumes/External/data/dscovr/'
kernels_path='/Volumes/External/data/kernels/'


"""
DSCOVR BAD DATA FILTER
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
DSCOVR MAG and PLAS DATA
# Can call MAG and PLAS last 7 days directly from https://services.swpc.noaa.gov/products/solar-wind/
# If those files aren't working, can download manually from https://www.swpc.noaa.gov/products/real-time-solar-wind and load both using get_noaa_realtime_alt 
# Raw data is in GSM coordinates; will implement transform to GSE/RTN
"""

## REALTIME

def get_noaa_mag_realtime_7days():
    #mag data request produces file in GSM coords
    request_mag=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    file_mag = request_mag.read()
    data_mag = json.loads(file_mag)
    noaa_mag_gsm = pd.DataFrame(data_mag[1:], columns=['time', 'bx', 'by', 'bz', 'lon_gsm', 'lat_gsm', 'bt'])

    noaa_mag_gsm['time'] = pd.to_datetime(noaa_mag_gsm['time'])
    noaa_mag_gsm['bx'] = noaa_mag_gsm['bx'].astype('float')
    noaa_mag_gsm['by'] = noaa_mag_gsm['by'].astype('float')
    noaa_mag_gsm['bz'] = noaa_mag_gsm['bz'].astype('float')
    noaa_mag_gsm['bt'] = noaa_mag_gsm['bt'].astype('float')

    noaa_mag_gsm.drop(columns = ['lon_gsm', 'lat_gsm'], inplace=True)

    return noaa_mag_gsm


def get_noaa_plas_realtime_7days():
    #plasma data request returns bulk parameters: density, v_bulk, temperature
    request_plas=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json')
    file_plas = request_plas.read()
    data_plas = json.loads(file_plas)
    noaa_plas = pd.DataFrame(data_plas[1:], columns=['time', 'np', 'vt', 'tp'])

    noaa_plas['time'] = pd.to_datetime(noaa_plas['time'])
    noaa_plas['np'] = noaa_plas['np'].astype('float')
    noaa_plas['vt'] = noaa_plas['vt'].astype('float')
    noaa_plas['tp'] = noaa_plas['tp'].astype('float')
    return noaa_plas


#Calling directly from json file: if json files are not working/ producing same data as seen on the realtime plots at https://www.swpc.noaa.gov/products/real-time-solar-wind:
#download file manually, e.g. load 7 days data, 'Save as text', and load using 'get_noaa_realtime_alt()'
def get_noaa_realtime_alt(path=f'{dscovr_path}'):

    filename = os.listdir(path)[0]
    noaa_alt = pd.read_table(f'{path}/{filename}', header=9, sep='\s+')
    noaa_alt = noaa_alt.reset_index()
    noaa_alt['time'] = pd.to_datetime(noaa_alt['index'] + ' ' + noaa_alt['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    noaa_alt = noaa_alt.drop(columns=['index', 'Timestamp'])

    noaa_alt.rename(columns={'Bt-med': 'bt', 'Bx-med': 'bx', 'By-med': 'by', 'Bz-med': 'bz'}, inplace=True)
    noaa_alt.rename(columns={'Dens-med': 'np', 'Speed-med': 'vt', 'Temp-med': 'tp'}, inplace=True)

    noaa_alt.drop(columns = ['Source', 'Bt-min', 'Bt-max', 'Bx-min', 'Bx-max', 'By-min', 'By-max', 'Bz-min', 'Bz-max'], inplace=True)
    noaa_alt.drop(columns = ['Phi-mean', 'Phi-min', 'Phi-max', 'Theta-med', 'Theta-min', 'Theta-max'], inplace=True)
    noaa_alt.drop(columns = ['Dens-min', 'Dens-max', 'Speed-min', 'Speed-max', 'Temp-min', 'Temp-max'], inplace=True)

    return noaa_alt


## DSCOVR DATA UP TO CURRENT DAY (INCL COMPONENTS, MORE PARAMETERS)


def get_dscovrplas_gse(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'proton_speed', 'proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_density', 'proton_temperature'], ['time','vt','vx', 'vy', 'vz', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrplas_gse_range(start_timestamp, end_timestamp, path=f'{dscovr_path}'+'plas/'):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/oe_f1m_dscovr_s{date_str}000000_*.nc')
        _df = get_dscovrplas_gse(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrplas_gsm(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'proton_speed', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_density', 'proton_temperature'], ['time','vt','vx', 'vy', 'vz', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrplas_gsm_range(start_timestamp, end_timestamp, path=f'{dscovr_path}'+'plas/'):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/oe_f1m_dscovr_s{date_str}000000_*.nc')
        _df = get_dscovrplas_gsm(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrmag_gsm(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'bt', 'bx_gsm', 'by_gsm', 'bz_gsm'], ['time','bt','bx', 'by', 'bz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


"""
DSCOVR POSITIONS
# Can call POS from last 7 days directly from https://services.swpc.noaa.gov/products/solar-wind/
# If those files aren't working, can download manually from https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/pop 
# Raw data is in GSE coordinates; will implement transform to HEEQ etc
"""


def get_noaa_pos_realtime_7days():
    #position data request returns gse coordinates
    request_pos=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json')
    file_pos = request_pos.read()
    data_pos = json.loads(file_pos)
    cols = ['time', 'x', 'y', 'z', 'vx_gse', 'vy_gse', 'vz_gse', "x_gsm", "y_gsm", "z_gsm", "vx_gsm", "vy_gsm", "vz_gsm"]
    noaa_pos = pd.DataFrame(data_pos[1:], columns=cols)

    noaa_pos['time'] = pd.to_datetime(noaa_pos['time'])
    noaa_pos['x'] = noaa_pos['x'].astype('float')
    noaa_pos['y'] = noaa_pos['y'].astype('float')
    noaa_pos['z'] = noaa_pos['z'].astype('float')

    noaa_pos.drop(columns = ['vx_gse', 'vy_gse', 'vz_gse', "x_gsm", "y_gsm", "z_gsm", "vx_gsm", "vy_gsm", "vz_gsm"], inplace=True)

    return noaa_pos


#If realtime doesn't work, 2nd best is download files manually (2 day behind)
#https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download//pop
#Load single position file from specific path using netcdf from scipy.io
#Will show depreciated warning message for netcdf namespace
def get_dscovrpos(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        #print(file2read.variables.keys()) to read variable names
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'sat_x_gse', 'sat_y_gse', 'sat_z_gse'], ['time', 'x', 'y', 'z'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrpositions(start_timestamp, end_timestamp):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{kernels_path}'+'dscovr/'+f'oe_pop_dscovr_s{date_str}000000_*.nc')
        _df = get_dscovrpos(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def create_dscovr_pkl(output_path='/Users/emmadavies/Documents/Projects/SolO_Realtime_Preparation/March2024/'):

    df_mag = get_noaa_mag_realtime_7days()
    if df_mag is None:
        print(f'DSCOVR MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)

    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_noaa_plas_realtime_7days()
    if df_plas is None:
        print(f'DSCOVR PLAS data is empty for this timerange')
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

    #get dscovr positions and transform from GSE to HEEQ
    #also insert empty nan columns for vx, vy, vz
    #positions are given every hour, so interpolate for 1min res; linear at the moment, can change
    df_pos = get_noaa_pos_realtime_7days()
    df_pos_HEE = pos_transform.GSE_to_HEE(df_pos)
    df_pos_HEEQ = pos_transform.HEE_to_HEEQ(df_pos_HEE)
    if df_pos_HEEQ is None:
        print(f'DSCOVR POS data is empty for this timerange')
        df_pos_HEEQ = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
        pos_rdf = df_pos_HEEQ
    else:
        pos_rdf = df_pos_HEEQ.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False)
        pos_rdf['vx'] = np.nan
        pos_rdf['vy'] = np.nan
        pos_rdf['vz'] = np.nan
        pos_rdf.set_index(pd.to_datetime(pos_rdf['time']), inplace=True)
        if pos_rdf.shape[0] != 0:
            pos_rdf = pos_rdf.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns
    #position and data files are different lengths; have trimmed to data length (no future positions)
    comb_df_nans = pd.concat([magplas_rdf, pos_rdf], axis=1)
    comb_df = comb_df_nans[comb_df_nans['bt'].notna()]

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    dscovr=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    dscovr = dscovr.view(np.recarray)

    dscovr.time=dt_lst
    dscovr.bx=comb_df['bx']
    dscovr.by=comb_df['by']
    dscovr.bz=comb_df['bz']
    dscovr.bt=comb_df['bt']
    dscovr.vx=comb_df['vx']
    dscovr.vy=comb_df['vy']
    dscovr.vz=comb_df['vz']
    dscovr.vt=comb_df['vt']
    dscovr.np=comb_df['np']
    dscovr.tp=comb_df['tp']
    dscovr.x=comb_df['x']
    dscovr.y=comb_df['y']
    dscovr.z=comb_df['z']
    dscovr.r=comb_df['r']
    dscovr.lat=comb_df['lat']
    dscovr.lon=comb_df['lon']

    #dump to pickle file
    header='Realtime past 7 day MAG, PLAS and position data from DSCOVR, sourced from https://services.swpc.noaa.gov/products/solar-wind/' + \
    'Timerange: '+dscovr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+dscovr.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by dscovr.time, dscovr.bx, dscovr.r etc. '+\
    'Total number of data points: '+str(dscovr.size)+'. '+\
    'Units are btxyz [nT, GSM], vtxy [km s^-1], np [cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    t_now_date_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
    pickle.dump([dscovr,header], open(output_path+f'dscovr_gsm_{t_now_date_hour}.p', "wb"))
