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