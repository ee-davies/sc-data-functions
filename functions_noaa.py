import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
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
import requests
import gzip
import shutil


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


def filter_bad_col(df, col, bad_val): #filter by individual columns
    if bad_val < 0:
        mask_vals = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask_vals = df[col] > bad_val  # boolean mask for all bad values
    df[col].mask(mask_vals, inplace=True)
    return df


"""
DSCOVR DOWNLOAD: SCIENCE DATA
#modified base functions from H.Ruedisser
#allow for MAG(m1m), SWE(f1m), and position(pop) datafiles to be called using wget system
"""

def to_epoch_millis_utc(dt):
    dt_utc = dt.replace(tzinfo=timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def extract_wget_links(start_datetime, end_datetime, datatype="f1m"): #"f1m", "m1m", "pop"

    start_ms = to_epoch_millis_utc(start_datetime)
    end_ms = to_epoch_millis_utc(end_datetime + timedelta(days=1, milliseconds=-1))

    api_url = f"https://www.ngdc.noaa.gov/dscovr-data-access/files?start_date={start_ms}&end_date={end_ms}"

    print(f"Requesting file list from: {api_url}")

    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"Error: Unable to fetch data from {api_url}")
        return []

    file_list = response.json()

    # Collect datatype to download URLs
    file_urls = []
    for date_key, file_types in file_list.items():
        if isinstance(file_types, dict) and datatype in file_types:
            file_urls.append(file_types[datatype])

    return file_urls


def download_dscovr(start_datetime, end_datetime, datatype:str, path=dscovr_path):  #"f1m", "m1m", "pop"
    if datatype == "f1m":
        output_path = path+'plas/'
    elif datatype == "m1m":
        output_path = path+'mag/'
    elif datatype == "pop":
        output_path = path+'orb/'
    file_urls = extract_wget_links(start_datetime, end_datetime, datatype)
    for file_url in file_urls:
        filename = os.path.basename(file_url)
        file_path = os.path.join(output_path, filename)
        print(f"Downloading {file_url} to {file_path}")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            continue
        #extract zips
        if filename.endswith(".gz"):
            with open(file_path[:-3], "wb") as f_out:
                with gzip.open(file_path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print(f"Extracted {filename[:-3]} from {filename}")


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
        df = filter_bad_data(df, 'vt', -99998)
        df = filter_bad_data(df, 'vx', -99998)
        df = filter_bad_data(df, 'vy', -99998)
        df = filter_bad_data(df, 'vz', -99998)
        df = filter_bad_data(df, 'np', -99998)
        df = filter_bad_data(df, 'tp', -99998)
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
        try:
            fn = glob.glob(f'{path}/oe_f1m_dscovr_s{date_str}000000_*.nc')
            _df = get_dscovrplas_gse(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrplas_gsm(fp):
    """raw = gsm"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'proton_speed', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_density', 'proton_temperature'], ['time','vt','vx', 'vy', 'vz', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = filter_bad_data(df, 'vt', -99998)
        df = filter_bad_data(df, 'vx', -99998)
        df = filter_bad_data(df, 'vy', -99998)
        df = filter_bad_data(df, 'vz', -99998)
        df = filter_bad_data(df, 'np', -99998)
        df = filter_bad_data(df, 'tp', -99998)
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
        try:
            _df = get_dscovrplas_gsm(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrmag_gse(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'bt', 'bx_gse', 'by_gse', 'bz_gse'], ['time','bt','bx', 'by', 'bz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = filter_bad_data(df, 'bt', -99998)
        df = filter_bad_data(df, 'bx', -99998)
        df = filter_bad_data(df, 'by', -99998)
        df = filter_bad_data(df, 'bz', -99998)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrmag_gse_range(start_timestamp, end_timestamp, path=f'{dscovr_path}'+'mag/'):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}/oe_m1m_dscovr_s{date_str}000000_*.nc')
            _df = get_dscovrmag_gse(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrmag_gsm(fp):
    """raw = gsm"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'bt', 'bx_gsm', 'by_gsm', 'bz_gsm'], ['time','bt','bx', 'by', 'bz'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = filter_bad_data(df, 'bt', -99998)
        df = filter_bad_data(df, 'bx', -99998)
        df = filter_bad_data(df, 'by', -99998)
        df = filter_bad_data(df, 'bz', -99998)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrmag_gsm_range(start_timestamp, end_timestamp, path=f'{dscovr_path}'+'mag/'):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}/oe_m1m_dscovr_s{date_str}000000_*.nc')
            _df = get_dscovrmag_gsm(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
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
def get_dscovrpos_gse(fp):
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


def get_dscovrpos_gsm(fp):
    """raw = gsm"""
    try:
        ncdf = netcdf.NetCDFFile(fp,'r')
        #print(file2read.variables.keys()) to read variable names
        data = {df_col: ncdf.variables[cdf_col][:] for cdf_col, df_col in zip(['time', 'sat_x_gsm', 'sat_y_gsm', 'sat_z_gsm'], ['time', 'x', 'y', 'z'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_dscovrpositions_gse(start_timestamp, end_timestamp):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{dscovr_path}'+'orb/'+f'oe_pop_dscovr_s{date_str}000000_*.nc')
            _df = get_dscovrpos_gse(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def get_dscovrpositions_gsm(start_timestamp, end_timestamp):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{dscovr_path}'+'orb/'+f'oe_pop_dscovr_s{date_str}000000_*.nc')
            _df = get_dscovrpos_gsm(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


## COMBINED FILES

#use this function for consistent co-ord system e.g. mag, plas, pos all GSM
def create_dscovr_pkl(start_timestamp, end_timestamp, output_path=dscovr_path):
    #mag data
    df_mag = get_dscovrmag_gsm_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'DSCOVR MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
    #plas data
    df_plas = get_dscovrplas_gsm_range(start_timestamp, end_timestamp)
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
    #position data
    df_pos = get_dscovrpositions_gsm(start_timestamp, end_timestamp)
    if df_pos is None:
        print(f'DSCOVR POS data is empty for this timerange')
        df_pos = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
        pos_rdf = df_pos
    else:
        pos_rdf = df_pos.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False)
        r, lat, lon = cart2sphere(pos_rdf['x'],pos_rdf['y'],pos_rdf['z'])
        pos_rdf['r'] = r
        pos_rdf['lat'] = lat
        pos_rdf['lon'] = lon
        pos_rdf.set_index(pd.to_datetime(pos_rdf['time']), inplace=True)
        if pos_rdf.shape[0] != 0:
            pos_rdf = pos_rdf.drop(columns=['time'])
    #combine
    comb_df_nans = pd.concat([magplas_rdf, pos_rdf], axis=1)
    comb_df = comb_df_nans[comb_df_nans['bt'].notna()]
    #create rec array
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
    #create header
    header='Science level MAG, PLAS and position data from DSCOVR, sourced from https://www.ngdc.noaa.gov/dscovr/portal/index.html' + \
    'Timerange: '+dscovr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+dscovr.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', 1 min time resolution.'+\
    'The data are available in a numpy recarray, fields can be accessed by dscovr.time, dscovr.bx, dscovr.r etc. '+\
    'Total number of data points: '+str(dscovr.size)+'. '+\
    'Units are btxyz [nT, GSM], vtxyz [km s^-1, GSM], np [cm^-3], tp [K], position x/y/z/r/lon/lat [km, degree, GSM]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([dscovr,header], open(output_path+f'dscovr_gsm_{datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")}.p', "wb"))


#use this function for non-realtime use in real-time codes, where mag and plas data is gsm, and positions are HEEQ
def create_dscovr_nonrealtime_pkl(start_timestamp, end_timestamp, output_path=dscovr_path):
    #mag data
    df_mag = get_dscovrmag_gsm_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'DSCOVR MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
    #plas data
    df_plas = get_dscovrplas_gsm_range(start_timestamp, end_timestamp)
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
    #position data
    df_pos = get_dscovrpositions_gse(start_timestamp, end_timestamp)
    df_pos_HEE = pos_transform.GSE_to_HEE(df_pos)
    df_pos_HEEQ = pos_transform.HEE_to_HEEQ(df_pos_HEE)
    if df_pos_HEEQ is None:
        print(f'DSCOVR POS data is empty for this timerange')
        df_pos_HEEQ = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
        pos_rdf = df_pos_HEEQ
    else:
        pos_rdf = df_pos_HEEQ.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False)
        # r, lat, lon = cart2sphere(pos_rdf['x'],pos_rdf['y'],pos_rdf['z'])
        # pos_rdf['r'] = r
        # pos_rdf['lat'] = lat
        # pos_rdf['lon'] = lon
        pos_rdf.set_index(pd.to_datetime(pos_rdf['time']), inplace=True)
        if pos_rdf.shape[0] != 0:
            pos_rdf = pos_rdf.drop(columns=['time'])
    #combine
    comb_df_nans = pd.concat([magplas_rdf, pos_rdf], axis=1)
    comb_df = comb_df_nans[comb_df_nans['bt'].notna()]
    #create rec array
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
    #create header
    header='Science level MAG, PLAS and position data from DSCOVR, sourced from https://www.ngdc.noaa.gov/dscovr/portal/index.html' + \
    'Timerange: '+dscovr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+dscovr.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', 1 min time resolution.'+\
    'The data are available in a numpy recarray, fields can be accessed by dscovr.time, dscovr.bx, dscovr.r etc. '+\
    'Total number of data points: '+str(dscovr.size)+'. '+\
    'Units are btxyz [nT, GSM], vtxyz [km s^-1, GSM], np [cm^-3], tp [K], position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([dscovr,header], open(output_path+f'dscovr_gsm_{datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")}.p', "wb"))


def create_dscovr_realtime_pkl(output_path='/Users/emmadavies/Documents/Projects/SolO_Realtime_Preparation/March2024/'):

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
