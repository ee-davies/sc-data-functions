import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
from scipy import constants
# import spiceypy
# import os
import glob
import urllib.request
from urllib.request import urlopen
import os.path
import pickle
from bs4 import BeautifulSoup

import data_frame_transforms as data_transform
import position_frame_transforms as pos_transform


"""
ACE DATA PATH
"""


ace_path = "/Volumes/External/Data/ACE/"


"""
ACE BAD DATA FILTER
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
        mask_vals = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask_vals = df[col] > bad_val  # boolean mask for all bad values
    df[col].mask(mask_vals, inplace=True)
    return df


"""
ACE DOWNLOAD DATA FUNCTIONS
"""

#SWE
def download_ace_swe(start_timestamp, end_timestamp, path=ace_path+'swe/'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('ac_h0_swe_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)


#MAG
def download_ace_mag(start_timestamp, end_timestamp, path=ace_path+'mfi/'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('ac_h0_mfi_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)


"""
LOAD ACE MAG DATA
"""


#approx RTN - flipped x and y component from GSE coords, use GSE function and data transform functions for higher accuratcay
def get_acemag_rtn_approx(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Magnitude'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSEc'][:].T
        df['bx'] = -1 * bx
        df['by'] = -1 * by
        df['bz'] = bz
        df = filter_bad_col(df, 'bt', -9.99E30)
        df = filter_bad_col(df, 'bx', 9.99E30)
        df = filter_bad_col(df, 'by', 9.99E30)
        df = filter_bad_col(df, 'bz', -9.99E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_acemag_gse(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Magnitude'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSEc'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df = filter_bad_col(df, 'bt', -9.99E30)
        df = filter_bad_col(df, 'bx', -9.99E30)
        df = filter_bad_col(df, 'by', -9.99E30)
        df = filter_bad_col(df, 'bz', -9.99E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_acemag_gsm(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Magnitude'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSM'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df = filter_bad_col(df, 'bt', -9.99E30)
        df = filter_bad_col(df, 'bx', -9.99E30)
        df = filter_bad_col(df, 'by', -9.99E30)
        df = filter_bad_col(df, 'bz', -9.99E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#RANGES
def get_acemag_rtn_approx_range(start_timestamp, end_timestamp, path=ace_path+'mfi'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_mfi_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_acemag_rtn_approx(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_acemag_gse_range(start_timestamp, end_timestamp, path=ace_path+'mfi'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_mfi_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_acemag_gse(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_acemag_gsm_range(start_timestamp, end_timestamp, path=ace_path+'mfi'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_mfi_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_acemag_gsm(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def acemag_gse_to_rtn(df_mag_gse, df_pos_gse):
    df_mag_heeq = data_transform.perform_mag_transform(df_mag_gse, 'GSE', 'HEEQ')
    df_pos_hee = pos_transform.GSE_to_HEE(df_pos_gse)
    df_pos_heeq = pos_transform.perform_transform(df_pos_hee, 'HEE', 'HEEQ')
    df_new_pos = data_transform.interp_to_newtimes(df_pos_heeq, df_mag_heeq) #should be same timestamps, nan depending
    combined_df = data_transform.combine_dataframes(df_mag_heeq,df_new_pos)
    df_mag_rtn = data_transform.HEEQ_to_RTN_mag(combined_df)
    return df_mag_rtn


def get_acemag_range(start_timestamp, end_timestamp, coord_sys:str):
    if coord_sys == 'GSE':
        df = get_acemag_gse_range(start_timestamp, end_timestamp)
    elif coord_sys == 'GSM':
        df = get_acemag_gsm_range(start_timestamp, end_timestamp)
    elif coord_sys == 'RTN':
        df_gse = get_acemag_gse_range(start_timestamp, end_timestamp)
        df_pos = get_acepos_frommag_range(start_timestamp, end_timestamp, coord_sys='GSE')
        df = acemag_gse_to_rtn(df_gse, df_pos)
    return df


"""
LOAD ACE SWE DATA
"""


def get_aceswe_rtn(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Vp', 'Np', 'Tpr'], ['time', 'vt', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vr, vt, vn = cdf['V_RTN'][:].T
        df['vx'] = vr
        df['vy'] = vt
        df['vz'] = vn
        df = filter_bad_col(df, 'np', -9.99E30)
        df = filter_bad_col(df, 'tp', -9.99E30)
        df = filter_bad_col(df, 'vt', -9.99E30)
        df = filter_bad_col(df, 'vx', -9.99E30)
        df = filter_bad_col(df, 'vy', -9.99E30)
        df = filter_bad_col(df, 'vz', -9.99E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_aceswe_gse(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Vp', 'Np', 'Tpr'], ['time', 'vt', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['V_GSE'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df = filter_bad_col(df, 'np', -9.99E30)
        df = filter_bad_col(df, 'tp', -9.99E30)
        df = filter_bad_col(df, 'vt', -9.99E30)
        df = filter_bad_col(df, 'vx', -9.99E30)
        df = filter_bad_col(df, 'vy', -9.99E30)
        df = filter_bad_col(df, 'vz', -9.99E30)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_aceswe_gsm(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Vp', 'Np', 'Tpr'], ['time', 'vt', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['V_GSM'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['tp'].mask((df['tp'] < -9.99e+30), inplace=True)
        df['np'].mask((df['np'] < -9.99e+30), inplace=True)
        df['vt'].mask((df['vt'] < -9.99e+30), inplace=True)
        df['vx'].mask((df['vx'] < -9.99e+30), inplace=True)
        df['vy'].mask((df['vy'] < -9.99e+30), inplace=True)
        df['vz'].mask((df['vz'] < -9.99e+30), inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#RANGES
def get_aceswe_rtn_range(start_timestamp, end_timestamp, path=ace_path+'swe'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_swe_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_aceswe_rtn(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_aceswe_gse_range(start_timestamp, end_timestamp, path=ace_path+'swe'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_swe_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_aceswe_gse(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_aceswe_gsm_range(start_timestamp, end_timestamp, path=ace_path+'swe'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_swe_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_aceswe_gsm(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_aceswe_range(start_timestamp, end_timestamp, coord_sys:str):
    if coord_sys == 'GSE':
        df = get_aceswe_gse_range(start_timestamp, end_timestamp)
    elif coord_sys == 'GSM':
        df = get_aceswe_gsm_range(start_timestamp, end_timestamp)
    elif coord_sys == 'RTN':
        df = get_aceswe_rtn_range(start_timestamp, end_timestamp)
    return df


"""
ACE POSITION FUNCTIONS: no spice kernels, uses data file
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


#positions in units of km
def get_acepos(fp, coord_sys='GSE'): #GSE and GSM available
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        x, y, z = cdf[f'SC_pos_{coord_sys}'][:].T
        r, lat, lon = cart2sphere(x,y,z)
        df['x'] = x
        df['y'] = y
        df['z'] = z
        df['r'] = r
        df['lat'] = lat
        df['lon'] = lon
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_acepos_frommag_range(start_timestamp, end_timestamp, coord_sys='GSE', path=ace_path+'mfi'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_mfi_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_acepos(f'{path_fn}', coord_sys)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_acepos_fromswe_range(start_timestamp, end_timestamp, coord_sys='GSE', path=ace_path+'swe'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        fn = f'ac_h0_swe_{start.year}{start.month:02}{start.day:02}'
        try:
            path_fn = glob.glob(f'{path}/{fn}*.cdf')[0]
        except Exception as e:
            path_fn = None
        _df = get_acepos(f'{path_fn}', coord_sys)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


# #initially attempts to get position from MAG file, if empty, tries SWE file
# def get_acepos_gsm_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE'):
#     """Pass two datetime objects and grab .cdf files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start <= end:
#         fn = f'ac_h0_mfi_{start.year}{start.month:02}{start.day:02}'
#         try:
#             path_fn = glob.glob(f'{path}/mfi/{fn}*.cdf')[0]
#         except Exception as e:
#             path_fn = None
#         _df = get_acepos_gsm(f'{path_fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = pd.concat([df, _df])
#         else:
#             fn = f'ac_h0_swe_{start.year}{start.month:02}{start.day:02}'
#             try:
#                 path_fn = glob.glob(f'{path}/swe/{fn}*.cdf')[0]
#             except Exception as e:
#                 path_fn = None
#             _df = get_acepos_gsm(f'{path_fn}')
#             if _df is not None:
#                 if df is None:
#                     df = _df.copy(deep=True)
#                 else:
#                     df = pd.concat([df, _df])
#         start += timedelta(days=1)
#     return df


"""
EXTRA FUNCTIONS (TO MOVE TO GENERAL IN SITU ANALYSIS)
"""


def resample_df(df, resample_min):
    rdf = df.set_index('time').resample(f'{resample_min}min').mean().reset_index(drop=False)
    return rdf


def merge_rdfs(df1, df2):
    df1.set_index(pd.to_datetime(df1['time']), inplace=True)
    df2.set_index(pd.to_datetime(df2['time']), inplace=True)
    mdf = pd.concat([df1, df2], axis=1)
    mdf = mdf.drop(['time'], axis=1)
    mdf = mdf.reset_index(drop=False)
    return mdf


def calc_pressure_params(plasmag_df):
# assuming Tpr is the (isotropic) temperature
# in reality is temperature in radial direction: https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H0_SWE
    plasmag_df['p_plas'] = (plasmag_df['density']*10**6)*constants.k*plasmag_df['temperature']
    plasmag_df['p_mag'] = 0.5*(plasmag_df['b_tot']*10**(-9))**2./constants.mu_0
    plasmag_df['beta'] = plasmag_df['p_plas']/plasmag_df['p_mag']
    plasmag_df['p_tot'] = plasmag_df['p_plas'] + plasmag_df['p_mag']
    return plasmag_df


"""
ACE PICKLE DATA
"""


def create_ace_gsm_pkl(start_timestamp, end_timestamp): #just initial quick version, may fail easily
    #create dataframes for mag, plas, and position
    df_mag = get_acemag_gsm_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'ACE MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)

    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_aceswe_gsm_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'ACE SWE data is empty for this timerange')
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

    df_pos = get_acepos_gsm_range(start_timestamp, end_timestamp)
    if df_pos is None:
        print(f'ACE POS data is empty for this timerange')
        df_pos = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
        pos_rdf = df_pos.drop(columns=['time'])
    else:
        pos_rdf = df_pos.set_index('time').resample('1min').mean().reset_index(drop=False)
        pos_rdf.set_index(pd.to_datetime(pos_rdf['time']), inplace=True)

    magplaspos_rdf = pd.concat([magplas_rdf, pos_rdf], axis=1)
    #some timestamps may be NaT so after joining, drop time column and reinstate from combined index col
    magplaspos_rdf = magplaspos_rdf.drop(columns=['time'])
    magplaspos_rdf['time'] = magplaspos_rdf.index

    #produce recarray with correct datatypes
    time_stamps = magplaspos_rdf['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    ace=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    ace = ace.view(np.recarray) 

    ace.time=dt_lst
    ace.bx=magplaspos_rdf['bx']
    ace.by=magplaspos_rdf['by']
    ace.bz=magplaspos_rdf['bz']
    ace.bt=magplaspos_rdf['bt']
    ace.vx=magplaspos_rdf['vx']
    ace.vy=magplaspos_rdf['vy']
    ace.vz=magplaspos_rdf['vz']
    ace.vt=magplaspos_rdf['vt']
    ace.np=magplaspos_rdf['np']
    ace.tp=magplaspos_rdf['tp']
    ace.x=magplaspos_rdf['x']
    ace.y=magplaspos_rdf['y']
    ace.z=magplaspos_rdf['z']
    ace.r=magplaspos_rdf['r']
    ace.lat=magplaspos_rdf['lat']
    ace.lon=magplaspos_rdf['lon']

    #dump to pickle file
    header='Science level 2 solar wind magnetic field (MFI), plasma (SWE), and positions from ACE, ' + \
    'obtained from https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb '+ \
    'Timerange: '+ace.time[0].strftime("%Y-%b-%d %H:%M")+' to '+ace.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by ace.time, ace.bx, etc. '+\
    'Total number of data points: '+str(ace.size)+'. '+\
    'Units are btxyz [nT, GSM], vtxyz [km/s, GSM], heliospheric position x/y/z/r/lon/lat [km, degree, GSM]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([ace,header], open(ace_path+'ace_gsm.p', "wb"))