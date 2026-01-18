import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
# import os
import glob
import urllib.request
from urllib.request import urlopen
import os.path
import pickle
from bs4 import BeautifulSoup

import data_frame_transforms as data_transform
import position_frame_transforms as pos_transform
import functions_general as fgen

from functions_general import load_path


"""
WIND SERVER DATA PATH
"""

wind_path=load_path(path_name='wind_path')


"""
WIND BAD DATA FILTER and PATH FORMATTING
"""


def format_path(fp):
    """Formatting required for CDF package."""
    return fp.replace('/', '\\')


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
    df[col].mask(mask_vals)
    return df


"""
WIND DATA INFO
"""

# Wind: earliest time available:
wind_earliest_mag = datetime(1994,11,13)
wind_earliest_swe = datetime(1994,11,17)
wind_earliest_pos = datetime(1994,11,1)


"""
WIND DOWNLOAD DATA
"""


def download_wind_orb(start_timestamp, end_timestamp, path=wind_path):
    path = path+'orbit/'
    if not os.path.exists(path):
        os.makedirs(path)
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/wind/orbit/pre_or/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('wi_or_pre_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)


def download_wind_mag_rtn(start_timestamp, end_timestamp, path=wind_path):
    path = path+'mfi/rtn/'
    if not os.path.exists(path):
        os.makedirs(path)
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h3-rtn/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('wi_h3-rtn_mfi_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)
        

def download_wind_mag_h0(start_timestamp, end_timestamp, path=wind_path):
    path = path+'mfi/h0/'
    if not os.path.exists(path):
        os.makedirs(path)
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h0/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('wi_h0_mfi_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)


def download_wind_swe(start_timestamp, end_timestamp, path=wind_path):
    path = path+'swe/h1/'
    if not os.path.exists(path):
        os.makedirs(path)
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('wi_h1_swe_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start += timedelta(days=1)


def download_wind_swe_rtn(start_timestamp, end_timestamp, path=wind_path):
    path = path+'swe/rtn/'
    if not os.path.exists(path):
        os.makedirs(path)
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try: 
            data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1_rtn/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('wi_h1_swe_rtn_'+date_str):
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
WIND MAG DATA
"""


def get_windmag_gse(fp):
    """raw = gse"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:,0] for cdf_col, df_col in zip(['Epoch', 'BF1'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSE'][:].T
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


def get_windmag_gse_range(start_timestamp, end_timestamp, path=wind_path):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'       
        try:
            fn = glob.glob(f'{path}mfi/h0/wi_h0_mfi_{date_str}*.cdf')[0]
            _df = get_windmag_gse(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
                print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


def get_windmag_gsm(fp):
    """raw = gse"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:,0] for cdf_col, df_col in zip(['Epoch', 'BF1'], ['time', 'bt'])}
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


def get_windmag_gsm_range(start_timestamp, end_timestamp, path=wind_path):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}mfi/h0/wi_h0_mfi_{date_str}*.cdf')[0]
            _df = get_windmag_gsm(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
                print('ERROR', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


def get_windmag_rtn(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'BF1'], ['time', 'bt'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BRTN'][:].T
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


def get_windmag_rtn_range(start_timestamp, end_timestamp, path=wind_path):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}mfi/rtn/wi_h3-rtn_mfi_{date_str}*')[0]
            _df = get_windmag_rtn(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


def get_windmag_range(start_timestamp, end_timestamp, coord_sys:str):
    if coord_sys == 'GSE':
        df = get_windmag_gse_range(start_timestamp, end_timestamp)
    elif coord_sys == 'GSM':
        df = get_windmag_gsm_range(start_timestamp, end_timestamp)
    elif coord_sys == 'RTN':
        df = get_windmag_rtn_range(start_timestamp, end_timestamp)
    return df


"""
WIND PLASMA DATA (3DP hidden for now, and SWE)
"""


# def get_wind3dp_pm(fp):
#     cdf = pycdf.CDF(fp)
#     data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'P_DENS', 'P_TEMP'], ['timestamp', 'density', 'v_therm'])}
#     df = pd.DataFrame.from_dict(data)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     vx, vy, vz = cdf['P_VELS'][:].T #proton velocity vector is in GSE coordinates
#     df['v_x'] = vx
#     df['v_y'] = vy
#     df['v_z'] = vz
#     df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
#     # for col in cols_new[1:]:
#     #     df[col] = df[col].astype('float32')
#     return df


# def get_wind3dp_pm_range(start_timestamp, end_timestamp, path):
#     """Pass two datetime objects and grab .cdf files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start <= end:
#         year = start.year
#         month = start.month
#         day = start.day
#         fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v05.cdf'
#         #print(f'{path}/{fn}')
#         _df = get_wind3dp_pm(f'{path}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=1)
#     return df


# def get_wind3dp_pm_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/WIND/3dp/pm'):
#     """Pass two datetime objects and grab .cdf files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start <= end:
#         year = start.year
#         month = start.month
#         day = start.day
#         if start >= (datetime(2011, 12, 30).date()):
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v05.cdf'
#         elif (start <= datetime(2011, 12, 29).date()) & (start > datetime(2011, 1, 13).date()):
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v04.cdf'
#         elif (start <= datetime(2011, 1, 13).date()) & (start > datetime(2010, 5, 11).date()):
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v03.cdf'
#         elif (start <= datetime(2010, 5, 11).date()) & (start > datetime(2009, 5, 19).date()):
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v04.cdf'
#         elif (start <= datetime(2009, 5, 19).date()) & (start > datetime(2008, 1, 1).date()):
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v03.cdf'
#         elif start == datetime(2008, 1, 1).date():
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v05.cdf'
#         elif year <= 2007:
#             fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v03.cdf'
#         #fn = f'wi_pm_3dp_{year}{month:02}{day:02}_v05.cdf'
#         #print(f'{path}/{fn}')
#         _df = get_wind3dp_pm(f'{path}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=1)
#     return df


def get_windswe_gse(fp): #choice between nonlin or moments
    try:
        cdf = pycdf.CDF(fp)
        cols_raw = ['Epoch', 'Proton_V_nonlin', 'Proton_VX_nonlin', 'Proton_VY_nonlin',
                    'Proton_VZ_nonlin', 'Proton_W_nonlin', 'Proton_Np_nonlin']
        cols_new = ['time', 'vt', 'vx', 'vy', 'vz', 'tp', 'np']
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        for col in cols_new[1:]:
            df[col] = df[col].astype('float32')
        df = filter_bad_col(df, 'np', 9.99E04)
        df = filter_bad_col(df, 'tp', 9.99E04) #called tp but actually is v_therm
        df = filter_bad_col(df, 'vt', 9.99E04)
        df = filter_bad_col(df, 'vx', 9.99E04)
        df = filter_bad_col(df, 'vy', 9.99E04)
        df = filter_bad_col(df, 'vz', 9.99E04)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_windswe_gse_range(start_timestamp, end_timestamp, path=wind_path):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}swe/h1/wi_h1_swe_{date_str}*')[0]
            _df = get_windswe_gse(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    return df


def get_windswe_rtn(fp): #choice between nonlin or moments
    try:
        cdf = pycdf.CDF(fp)
        cols_raw = ['Epoch', 'Proton_V_nonlin', 'Proton_VR_nonlin', 'Proton_VT_nonlin',
                    'Proton_VN_nonlin', 'Proton_W_nonlin', 'Proton_Np_nonlin']
        cols_new = ['time', 'vt', 'vx', 'vy', 'vz', 'tp', 'np']
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
        df = pd.DataFrame.from_dict(data)
        df = filter_bad_col(df, 'np', 9.99E04)
        df = filter_bad_col(df, 'tp', 9.99E04) #called tp but actually is v_therm, km/s
        df = filter_bad_col(df, 'vt', 9.99E04)
        df = filter_bad_col(df, 'vx', 9.99E04)
        df = filter_bad_col(df, 'vy', 9.99E04)
        df = filter_bad_col(df, 'vz', 9.99E04)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_windswe_rtn_range(start_timestamp, end_timestamp, path=wind_path):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(f'{path}swe/rtn/wi_h1_swe_rtn_{date_str}*')[0]
            _df = get_windswe_rtn(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'. wi_h1_swe_rtn_{date_str} does not exist, converting from wi_h1_swe_{date_str} (GSE) instead...')
            try:
                fn_swe = glob.glob(f'{path}swe/h1/wi_h1_swe_{date_str}*')[0]
                _df_swe = get_windswe_gse(fn_swe)
                fn_pos = glob.glob(f'{path}orbit/wi_or_pre_{date_str}*')[0]
                _df_pos = get_wind_pos(fn_pos, 'GSE')
                _df = windswe_gse_to_rtn(_df_swe, _df_pos)
                if _df is not None:
                    if df is None:
                        df = _df.copy(deep=True)
                    else:
                        df = pd.concat([df, _df])
            except Exception as e:
                print('ERROR:', e, f'. wi_h1_swe_{date_str} does not exist either.')
        start += timedelta(days=1)
    return df


def windswe_gse_to_rtn(df_swe_gse, df_pos_gse):
    df_swe_heeq = data_transform.perform_plas_transform(df_swe_gse, 'GSE', 'HEEQ')
    df_pos_hee = pos_transform.GSE_to_HEE(df_pos_gse)
    df_pos_heeq = pos_transform.perform_transform(df_pos_hee, 'HEE', 'HEEQ')
    df_new_pos = data_transform.interp_to_newtimes(df_pos_heeq, df_swe_heeq)
    combined_df = data_transform.combine_dataframes(df_swe_heeq,df_new_pos)
    df_swe_rtn = data_transform.HEEQ_to_RTN_plas(combined_df)
    return df_swe_rtn


def get_windswe_range(start_timestamp, end_timestamp, coord_sys:str):
    if coord_sys == 'GSE':
        df = get_windswe_gse_range(start_timestamp, end_timestamp)
    elif coord_sys == 'GSM':
        df_gse = get_windswe_gse_range(start_timestamp, end_timestamp)
        df = data_transform.GSE_to_GSM_plas(df_gse)
    elif coord_sys == 'RTN':
        df = get_windswe_rtn_range(start_timestamp, end_timestamp)
    return df


"""
WIND PAD FUNCTIONS:
EHPD data from 3DP instrument
"""


def get_wind3dp_ehpd(start, end, input_dir='/Volumes/External/Data/WIND/3dp/ehpd/'):

    epoch_arr  = np.empty((0,), dtype='datetime64[ns]') 
    energy_arr = np.empty((0,15)) 
    pangle_arr = np.empty((0,8))
    flux_arr   = np.empty((0,8,15))

    while start <= end:
        fp = input_dir+'wi_ehpd_3dp_'+start.strftime("%Y")+start.strftime("%m")+start.strftime("%d")+'_v02.cdf'
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'ENERGY', 'PANGLE', 'FLUX'], ['time', 'energy', 'pangle', 'flux'])}
        epoch  = pd.to_datetime(data['time'])  # array of dim X  (time entries)
        energy = data['energy']  # array of dim [X,Y] (time entries and 15 energy channels) 
        pangle = data['pangle']  # array of dim [X,Z] (time entries and 8 angles) 
        flux   = data['flux']    # array of dim [X,Z,Y] (time entries, 8 angles, 15 energy channels) 
        epoch_arr  = np.concatenate((epoch_arr, epoch))
        energy_arr = np.concatenate((energy_arr, energy))
        pangle_arr = np.concatenate((pangle_arr, pangle))
        flux_arr   = np.concatenate((flux_arr, flux))
        start+=timedelta(days=1)
    epoch_arr = pd.to_datetime(epoch_arr)
    return epoch_arr, energy_arr, pangle_arr, flux_arr


def wind_pad_array(epoch_arr, pangle_arr, flux_arr):
    x_arr = epoch_arr.to_numpy()
    y_arr = np.nanmean(pangle_arr, axis=0) 
    z = np.sum(flux_arr, axis=2) #z = np.sum(flux_arr[:, :, i:j], axis=2) if want to select specific channel range
    z_arr = z.T
    return x_arr, y_arr, z_arr


"""
WIND POSITION FUNCTIONS:
Wind has no spice kernels, but does have predicted orbit files including J2000 GCI, GSE, GSM, HEC
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def get_wind_pos(fp, coord_sys='GSE'):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        x, y, z = cdf[f'{coord_sys}_POS'][:].T
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


def get_wind_positions(start_timestamp, end_timestamp, coord_sys='GSE', path=wind_path+'orbit/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    coord_sys2 = coord_sys
    if coord_sys2 == 'HEE':
        coord_sys = 'GSE'
    elif coord_sys2 == 'HEEQ':
        coord_sys = 'GSE'
    elif coord_sys2 == 'HAE':
        coord_sys = 'GSE'
    df=None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        try:
            fn = glob.glob(str(path)+f'wi_or_pre_{date_str}_*')[0]
            _df = get_wind_pos(fn, coord_sys)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += timedelta(days=1)
    if coord_sys2 == 'HEE':
        df_hee = pos_transform.GSE_to_HEE(df)
        return df_hee
    elif coord_sys2 == 'HEEQ':
        df_heeq = pos_transform.GSE_to_HEEQ(df)
        return df_heeq
    elif coord_sys2 == 'HAE':
        df_hae = pos_transform.GSE_to_HAE(df)
        return df_hae
    else:
        return df


"""
WIND DATA SAVING FUNCTIONS:
"""


def create_wind_mag_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=wind_path):
    df_mag = get_windmag_range(start_timestamp, end_timestamp, coord_sys)
    if df_mag is None:
        print(f'Wind MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
    rarr = fgen.make_mag_recarray(df_mag)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Science level magnetometer (MFI) data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Magnetometer data available in original cadence of 1 min, units in nT.'+\
    ' Available coordinate systems include GSE, GSM, and RTN. GSE and GSM data are taken directly from wi_h0_mfi files, RTN data from wi_h3-rtn_mfi.'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.time, wind.bt, wind.bx, wind.by, wind.bz.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'wind_mag_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


def create_wind_plas_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=wind_path):
    df_plas = get_windswe_range(start_timestamp, end_timestamp, coord_sys)
    if df_plas is None:
        print(f'Wind SWE data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
    rarr = fgen.make_plas_recarray(df_plas)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Science level plasma (SWE) data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Plasma data available in original cadence of ~ 92 seconds. Parameters obtained from non-linear fitting to the ion CDF, rather than moment analysis (available by request).'+\
    ' Units: proton velocity [km/s], proton temperature => proton thermal speed [km/s], proton number density [n/cc].'+\
    ' Available coordinate systems include GSE, GSM, and RTN. GSE are taken directly from wi_h1_swe files, GSM data has been converted using data_frame_transforms based on Hapgood 1992.'+\
    ' RTN data is taken directly from wi_h1_swe_rtn, except for the years 2010--2014 (inclusive). Where RTN files are unavailable, original GSE files are converted to RTN using data_frame_transforms (Hapgood 1992 and spice kernels).'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.time, wind.vt, wind.vx, wind.vy, wind.vz, wind.np, and wind.tp.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'wind_plas_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


def create_wind_pos_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=wind_path):
    df_pos = get_wind_positions(start_timestamp, end_timestamp, coord_sys)
    if df_pos is None:
        print(f'Wind predicted orbit data is empty for this timerange')
        df_pos = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
    rarr = fgen.make_pos_recarray(df_pos)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Predicted orbit data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/orbit/pre_or/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Orbit available in original cadence of 10 minutes.'+\
    ' Units: xyz [km], r [AU], lat/lon [deg].'+\
    ' Available coordinate systems include GSE, GSM, J2000 GCI, HEC, HEE, HAE, and HEEQ. GSE, GSM, J2000 GCI and HEC are taken directly from wi_or_pre files, others using data_frame_transforms based on Hapgood 1992 and spice kernels.'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.x, wind.y, wind.z, wind.r, wind.lat, and wind.lon.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'wind_pos_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


def make_yearly_pkl_files(start_timestamp, end_timestamp, data_type:str, coord_sys:str, output_path=wind_path):
    start = start_timestamp.year
    end = end_timestamp.year
    while start <= end:
        if data_type == 'MAG':
            create_wind_mag_pkl(datetime(start, 1, 1), datetime(start, 12, 31), coord_sys, output_path)
        elif data_type == 'PLAS':
            create_wind_plas_pkl(datetime(start, 1, 1), datetime(start, 12, 31), coord_sys, output_path)
        elif data_type == 'POS':
            create_wind_pos_pkl(datetime(start, 1, 1), datetime(start, 12, 31), coord_sys, output_path)
        print(f'Finished creating pkl file for Wind {data_type} {start}')
        start += 1


def create_wind_all_pkl(start_timestamp, end_timestamp, data_coord_sys='RTN', pos_coord_sys='HEEQ', output_path=wind_path):
    #MAG DATA
    df_mag = get_windmag_range(start_timestamp, end_timestamp, data_coord_sys)
    if df_mag is None:
        print(f'Wind MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
    #PLASMA DATA
    df_plas = get_windswe_range(start_timestamp, end_timestamp, data_coord_sys)
    if df_plas is None:
        print(f'WIND SWE data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
        plas_rdf = df_plas
    else:
        plas_rdf = df_plas.set_index('time').resample('1min').mean().reset_index(drop=False)
        plas_rdf.set_index(pd.to_datetime(plas_rdf['time']), inplace=True)
        if mag_rdf.shape[0] != 0:
            plas_rdf = plas_rdf.drop(columns=['time'])
    #Combine MAG and PLASMA dfs
    magplas_rdf = pd.concat([mag_rdf, plas_rdf], axis=1)
    #some timestamps may be NaT so after joining, drop time column and reinstate from combined index col
    magplas_rdf = magplas_rdf.drop(columns=['time'])
    magplas_rdf['time'] = magplas_rdf.index
    #POSITION DATA
    df_pos = get_wind_positions(start_timestamp, end_timestamp, pos_coord_sys)
    if df_pos is None:
        print(f'Wind POS data is empty for this timerange')
        df_pos = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
        pos_rdf = df_pos
    else:
        pos_rdf = df_pos.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False)
        pos_rdf.set_index(pd.to_datetime(pos_rdf['time']), inplace=True)
        if pos_rdf.shape[0] != 0:
            pos_rdf = pos_rdf.drop(columns=['time'])
    #Combine again: 
    comb_df_nans = pd.concat([magplas_rdf, pos_rdf], axis=1)
    comb_df = comb_df_nans[comb_df_nans['bt'].notna()]
    #Create rec array
    rarr = fgen.make_combined_recarray(comb_df)
    #Make header for pickle file
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    header='Science level magnetometer (MFI) data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/.'+\
    ' Available coordinate systems include GSE, GSM, and RTN. GSE and GSM data are taken directly from wi_h0_mfi files, RTN data from wi_h3-rtn_mfi.'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.time, wind.bt, wind.bx, wind.by, wind.bz.'+\
    ' Science level plasma (SWE) data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/.'+\
    ' Parameters obtained from non-linear fitting to the ion CDF, rather than moment analysis (available by request).'+\
    ' Units: proton velocity [km/s], proton temperature => proton thermal speed [km/s], proton number density [n/cc].'+\
    ' Available coordinate systems include GSE, GSM, and RTN. GSE are taken directly from wi_h1_swe files, GSM data has been converted using data_frame_transforms based on Hapgood 1992.'+\
    ' RTN data is taken directly from wi_h1_swe_rtn, except for the years 2010--2014 (inclusive). Where RTN files are unavailable, original GSE files are converted to RTN using data_frame_transforms (Hapgood 1992 and spice kernels).'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.time, wind.vt, wind.vx, wind.vy, wind.vz, wind.np, and wind.tp.'+\
    ' Predicted orbit data from Wind, sourced from https://cdaweb.gsfc.nasa.gov/pub/data/wind/orbit/pre_or/.'+\
    ' Units: xyz [km], r [AU], lat/lon [deg].'+\
    ' Available coordinate systems include GSE, GSM, J2000 GCI, HEC, HEE, HAE, and HEEQ. GSE, GSM, J2000 GCI and HEC are taken directly from wi_or_pre files, others using data_frame_transforms based on Hapgood 1992 and spice kernels.'+\
    ' The data are available in a numpy recarray, fields can be accessed by wind.x, wind.y, wind.z, wind.r, wind.lat, and wind.lon.'+\
    ' All data resampled to cadence of 1 min. Position data has been linearly interpolated.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    pickle.dump([rarr,header], open(output_path+f'wind_data_{data_coord_sys}_pos_{pos_coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


# def transform_data(df, instrument, coord_system):
#     #TODO: apply logic to handle b and v data transformations here
#     if instrument == 'mag':
#         prefix = 'b'
#     elif instrument == 'swe' or instrument == '3dp_pm':
#         prefix = 'v'
#     if coord_system == 'rtn_approx':
#         df[f'{prefix}_x'] = -1 * df[f'{prefix}_x']
#         df[f'{prefix}_y'] = -1 * df[f'{prefix}_y']
#     else:
#         raise ValueError('transform does not exist yet, rtn_approx does exist')
#     return df