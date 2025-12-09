import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import glob
import urllib.request
from urllib.request import urlopen
import os.path
import pickle
from bs4 import BeautifulSoup

from tqdm import tqdm

import data_frame_transforms as data_transform
import position_frame_transforms as pos_transform
import functions_general as fgen

from functions_general import load_path

"""
ACE DATA PATH
"""

ace_path = load_path(path_name='ace_path')
print(f"ACE data path loaded: {ace_path}")


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
    """Efficiently get ACE position data from MAG CDF files in a date range."""
    start = start_timestamp.date()
    end = end_timestamp.date()

    # List all files once
    all_files = sorted(glob.glob(f'{path}/ac_h0_mfi_*.cdf'))

    # Filter by date range
    files_to_load = []
    for f in all_files:
        try:
            date_str = f.split('_')[-2]
            file_date = pd.to_datetime(date_str, format='%Y%m%d').date()
            if start <= file_date <= end:
                files_to_load.append(f)
        except Exception:
            continue

    if not files_to_load:
        print("No matching CDF files found.")
        return None

    # Read and combine
    dfs = []
    for f in tqdm(files_to_load):
        _df = get_acepos(f, coord_sys)
        if _df is not None:
            dfs.append(_df)

    if not dfs:
        return None

    # Concatenate once
    df = pd.concat(dfs, ignore_index=True)
    return df

def get_acepos_frommag_range_alt(start_timestamp, end_timestamp, coord_sys='GSE', path=ace_path+'mfi'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end:
        print(f"Getting ACE position from MAG for {start}...")
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
ACE SWEPAM PAD FUNCTIONS:
"""
#OLD FUNCTIONS DIRECTLY COPIED, DO NOT USE, NEED UPDATING!


def get_aceswepam_pad_range(start, end, input_dir='/Volumes/External/Data/ACE/swepam/'):

    # function loading ACE SWEPAM electron pitch angle data
    # from https://izw1.caltech.edu/ACE/ASC/DATA/level3/swepam/
    # column format: 0 year, 1 DOY, 2 hour, 3 min, 4 sec
    #				 5 distribution function by pitch angle [20]
    # Pitch angle bins are 9 degrees wide, covering 0 to 180 degrees.
    # Energy channels [eV]: 272 eV

    data = {key: [] for key in ['Timestamp', '2d_data']}
    
    while start <= end:
        #try:
        fp = input_dir + 'ace_swepam_pa272ev_'+start.strftime("%Y")+"-"+start.strftime("%j")+'_v1.dat'
        data_tmp = np.loadtxt(fp, skiprows=12, dtype='str')
        df_tmp = pd.DataFrame({'year': data_tmp[:, 0],
                            'doy': data_tmp[:, 1], 
                           'hour': data_tmp[:, 2],
                            'min': data_tmp[:, 3],
                            'sec': data_tmp[:, 4]})
        df_tmp['Timestamp'] = df_tmp['year']+'-'+df_tmp['doy']+' '+df_tmp['hour']+':'+df_tmp['min']+':'+df_tmp['sec']  
        data['Timestamp'].append(pd.to_datetime(df_tmp['Timestamp'], format="%Y-%j %H:%M:%S"))
        data['2d_data'].append(data_tmp[:, 6:])
        print(start, len(data['Timestamp']), len(data['2d_data']))
        #except Exception as e:
        #    print('ERROR:', e, fp)
        #    df = None
        start+=timedelta(days=1)

    data['Timestamp'] = [item for subl in data['Timestamp'] for item in subl] # flattens list of timestamps
    data['2d_data'] = [item for subl in data['2d_data'] for item in subl] # flattens list of timestamps
    
    data['2d_data'] = np.asarray(data['2d_data'], dtype=float)   # convert from list to numpy array
    data['2d_data'][data['2d_data'] == -1e+31] = np.nan				# clean up bad flags
    data['2d_data'][data['2d_data'] == np.inf] = np.nan				# clean up bad flags
    data['2d_data'][data['2d_data'] == -np.inf] = np.nan			# clean up bad flags
    data['2d_data'][data['2d_data'] == 0.0] = np.nan				# clean up bad flags
    data['2d_data'] = data['2d_data'].reshape(-1, data['2d_data'].shape[-1]) # flattens array (across days)

    return data


def ace_pad_xyz(icme_start, mo_end):
    swepam_data = get_data.get_aceswepam_pad_range(icme_start, mo_end)
    x_swepam = swepam_data['Timestamp']
    y_swepam = np.linspace(4.5, 175.5, 20)
    z_swepam = np.reshape(swepam_data['2d_data'], (len(x_swepam), len(y_swepam)))
    return x_swepam, y_swepam, z_swepam


def reshape_acepad_array(z_swepam):
    z_new = z_swepam.tolist()
    z0 = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []
    z5 = []
    z6 = []
    z7 = []
    z8 = []
    z9 = []
    z10 = []
    z11 = []
    z12 = []
    z13 = []
    z14 = []
    z15 = []
    z16 = []
    z17 = []
    z18 = []
    z19 = []
    for i in range(len(z_new)):
        z0_ = z_new[i][0]
        z0 = np.append(z0, z0_)
        z1_ = z_new[i][1]
        z1 = np.append(z1, z1_)
        z2_ = z_new[i][2]
        z2 = np.append(z2, z2_)
        z3_ = z_new[i][3]
        z3 = np.append(z3, z3_)
        z4_ = z_new[i][4]
        z4 = np.append(z4, z4_)
        z5_ = z_new[i][5]
        z5 = np.append(z5, z5_)
        z6_ = z_new[i][6]
        z6 = np.append(z6, z6_)
        z7_ = z_new[i][7]
        z7 = np.append(z7, z7_)
        z8_ = z_new[i][8]
        z8 = np.append(z8, z8_)
        z9_ = z_new[i][9]
        z9 = np.append(z9, z9_)
        z10_ = z_new[i][10]
        z10 = np.append(z10, z10_)
        z11_ = z_new[i][11]
        z11 = np.append(z11, z11_)
        z12_ = z_new[i][12]
        z12 = np.append(z12, z12_)
        z13_ = z_new[i][13]
        z13 = np.append(z13, z13_)
        z14_ = z_new[i][14]
        z14 = np.append(z14, z14_)
        z15_ = z_new[i][15]
        z15 = np.append(z15, z15_)
        z16_ = z_new[i][16]
        z16 = np.append(z16, z16_)
        z17_ = z_new[i][17]
        z17 = np.append(z17, z17_)
        z18_ = z_new[i][18]
        z18 = np.append(z18, z18_)
        z19_ = z_new[i][19]
        z19 = np.append(z19, z19_)
    return z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19


def ace_pad_z_df(x_swepam, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19):
    df = pd.DataFrame(x_swepam, columns=['Timestamp'])
#     z_new = z.tolist()
    df['z0'] = z0
    df['z1'] = z1
    df['z2'] = z2
    df['z3'] = z3
    df['z4'] = z4
    df['z5'] = z5
    df['z6'] = z6
    df['z7'] = z7
    df['z8'] = z8
    df['z9'] = z9
    df['z10'] = z10
    df['z11'] = z11
    df['z12'] = z12
    df['z13'] = z13
    df['z14'] = z14
    df['z15'] = z15
    df['z16'] = z16
    df['z17'] = z17
    df['z18'] = z18
    df['z19'] = z19
    return df


def make_final_acepad_array(final_df, y_swepam):
    z_arr = final_df[['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19']].to_numpy()
    x_arr = final_df['New_Timestamp'].to_numpy()
    y_arr = y_swepam
    return x_arr, y_arr, z_arr


"""
ACE DATA SAVING FUNCTIONS:
"""


def create_ace_mag_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=ace_path):
    df_mag = get_acemag_range(start_timestamp, end_timestamp, coord_sys)
    if df_mag is None:
        print(f'ACE MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
    rarr = fgen.make_mag_recarray(df_mag)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Science level magnetometer (MFI) data from ACE, sourced from https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Magnetometer data available in original cadence of ~15 seconds, units in nT.'+\
    ' Available coordinate systems include GSE, GSM, and RTN. GSE and GSM data are taken directly from ac_h0_mfi files, RTN data is converted using data_transforms (Hapgood 1992 and spice kernels).'+\
    ' The data are available in a numpy recarray, fields can be accessed by ace.time, ace.bt, ace.bx, ace.by, ace.bz.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'ace_mag_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


def create_ace_plas_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=ace_path):
    df_plas = get_aceswe_range(start_timestamp, end_timestamp, coord_sys)
    if df_plas is None:
        print(f'ACE SWE data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
    rarr = fgen.make_plas_recarray(df_plas)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Science level plasma (SWE) data from ACE, sourced from https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Plasma data available in original cadence of 64 seconds. Proton temp and density derived by integrating the ion distribution function.'+\
    ' Units: proton velocity [km/s], proton temperature -> radial component of temperature tensor, proton number density [cm-3].'+\
    ' Available coordinate systems include GSE, GSM, and RTN. All are taken directly from ac_h0_swe_ files.'+\
    ' The data are available in a numpy recarray, fields can be accessed by ace.time, ace.vt, ace.vx, ace.vy, ace.vz, ace.np, and ace.tp.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'ace_plas_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


def create_ace_pos_pkl(start_timestamp, end_timestamp, coord_sys:str, output_path=ace_path):
    df_pos = get_acepos_frommag_range(start_timestamp, end_timestamp, coord_sys)
    if df_pos is None:
        df_pos = get_acepos_fromswe_range(start_timestamp, end_timestamp, coord_sys)
        if df_pos is None:
            print(f'ACE orbit data is empty for this timerange')
            df_pos = pd.DataFrame({'time':[], 'x':[], 'y':[], 'z':[], 'r':[], 'lat':[], 'lon':[]})
    rarr = fgen.make_pos_recarray(df_pos)
    start = start_timestamp.date()
    end = end_timestamp.date()
    datestr_start = f'{start.year}{start.month:02}{start.day:02}'
    datestr_end = f'{end.year}{end.month:02}{end.day:02}'
    #create header
    header='Orbit data from ACE, sourced from mag or plasma data files from https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/ or https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/.'+\
    ' Timerange: '+rarr.time[0].strftime("%Y-%b-%d %H:%M")+' to '+rarr.time[-1].strftime("%Y-%b-%d %H:%M")+'.'+\
    ' Orbit available in same cadence as data files: if taken from mag, ~15 seconds, if taken from swe, ~64 seconds.'+\
    ' Units: xyz [km], r [AU], lat/lon [deg].'+\
    ' Available coordinate systems include GSE and GSM. GSE and GSM are taken directly from files.'+\
    ' The data are available in a numpy recarray, fields can be accessed by ace.x, ace.y, ace.z, ace.r, ace.lat, and ace.lon.'+\
    ' Made with script by E. E. Davies (github @ee-davies, sc-data-functions). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    #dump to pickle file
    pickle.dump([rarr,header], open(output_path+f'ace_pos_{coord_sys}_{datestr_start}_{datestr_end}.p', "wb"))


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