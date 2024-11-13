import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
from scipy import constants
# import spiceypy
# import os
import glob
import urllib.request
import os.path
import pickle

ace_path = "/Volumes/External/Data/ACE/"


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def download_ace_swe(start_timestamp, end_timestamp, path="/Volumes/External/Data/ACE/swe"):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'ac_h0_swe_{date_str}_v11'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_ace_mag(start_timestamp, end_timestamp, path="/Volumes/External/Data/ACE/mfi"):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'ac_h0_mfi_{date_str}_v07'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def get_acemag_rtn(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Magnitude'], ['Timestamp', 'B_TOT'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSEc'][:].T
        df['B_R'] = -1 * bx
        df['B_T'] = -1 * by
        df['B_N'] = bz
        df['B_TOT'].mask((df['B_TOT'] < -9.99e+30), inplace=True)
        df['B_R'].mask((df['B_R'] < -9.99e+30), inplace=True)
        df['B_T'].mask((df['B_T'] < -9.99e+30), inplace=True)
        df['B_N'].mask((df['B_N'] < -9.99e+30), inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_acemag_gse(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Magnitude'], ['timestamp', 'b_tot'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['BGSEc'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df['b_tot'].mask((df['b_tot'] < -9.99e+30), inplace=True)
        df['b_x'].mask((df['b_x'] < -9.99e+30), inplace=True)
        df['b_y'].mask((df['b_y'] < -9.99e+30), inplace=True)
        df['b_z'].mask((df['b_z'] < -9.99e+30), inplace=True)
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
        df['bt'].mask((df['bt'] < -9.99e+30), inplace=True)
        df['bx'].mask((df['bx'] < -9.99e+30), inplace=True)
        df['by'].mask((df['by'] < -9.99e+30), inplace=True)
        df['bz'].mask((df['bz'] < -9.99e+30), inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_aceswe_rtn(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Vp', 'Np', 'Tpr'], ['timestamp', 'v_bulk', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vr, vt, vn = cdf['V_RTN'][:].T
        df['v_x'] = vr
        df['v_y'] = vt
        df['v_z'] = vn
        df['temperature'].mask((df['temperature'] < -9.99e+30), inplace=True)
        df['density'].mask((df['density'] < -9.99e+30), inplace=True)
        df['v_bulk'].mask((df['v_bulk'] < -9.99e+30), inplace=True)
        df['v_x'].mask((df['v_x'] < -9.99e+30), inplace=True)
        df['v_y'].mask((df['v_y'] < -9.99e+30), inplace=True)
        df['v_z'].mask((df['v_z'] < -9.99e+30), inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_aceswe_gse(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'Vp', 'Np', 'Tpr'], ['timestamp', 'v_bulk', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['V_GSE'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['temperature'].mask((df['temperature'] < -9.99e+30), inplace=True)
        df['density'].mask((df['density'] < -9.99e+30), inplace=True)
        df['v_bulk'].mask((df['v_bulk'] < -9.99e+30), inplace=True)
        df['v_x'].mask((df['v_x'] < -9.99e+30), inplace=True)
        df['v_y'].mask((df['v_y'] < -9.99e+30), inplace=True)
        df['v_z'].mask((df['v_z'] < -9.99e+30), inplace=True)
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


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


#positions in units of km
def get_acepos_gsm(fp):
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        x, y, z = cdf['SC_pos_GSM'][:].T
        df['x'] = x
        df['y'] = y
        df['z'] = z
        r, lat, lon = cart2sphere(x,y,z)
        df['r'] = r
        df['lat'] = lat
        df['lon'] = lon
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_acemag_rtn_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/mfi'):
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
        _df = get_acemag_rtn(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_acemag_gse_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/mfi'):
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
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_acemag_gsm_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/mfi'):
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


def get_aceswe_rtn_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/swe'):
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
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_aceswe_gse_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/swe'):
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
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_aceswe_gsm_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/swe'):
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


def get_acepos_gsm_range(start_timestamp, end_timestamp, path=r'/Volumes/External/Data/ACE/swe'):
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
        _df = get_acepos_gsm(f'{path_fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


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


def create_ace_gsm_pkl(start_timestamp, end_timestamp): #just initial quick version, may fail easily
    #create dataframes for mag, plas, and position
    df_mag = get_acemag_gsm_range(start_timestamp, end_timestamp)
    df_swe = get_aceswe_gsm_range(start_timestamp, end_timestamp)
    df_pos = get_acepos_gsm_range(start_timestamp, end_timestamp)
    #resample all dfs
    rdf_mag = resample_df(df_mag, 1)
    rdf_swe = resample_df(df_swe, 1)
    rdf_pos = resample_df(df_pos, 1)
    #merge resampled dataframes
    mdf_mag_swe = merge_rdfs(rdf_mag, rdf_swe)
    mdf_mag_swe_pos = merge_rdfs(mdf_mag_swe, rdf_pos)
    #produce recarray with correct datatypes
    time_stamps = mdf_mag_swe_pos['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    ace=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    ace = ace.view(np.recarray) 

    ace.time=dt_lst
    ace.bx=mdf_mag_swe_pos['bx']
    ace.by=mdf_mag_swe_pos['by']
    ace.bz=mdf_mag_swe_pos['bz']
    ace.bt=mdf_mag_swe_pos['bt']
    ace.vx=mdf_mag_swe_pos['vx']
    ace.vy=mdf_mag_swe_pos['vy']
    ace.vz=mdf_mag_swe_pos['vz']
    ace.vt=mdf_mag_swe_pos['vt']
    ace.np=mdf_mag_swe_pos['np']
    ace.tp=mdf_mag_swe_pos['tp']
    ace.x=mdf_mag_swe_pos['x']
    ace.y=mdf_mag_swe_pos['y']
    ace.z=mdf_mag_swe_pos['z']
    ace.r=mdf_mag_swe_pos['r']
    ace.lat=mdf_mag_swe_pos['lat']
    ace.lon=mdf_mag_swe_pos['lon']

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


def calc_pressure_params(plasmag_df):
# assuming Tpr is the (isotropic) temperature
# in reality is temperature in radial direction: https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H0_SWE
    plasmag_df['p_plas'] = (plasmag_df['density']*10**6)*constants.k*plasmag_df['temperature']
    plasmag_df['p_mag'] = 0.5*(plasmag_df['b_tot']*10**(-9))**2./constants.mu_0
    plasmag_df['beta'] = plasmag_df['p_plas']/plasmag_df['p_mag']
    plasmag_df['p_tot'] = plasmag_df['p_plas'] + plasmag_df['p_mag']
    return plasmag_df