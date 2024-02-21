import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spiceypy
import os.path
import glob
import urllib.request
from urllib.request import urlopen
import json
import astrospice
from sunpy.coordinates import HeliocentricInertial, HeliographicStonyhurst
from bs4 import BeautifulSoup
import cdflib
import pickle


stereoa_path='/Users/emmadavies/Documents/Data-test/stereoa/'

## function to download stereo merged impact data from nasa spdf service
## files are yearly
def download_stereoa_merged(start_timestamp, end_timestamp=datetime.utcnow(), path=stereoa_path+'impact/merged/level2/'):
    start = start_timestamp.year
    end = end_timestamp.year + 1
    while start < end:
        year = start
        date_str = f'{year}0101'
        try: 
            data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/l2/impact/magplasma/1min/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('sta_l2_magplasma_1m_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start+=1


#function to read in yearly cdf file 
#also filters bad data values
#creates pandas df 
def get_stereoa_merged(fp):
    """raw = rtn"""
    try:
        cdf = cdflib.CDF(fp)
        t1 = cdflib.cdfepoch.to_datetime(cdf.varget('Epoch'))
        df = pd.DataFrame(t1, columns=['time'])
        bx, by, bz = cdf['BFIELDRTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = cdf['BTOTAL']
        df['np'] = cdf['Np']
        df['tp'] = cdf['Tp']
        df['vt'] = cdf['Vp']
        cols = ['bx', 'by', 'bz', 'bt', 'np', 'tp', 'vt']
        for col in cols:
            df[col].mask(df[col] < -9.999E29 , pd.NA, inplace=True)
        df['vx'] = cdf['Vr_Over_V_RTN']*df['vt']
        df['vy'] = cdf['Vt_Over_V_RTN']*df['vt']
        df['vz'] = cdf['Vn_Over_V_RTN']*df['vt']
        v_cols = ['vx', 'vy', 'vz']
        for v_col in v_cols:
            df[v_col].mask(df[v_col] < -9.999E29 , pd.NA, inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


# uses get_stereoa_merged function to load multiple years of data 
# end timestamp can be modified, but default is set as now 
def get_stereoa_merged_range(start_timestamp, end_timestamp=datetime.utcnow(), path=stereoa_path+'impact/merged/level2/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df=None
    start = start_timestamp.year
    end = datetime.utcnow().year + 1
    while start < end:
        year = start
        date_str = f'{year}0101'
        try: 
            fn = glob.glob(path+f'sta_l2_magplasma_1m_{date_str}*')[0]
            _df = get_stereoa_merged(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += 1
    timemask = (df['time']>=start_timestamp) & (df['time']<=end_timestamp)
    df = df[timemask]
    return df


def download_sta_beacon_mag(path="/Volumes/External/Data/STEREO-A/beacon/impact"):
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'sta_lb_impact_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon/{year}'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_sta_beacon_plas(path="/Volumes/External/Data/STEREO-A/beacon/plastic"):
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'sta_lb_pla_browse_{date_str}_v14'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon_plastic/{year}'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def get_sta_beacon_plas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch1', 'Bulk_Speed', 'Vr_RTN', 'Vt_RTN', 'Vn_RTN', 'Density', 'Temperature_Inst'], ['timestamp', 'v_bulk', 'v_x', 'v_y', 'v_z', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_sta_beacon_mag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch_MAG'], ['timestamp'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['MAGBField'][:].T
        df['b_x'] = bx
        df['b_y'] = by
        df['b_z'] = bz
        df['b_tot'] = np.linalg.norm(df[['b_x', 'b_y', 'b_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_sta_beacon_mag_7days(path="/Volumes/External/Data/STEREO-A/beacon/impact"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_impact_{date_str}_v02.cdf'
        _df = get_sta_beacon_mag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def get_sta_beacon_plas_7days(path="/Volumes/External/Data/STEREO-A/beacon/plastic"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_pla_browse_{date_str}_v14.cdf'
        _df = get_sta_beacon_plas(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    df = filter_bad_data(df, 'temperature', -1E30)
    df = filter_bad_data(df, 'density', -1E30)
    df = filter_bad_data(df, 'v_bulk', -1E30)
    df = filter_bad_data(df, 'v_x', -1E30)
    df = filter_bad_data(df, 'v_y', -1E30)
    df = filter_bad_data(df, 'v_z', -1E30)
    return df