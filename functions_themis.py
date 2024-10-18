import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
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
from spacepy import pycdf

"""
THEMIS DATA PATH
"""

themis_path='/Volumes/External/data/themis/'


"""
THEMIS DOWNLOAD DATA
#https://cdaweb.gsfc.nasa.gov/pub/data/themis/
"""


def download_themis_mag(probe:str, start_timestamp, end_timestamp, path=f'{themis_path}'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'{probe}_l2_fgm_{date_str}_v01'
        if os.path.isfile(f"{path}/{probe}/mag/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.') 
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/themis/{probe}/l2/fgm/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{probe}/mag/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
            except Exception as e:
                print('ERROR', e, data_item_id)
        start += timedelta(days=1)


def download_themis_plas(probe:str, start_timestamp, end_timestamp, path=f'{themis_path}'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'{probe}_l2_esa_{date_str}_v01'
        if os.path.isfile(f"{path}/{probe}/plas/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.') 
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/themis/{probe}/l2/esa/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{probe}/plas/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
            except Exception as e:
                print('ERROR', e, data_item_id)
        start += timedelta(days=1)


def download_themis_orb(probe:str, start_timestamp, end_timestamp, path=f'{themis_path}'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}01'
        data_item_id = f'{probe}_or_ssc_{date_str}_v01'
        if os.path.isfile(f"{path}/{probe}/orb/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.') 
        else:
            try:
                data_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/themis/{probe}/ssc/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{probe}/orb/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
            except Exception as e:
                print('ERROR', e, data_item_id)
        start += timedelta(days=28) #can't use month so use 28 days


"""
THEMIS MAG DATA
"""


#Load single file from specific path using pycdf from spacepy
def get_themismag(probe:str, fp):
    """raw = gse"""
    #also available in gsm
    try:
        cdf = pycdf.CDF(fp) #can change to cdflib.CDF(fp)
        times = cdf[f'{probe}_fgs_time'][:]
        times_converted = []
        for i in range(len(times)):
            time_convert = datetime.fromtimestamp(times[i], timezone.utc)
            times_converted.append(time_convert)
        df = pd.DataFrame(times_converted, columns=['time'])
        bx, by, bz = cdf[f'{probe}_fgs_gse'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = cdf[f'{probe}_fgs_btotal'][:]
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_themismag_range(probe:str, start_timestamp, end_timestamp, path=f'{themis_path}'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'{probe}_l2_fgm_{date_str}_v01'
        fn = f'{path}/{probe}/mag/{data_item_id}.cdf'
        _df = get_themismag(f'{probe}',fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
THEMIS PLAS DATA
#note, quality flags are available for both mag and plas files i.e. "thb_peif_data_quality", "thb_fgm_fgs_quality"
#need to implement data filtering
"""


def get_themisplas(probe:str, fp):
    """raw = gse"""
    #also available in gsm
    try:
        cdf = pycdf.CDF(fp) #can change to cdflib.CDF(fp)
        times = cdf[f'{probe}_peif_time'][:]
        times_converted = []
        for i in range(len(times)):
            time_convert = datetime.fromtimestamp(times[i], timezone.utc)
            times_converted.append(time_convert)
        df = pd.DataFrame(times_converted, columns=['time'])
        df['np'] = cdf[f'{probe}_peif_density'][:]
        df['tp'] = cdf[f'{probe}_peif_avgtemp'][:]
        vx, vy, vz = cdf[f'{probe}_peif_velocity_gse'][:].T #note thermal velocity is also available
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps
def get_themisplas_range(probe:str, start_timestamp, end_timestamp, path=f'{themis_path}'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'{probe}_l2_esa_{date_str}_v01'
        fn = f'{path}/{probe}/plas/{data_item_id}.cdf'
        _df = get_themisplas(f'{probe}',fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
THEMIS POSITION DATA
#original files contain range of co-ord systems
#units originally in Earth radii, code converts to km
"""


def get_themisorb(probe:str, coord_sys:str, fp):
    """options for GSE, GSM, GEO, GM, SM"""
    R_E = 6378.16 #original cdf file in earth radii, convert to km
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        x, y, z = cdf[f'XYZ_{coord_sys}'][:].T
        df['x'] = x*R_E
        df['y'] = y*R_E
        df['z'] = z*R_E
        df['r'] = cdf['RADIUS'][:]*R_E
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df



