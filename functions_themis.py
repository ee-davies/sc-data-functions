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


