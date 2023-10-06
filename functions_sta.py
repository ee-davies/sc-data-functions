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