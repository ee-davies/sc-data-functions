import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import cdflib
import spiceypy
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


"""
ADITYA L1 SERVER DATA PATH
"""

aditya_path='/Volumes/External/data/aditya/'
kernels_path='/Volumes/External/data/kernels/'


"""
ADITYA DOWNLOAD DATA
MAG: MAG data available from 20240701
"""


#MAG DATA: https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=mag
# FORMAT example: /al1/protected/downloadData/mag/level2/2025/08/13/L2_AL1_MAG_20250813_V00.nc?mag

def download_adityamag(start_timestamp, end_timestamp, path=f'{aditya_path}'+'mag/'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'L2_AL1_MAG_{date_str}_V00'
        if os.path.isfile(f"{path}/{data_item_id}.nc") == True:
            print(f'{data_item_id}.nc has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://pradan1.issdc.gov.in/al1/protected/downloadData/mag/level2/{year}/{start.month:02}/{start.day:02}/{data_item_id}.nc?mag'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.nc")
                print(f'Successfully downloaded {data_item_id}.nc')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


"""
ADITYA POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=spice
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def aditya_furnish():
    """Main"""
    aditya_path = kernels_path+'aditya/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(aditya_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(aditya_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_aditya_pos(t, coord_sys='ECLIPJ2000'): 
    if spiceypy.ktotal('ALL') < 1:
        aditya_furnish()
    if coord_sys == 'GSE':
        try:
            pos = spiceypy.spkpos("ADITYA", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "EARTH")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
    else:
        try:
            pos = spiceypy.spkpos("ADITYA", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "SUN")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
        
