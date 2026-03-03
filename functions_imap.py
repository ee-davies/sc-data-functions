import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
# import os
import urllib.request
import os.path
import pickle
import glob

from functions_general import load_path


"""
IMAP SERVER DATA PATH
"""

imap_path=load_path(path_name='imap_path')
print(f"IMAP path loaded: {imap_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
IMAP POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Coord_systems available: ECLIPJ2000, HEEQ, HEE, GSE
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def imap_furnish():
    """Main"""
    imap_path = kernels_path+'imap/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(imap_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(imap_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_imap_pos(t, coord_sys='ECLIPJ2000'): 
    if spiceypy.ktotal('ALL') < 1:
        imap_furnish()
    if coord_sys == 'GSE':
        try:
            pos = spiceypy.spkpos("IMAP", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "EARTH")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
    else:
        try:
            pos = spiceypy.spkpos("IMAP", spiceypy.datetime2et(t), f"{coord_sys}", "NONE", "SUN")[0] 
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
        
