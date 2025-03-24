import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
import spiceypy
# import os
import glob
import urllib.request
import os.path
import pickle


"""
SOLAR ORBITER SERVER DATA PATH
"""

juice_path='/Volumes/External/data/juice/'
kernels_path='/Volumes/External/data/kernels/'


"""
JUICE POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def juice_furnish():
    juice_path = kernels_path+'juice/'
    generic_path = kernels_path+'generic/'
    juice_kernels = os.listdir(juice_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in juice_kernels:
        spiceypy.furnsh(os.path.join(juice_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_juice_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        juice_furnish()
    pos = spiceypy.spkpos("JUICE", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_juice_positions(time_series):
    positions = []
    for t in time_series:
        position = get_juice_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions



def get_juice_positions_daily(start, end, cadence=1):
    t = start
    positions = []
    while t < end:
        position = get_juice_pos(t)
        positions.append(position)
        t += timedelta(days=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_juice_positions_hourly(start, end, cadence=1):
    t = start
    positions = []
    while t < end:
        position = get_juice_pos(t)
        positions.append(position)
        t += timedelta(hours=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_juice_positions_minute(start, end, cadence=1):
    t = start
    positions = []
    while t < end:
        position = get_juice_pos(t)
        positions.append(position)
        t += timedelta(minutes=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions