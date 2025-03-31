import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import cdflib
import spiceypy
import glob
import urllib.request
import os.path
import pickle


"""
LAGRANGE POINT functions: positions of L1, L2, L4 and L4 from generic and specific lagrange spice kernels
info: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/lagrange_point/AAREADME_Lagrange_point_SPKs.txt
L1: 391, L2: 392, L4: 394, L5: 395
"""


kernels_path='/Volumes/External/data/kernels/'


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)

"""
FURNISH KERNELS
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/lagrange_point/
"""


def lagrange_furnish():
    """Main"""
    lagrange_path = kernels_path+'lagrange/'
    generic_path = kernels_path+'generic/'
    lagrange_kernels = os.listdir(lagrange_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in lagrange_kernels:
        spiceypy.furnsh(os.path.join(lagrange_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_lagrange_pos(t, lagrange_point): #doesn't automatically furnish, furnish first
    if lagrange_point == "L1":
        lagrange_code = "391"
    elif lagrange_point == "L2":
        lagrange_code = "392"
    elif lagrange_point == "L3":
        print("L3 position calculation is unavailable")
        return
    elif lagrange_point == "L4":
        lagrange_code = "394"
    elif lagrange_point == "L5":
        lagrange_code = "395"
    try:
        pos = spiceypy.spkpos(lagrange_code, spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [None, None, None, None]


def get_lagrange_positions_daily(start, end, lagrange_point, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_lagrange_pos(t, lagrange_point)
        positions.append(position)
        t += timedelta(days=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions


def get_lagrange_positions_hourly(start, end, lagrange_point, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_lagrange_pos(t, lagrange_point)
        positions.append(position)
        t += timedelta(hours=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions


def get_lagrange_positions_minute(start, end, lagrange_point, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_lagrange_pos(t, lagrange_point)
        positions.append(position)
        t += timedelta(minutes=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    if dist_unit == 'au':
        df_positions.x = df_positions.x/1.495978707E8 
        df_positions.y = df_positions.y/1.495978707E8
        df_positions.z = df_positions.z/1.495978707E8
    if ang_unit == 'rad':
        df_positions.lat = df_positions.lat * np.pi / 180
        df_positions.lon = df_positions.lon * np.pi / 180
    spiceypy.kclear()
    return df_positions