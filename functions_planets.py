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
PLANETS functions: positions of planets from generic spice kernels
"""

kernels_path='/Volumes/External/data/kernels/'


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


"FURNISH GENERIC KERNELS"

def generic_furnish():
    """Main"""
    generic_path = kernels_path+'generic/'
    generic_kernels = os.listdir(generic_path)
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


""""
PLANET POSITIONS
options include:
        MERCURY BARYCENTER (1)  SATURN BARYCENTER (6)   MERCURY (199)
        VENUS BARYCENTER (2)    URANUS BARYCENTER (7)   VENUS (299)
        EARTH BARYCENTER (3)    NEPTUNE BARYCENTER (8)  MOON (301)
        MARS BARYCENTER (4)     PLUTO BARYCENTER (9)    EARTH (399)
        JUPITER BARYCENTER (5)  SUN (10)
"""


def get_planet_pos(t, planet): #doesn't automatically furnish, fix
    if spiceypy.ktotal('ALL') < 1:
        generic_furnish()
    pos = spiceypy.spkpos(planet, spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_planet_positions(time_series, planet):
    positions = []
    for t in time_series:
        position = get_planet_pos(t, planet)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_planet_positions_daily(start, end):
    t = start
    positions = []
    while t < end:
        position = get_planet_pos(t)
        positions.append(position)
        t += timedelta(days=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_planet_positions_minute(start, end):
    t = start
    positions = []
    while t < end:
        position = get_planet_pos(t)
        positions.append(position)
        t += timedelta(minutes=1)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions