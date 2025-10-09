import numpy as np
import pandas as pd
from datetime import timedelta
import spiceypy
import os.path

from .functions_general import load_path

"""
PLANETS functions: positions of planets from generic spice kernels
"""


# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


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


def get_planet_pos(t, planet, coord_sys): #doesn't automatically furnish, fix
    if spiceypy.ktotal('ALL') < 1:
        generic_furnish()
    try:
        pos = spiceypy.spkpos(planet, spiceypy.datetime2et(t), coord_sys, "NONE", "SUN")[0] #works for heliocentric coords
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        position = t, pos[0], pos[1], pos[2], r, lat, lon
        return position
    except Exception as e:
        print(e)
        return [t, None, None, None, None, None, None]


def get_planet_positions(time_series, planet, coord_sys):
    positions = []
    for t in time_series:
        position = get_planet_pos(t, planet, coord_sys)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_planet_positions_daily(start, end, planet, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_planet_pos(t, planet, coord_sys)
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


def get_planet_positions_hourly(start, end, planet, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_planet_pos(t, planet, coord_sys)
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


def get_planet_positions_minute(start, end, planet, coord_sys, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_planet_pos(t, planet, coord_sys)
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