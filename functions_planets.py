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
PLANETS functions: positions of planets from spice kernels
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