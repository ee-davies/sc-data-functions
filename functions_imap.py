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

