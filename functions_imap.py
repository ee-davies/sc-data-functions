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