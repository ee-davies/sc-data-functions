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