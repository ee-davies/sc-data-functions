import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import integrate

"""
FUNCTIONS TO ANALYSE DIFFERENT PARTS OF ICMES
"""

"""
GENERAL
"""

"""
SHOCK
"""

"""
SHEATH
"""

"""
MAGNETIC EJECTA
"""

def get_DiP(df, mo_start, mo_end):
    mask = (df.time >= mo_start) & (df.time <= mo_end)
    ME = df[mask]
    times = [(time - np.min(ME['time'])).total_seconds() for time in ME['time']]
    B_int = integrate.cumtrapz(ME['bt'], times)
    #loop to find halfway point
    i = 0
    while B_int[i] < B_int[-1] / 2:
        i += 1    
    DiP = times[i]/times[-1]
    return DiP

