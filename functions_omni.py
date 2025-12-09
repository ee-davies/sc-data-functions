import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import glob


""" DISCLAIMER: THESE FUNCTIONS NEED UPDATING, ARE NOT USEABLE IN CURRENT STATE"""


def get_omni(fp):
    try:
        cdf = pycdf.CDF(fp)
        old_cols = ['Epoch', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'proton_density', 'T', 'Beta']
        new_cols = ['Timestamp', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'Vx', 'Vy', 'Vz', 'density', 'temperature', 'beta']
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(old_cols, new_cols)}
        df = pd.DataFrame.from_dict(data)
        # filter out bad data
        df['BX_GSE'].mask((df['BX_GSE'] > 9999), inplace=True)
        df['BY_GSE'].mask((df['BY_GSE'] > 9999), inplace=True)
        df['BZ_GSE'].mask((df['BZ_GSE'] > 9999), inplace=True)
        df['Vx'].mask((df['Vx'] > 9999), inplace=True)
        df['Vy'].mask((df['Vy'] > 9999), inplace=True)
        df['Vz'].mask((df['Vz'] > 9999), inplace=True)
        df['density'].mask((df['density'] > 9999), inplace=True)
        df['temperature'].mask((df['temperature'] > 9999), inplace=True)
        df['beta'].mask((df['beta'] > 999), inplace=True)
        # convert from GSE to RTN
        df['B_R'] = -1 * df['BX_GSE']
        df['B_T'] = -1 * df['BY_GSE']
        df['B_N'] = df['BZ_GSE']
        df['v_R'] = -1 * df['Vx']
        df['v_T'] = -1 * df['Vy']
        df['v_N'] = df['Vz']     
        df['B_TOT']  = np.sqrt(df['B_R']*df['B_R']+df['B_T']*df['B_T']+df['B_N']*df['B_N'])
        df['v_bulk'] = np.sqrt(df['v_R']*df['v_R']+df['v_T']*df['v_T']+df['v_N']*df['v_N'])
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df

# function that combines time range of monthly files from OMNI
def get_omni_range(start_timestamp, end_timestamp, path='/Volumes/External/Data/OMNI'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start <= end + relativedelta(months=1):
        fn = f'omni_hro2_1min_{start.year}{start.month:02}01_v01.cdf'
        print('loading', fn, start, end)
        _df = get_omni(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += relativedelta(months=1) #+= timedelta(days=1)
    return df