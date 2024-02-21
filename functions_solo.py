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

solo_path='/Volumes/External/data/solo/'
kernels_path='/Volumes/External/data/kernels/'


"""
SOLO BAD DATA FILTER
"""


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
SOLO MAG DATA
# Potentially different SolO MAG file names: internal low latency, formagonly, and formagonly 1 min.
e.g.
- Internal files: solo_L2_mag-rtn-ll-internal_20230225_V00.cdf
- For MAG only files: solo_L2_mag-rtn-normal-formagonly_20200415_V01.cdf
- For MAG only 1 minute res files: solo_L2_mag-rtn-normal-1-minute-formagonly_20200419_V01.cdf
# All in RTN coords
# Should all follow same variable names within cdf
"""


#DOWNLOAD FUNCTIONS for 1min or 1sec data


def download_solomag_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1min'): 
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_mag-rtn-normal-1-minute_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_solomag_1sec(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1sec'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_mag-rtn-normal_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#LOAD FUNCTIONS for MAG data 


#Load single file from specific path using pycdf from spacepy
def get_solomag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['EPOCH'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['B_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df

# #Load single file from specific path using cdflib
# def get_solomag(fp):
#     """raw = rtn"""
#     try:
#         cdf = cdflib.CDF(fp)
#         t1 = cdflib.cdfepoch.to_datetime(cdf.varget('EPOCH'))
#         df = pd.DataFrame(t1, columns=['time'])
#         bx, by, bz = cdf['B_RTN'][:].T
#         df['bx'] = bx
#         df['by'] = by
#         df['bz'] = bz
#         df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
#         cols = ['bx', 'by', 'bz', 'bt']
#         for col in cols:
#             df[col].mask(df[col] < -9.999E29 , pd.NA, inplace=True)
#     except Exception as e:
#         print('ERROR:', e, fp)
#         df = None
#     return df


#Load range of files using specified start and end dates/ timestamps
def get_solomag_range_formagonly_internal(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/ll'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-ll-internal_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_solomag_range_formagonly(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/formagonly/full'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal-formagonly_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_solomag_range_formagonly_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/formagonly/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal-1-minute-formagonly_{date_str}_*.cdf')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_solomag_range_1sec(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1sec'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_mag-rtn-normal_{date_str}*')
        _df = get_solomag(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_solomag_range_1min(start_timestamp, end_timestamp, path=f'{solo_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/solo_L2_mag-rtn-normal-1-minute_{date_str}.cdf'
        _df = get_solomag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


#combined solomag range function to specify level and resolution of data 
def get_solomag_range(start_timestamp, end_timestamp, level="l2", res="1min"):
    if level == "l2":
        if res == "1min":
            df = get_solomag_range_1min(start_timestamp, end_timestamp)
        elif res == "1sec":
            df = get_solomag_range_1sec(start_timestamp, end_timestamp)
    elif level == "ll":
        df = get_solomag_range_formagonly_internal(start_timestamp, end_timestamp)
    elif level == "formagonly":
        if res == "full":
            df = get_solomag_range_formagonly(start_timestamp, end_timestamp)
        elif res == "1min":
            df = get_solomag_range_formagonly_1min(start_timestamp, end_timestamp)
    return df 


"""
SOLO PLASMA DATA
# Level 2 science SWA PLAS grnd moment data
"""


#DOWNLOAD FUNCTION for swa/plas data
def download_soloplas(start_timestamp, end_timestamp, path=f'{solo_path}'+'swa/plas/l2'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        data_item_id = f'solo_L2_swa-pas-grnd-mom_{date_str}'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


#Load single file from specific path using pycdf from spacepy
def get_soloplas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'N', 'T'], ['time', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'])
        vx, vy, vz = cdf['V_RTN'][:].T
        df['vx'] = vx
        df['vy'] = vy
        df['vz'] = vz
        df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


# #Load single file from specific path using cdflib
# def get_soloplas(fp):
#     """raw = rtn"""
#     try:
#         cdf = cdflib.CDF(fp)
#         t1 = cdflib.cdfepoch.to_datetime(cdf.varget('EPOCH'))
#         df = pd.DataFrame(t1, columns=['time'])
#         df['np'] = cdf['N']
#         df['tp'] = cdf['T']
#         vx, vy, vz = cdf['V_RTN'][:].T
#         df['vx'] = vx
#         df['vy'] = vy
#         df['vz'] = vz
#         df['vt'] = np.linalg.norm(df[['vx', 'vy', 'vz']], axis=1)
#         cols = ['np', 'tp', 'vx', 'vy', 'vz', 'vt']
#         for col in cols:
#             df[col].mask(df[col] < -9.999E29 , pd.NA, inplace=True)
#     except Exception as e:
#         print('ERROR:', e, fp)
#         df = None
#     return df


#Load range of files using specified start and end dates/ timestamps
def get_soloplas_range(start_timestamp, end_timestamp, path=f'{solo_path}'+'swa/plas/l2'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = glob.glob(f'{path}/solo_L2_swa-pas-grnd-mom_{date_str}*')
        _df = get_soloplas(fn[0])
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
SOLO POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


#http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/fk/ for solo_ANC_soc-sci-fk_V08.tf
#http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/ for solo orbit .bsp


def solo_furnish():
    """Main"""
    solo_path = kernels_path+'solo/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(solo_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(solo_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_solo_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        solo_furnish()
    pos = spiceypy.spkpos("SOLAR ORBITER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ; can be changed
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_solo_positions(time_series):
    positions = []
    for t in time_series:
        position = get_solo_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


"""
OUTPUT COMBINED PICKLE FILE
including MAG, PLAS, and POSITION data
"""


def create_solo_pkl(start_timestamp, end_timestamp, level='l2', res='1min'):
    
    # #download solo mag and plasma data up to now 
    # download_solomag_1min(start_timestamp)
    # download_soloplas(start_timestamp)

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_solomag_range(start_timestamp, end_timestamp, level, res)
    if df_mag is None:
        print(f'SolO MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        
    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_soloplas_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'SolO SWA data is empty for this timerange')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
        plas_rdf = df_plas
    else:
        plas_rdf = df_plas.set_index('time').resample('1min').mean().reset_index(drop=False)
        plas_rdf.set_index(pd.to_datetime(plas_rdf['time']), inplace=True)
        if mag_rdf.shape[0] != 0:
            plas_rdf = plas_rdf.drop(columns=['time'])

    #need to combine mag and plasma dfs to get complete set of timestamps for position calculation
    magplas_rdf = pd.concat([mag_rdf, plas_rdf], axis=1)
    #some timestamps may be NaT so after joining, drop time column and reinstate from combined index col
    magplas_rdf = magplas_rdf.drop(columns=['time'])
    magplas_rdf['time'] = magplas_rdf.index
     
    #get solo positions for corresponding timestamps
    solo_pos = get_solo_positions(magplas_rdf['time'])
    solo_pos.set_index(pd.to_datetime(solo_pos['time']), inplace=True)
    solo_pos = solo_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, solo_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    solo=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    solo = solo.view(np.recarray) 

    solo.time=dt_lst
    solo.bx=comb_df['bx']
    solo.by=comb_df['by']
    solo.bz=comb_df['bz']
    solo.bt=comb_df['bt']
    solo.vx=comb_df['vx']
    solo.vy=comb_df['vy']
    solo.vz=comb_df['vz']
    solo.vt=comb_df['vt']
    solo.np=comb_df['np']
    solo.tp=comb_df['tp']
    solo.x=comb_df['x']
    solo.y=comb_df['y']
    solo.z=comb_df['z']
    solo.r=comb_df['r']
    solo.lat=comb_df['lat']
    solo.lon=comb_df['lon']
    
    #dump to pickle file
    header='Internal low latency solar wind magnetic field (MAG) from Solar Orbiter, ' + \
    'provided directly by Imperial College London SolO MAG Team '+ \
    'Timerange: '+solo.time[0].strftime("%Y-%b-%d %H:%M")+' to '+solo.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by solo.time, solo.bx, solo.r etc. '+\
    'Total number of data points: '+str(solo.size)+'. '+\
    'Units are btxyz [nT, RTN], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([solo,header], open(solo_path+'solo_rtn.p', "wb"))