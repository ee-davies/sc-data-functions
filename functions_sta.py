import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spiceypy
import os.path
import glob
import urllib.request
from urllib.request import urlopen
import json
import astrospice
from sunpy.coordinates import HeliocentricInertial, HeliographicStonyhurst
from bs4 import BeautifulSoup
import cdflib
import pickle
from spacepy import pycdf


"""
STEREO-A DATA PATH
"""


stereoa_path='/Volumes/External/data/stereoa/'
kernels_path='/Volumes/External/data/kernels/'


"""
STEREO-A BAD DATA FILTER
"""


def filter_bad_df(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df


def filter_bad_col(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    df[col][mask] = np.nan
    return df


"""
STEREO-A DOWNLOAD DATA
#can download yearly merged IMPACT files or beacon data
#beacon data files downloaded may be corrupt
"""


## function to download stereo merged impact data from nasa spdf service
## files are yearly
def download_stereoa_merged(start_timestamp, end_timestamp=datetime.utcnow(), path=stereoa_path+'impact/merged/level2/'):
    start = start_timestamp.year
    end = end_timestamp.year + 1
    while start < end:
        year = start
        date_str = f'{year}0101'
        try: 
            data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/l2/impact/magplasma/1min/{year}/'
            soup = BeautifulSoup(urlopen(data_url), 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None and href.startswith('sta_l2_magplasma_1m_'+date_str):
                    filename = href
                    if os.path.isfile(f"{path}{filename}") == True:
                        print(f'{filename} has already been downloaded.')
                    else:
                        urllib.request.urlretrieve(data_url+filename, f"{path}{filename}")
                        print(f'Successfully downloaded {filename}')
        except Exception as e:
            print('ERROR', e, f'.File for {year} does not exist.')
        start+=1


#download data from https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon/
#download functions work but seem to download corrupt files: may need to manually download cdfs instead
def download_sta_beacon_mag(path=stereoa_path+'beacon/mag'):
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'sta_lb_impact_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.') 
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon/{year}'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
            except Exception as e:
                print('ERROR', e, data_item_id)
        start += timedelta(days=1)


#download data from https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon_plastic/
#download functions work but seem to download corrupt files: may need to manually download cdfs instead
def download_sta_beacon_plas(path=stereoa_path+'beacon/plas'):
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'sta_lb_pla_browse_{date_str}_v14'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/beacon_plastic/{year}'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


"""
STEREO-A MAG AND PLAS DATA 
# Option to load in merged mag and plas data files
# Can also load separate MAG and PLAS beacon data files for real-time use
"""


#function to read in yearly cdf file 
#also filters bad data values
#creates pandas df 
def get_stereoa_merged(fp):
    """raw = rtn"""
    try:
        cdf = cdflib.CDF(fp)
        t1 = cdflib.cdfepoch.to_datetime(cdf.varget('Epoch'))
        df = pd.DataFrame(t1, columns=['time'])
        bx, by, bz = cdf['BFIELDRTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = cdf['BTOTAL']
        df['np'] = cdf['Np']
        df['tp'] = cdf['Tp']
        df['vt'] = cdf['Vp']
        cols = ['bx', 'by', 'bz', 'bt', 'np', 'tp', 'vt']
        for col in cols:
            df[col].mask(df[col] < -9.999E29 , pd.NA, inplace=True)
        df['vx'] = cdf['Vr_Over_V_RTN']*df['vt']
        df['vy'] = cdf['Vt_Over_V_RTN']*df['vt']
        df['vz'] = cdf['Vn_Over_V_RTN']*df['vt']
        v_cols = ['vx', 'vy', 'vz']
        for v_col in v_cols:
            df[v_col].mask(df[v_col] < -9.999E29 , pd.NA, inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


# uses get_stereoa_merged function to load multiple years of data 
# end timestamp can be modified, but default is set as now 
def get_stereoa_merged_range(start_timestamp, end_timestamp=datetime.utcnow(), path=stereoa_path+'impact/merged/level2/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df=None
    start = start_timestamp.year
    end = datetime.utcnow().year + 1
    while start < end:
        year = start
        date_str = f'{year}0101'
        try: 
            fn = glob.glob(path+f'sta_l2_magplasma_1m_{date_str}*')[0]
            _df = get_stereoa_merged(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        except Exception as e:
            print('ERROR:', e, f'{date_str} does not exist')
        start += 1
    timemask = (df['time']>=start_timestamp) & (df['time']<=end_timestamp)
    df = df[timemask]
    return df


def get_sta_beacon_plas(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch1', 'Bulk_Speed', 'Vr_RTN', 'Vt_RTN', 'Vn_RTN', 'Density', 'Temperature_Inst'], ['time', 'vt', 'vx', 'vy', 'vz', 'np', 'tp'])}
        df = pd.DataFrame.from_dict(data)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_sta_beacon_mag(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch_MAG'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['MAGBField'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_sta_beacon_mag_7days(path=f'{stereoa_path}'+'beacon/mag/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_impact_{date_str}_v02.cdf'
        _df = get_sta_beacon_mag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_sta_beacon_mag_range(start_timestamp, end_timestamp, path=f'{stereoa_path}'+'beacon/mag/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_impact_{date_str}_v02.cdf'
        _df = get_sta_beacon_mag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_sta_beacon_plas_7days(path=f'{stereoa_path}'+'beacon/plas/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = datetime.utcnow().date()-timedelta(days=7)
    end = datetime.utcnow().date()
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_pla_browse_{date_str}_v14.cdf'
        _df = get_sta_beacon_plas(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = filter_bad_col(df, 'tp', -1E30) #will give slice warnings
    df = filter_bad_col(df, 'np', -1E30)
    df = filter_bad_col(df, 'vt', -1E30)
    df = filter_bad_col(df, 'vx', -1E30)
    df = filter_bad_col(df, 'vy', -1E30)
    df = filter_bad_col(df, 'vz', -1E30)
    return df


def get_sta_beacon_plas_range(start_timestamp, end_timestamp, path=f'{stereoa_path}'+'beacon/plas/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'{path}/sta_lb_pla_browse_{date_str}_v14.cdf'
        _df = get_sta_beacon_plas(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = filter_bad_col(df, 'tp', -1E30) #will give slice warnings
    df = filter_bad_col(df, 'np', -1E30)
    df = filter_bad_col(df, 'vt', -1E30)
    df = filter_bad_col(df, 'vx', -1E30)
    df = filter_bad_col(df, 'vy', -1E30)
    df = filter_bad_col(df, 'vz', -1E30)
    return df


"""
STEREO A POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


#kernels from https://soho.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/depm/ahead/ 
#and https://soho.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/epm/ahead/ for predicted orbit kernel
def stereoa_furnish():
    """Main"""
    stereoa_path = kernels_path+'stereoa/'
    generic_path = kernels_path+'generic/'
    stereoa_kernels = os.listdir(stereoa_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in stereoa_kernels:
        spiceypy.furnsh(os.path.join(stereoa_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_sta_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        stereoa_furnish()
    pos = spiceypy.spkpos("STEREO AHEAD", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_sta_positions(time_series):
    positions = []
    for t in time_series:
        position = get_sta_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_sta_pos_range(start, end, cadence=1):
    """Cadence in minutes"""
    t = start
    positions = []
    while t < end:
        position = get_sta_pos(t)
        positions.append(position)  
        t += timedelta(minutes=cadence)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def main():
    tfmt = r'%Y%m%dT%H%M%S'
    start = datetime.strptime(input('->'), tfmt)
    end = datetime.strptime(input('->'), tfmt)
    x = get_sta_pos(start, end)
    print(x)

if __name__ == '__main__':
    main()


"""
OUTPUT COMBINED PICKLE FILES
including MAG, PLAS, and POSITION data
"""

def create_sta_beacon_pkl(start_timestamp, end_timestamp, output_path='/Users/emmadavies/Documents/Projects/SolO_Realtime_Preparation/March2024/'):

    # start_timestamp=datetime.utcnow()-timedelta(days=7)
    # end_timestamp=datetime.utcnow()

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_sta_beacon_mag_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'STA Beacon MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)

    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_sta_beacon_plas_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'STA Beacon PLAS data is empty for this timerange')
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

    #get sta positions for corresponding timestamps
    stereoa_furnish()
    sta_pos = get_sta_positions(magplas_rdf['time'])
    sta_pos.set_index(pd.to_datetime(sta_pos['time']), inplace=True)
    sta_pos = sta_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, sta_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    stereoa=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    stereoa = stereoa.view(np.recarray) 

    stereoa.time=dt_lst
    stereoa.bx=comb_df['bx']
    stereoa.by=comb_df['by']
    stereoa.bz=comb_df['bz']
    stereoa.bt=comb_df['bt']
    stereoa.vx=comb_df['vx']
    stereoa.vy=comb_df['vy']
    stereoa.vz=comb_df['vz']
    stereoa.vt=comb_df['vt']
    stereoa.np=comb_df['np']
    stereoa.tp=comb_df['tp']
    stereoa.x=comb_df['x']
    stereoa.y=comb_df['y']
    stereoa.z=comb_df['z']
    stereoa.r=comb_df['r']
    stereoa.lat=comb_df['lat']
    stereoa.lon=comb_df['lon']

    #dump to pickle file
    header='Beacon solar wind magnetic field (MAG) and plasma (PLAS) data from IMPACT onboard STEREO-A, ' + \
    'Timerange: '+stereoa.time[0].strftime("%Y-%b-%d %H:%M")+' to '+stereoa.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by stereoa.time, stereoa.bx, stereoa.r etc. '+\
    'Total number of data points: '+str(stereoa.size)+'. '+\
    'Units are btxyz [nT, RTN], vtxy  [km s^-1], np[cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    t_now_date_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
    pickle.dump([stereoa,header], open(output_path+f'stereoa_beacon_rtn_{t_now_date_hour}.p', "wb"))


def create_sta_pkl(start_timestamp, end_timestamp):

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_ = get_stereoa_merged_range(start_timestamp, end_timestamp)
    if df_ is None:
        print(f'STA merged data is empty for this timerange')
        df_ = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        rdf = df_.drop(columns=['time'])
    else:
        rdf = df_.set_index('time').resample('1min').mean().reset_index(drop=False)
        rdf.set_index(pd.to_datetime(rdf['time']), inplace=True)

    #get sta positions for corresponding timestamps
    stereoa_furnish()
    sta_pos = get_sta_positions(rdf['time'])
    sta_pos.set_index(pd.to_datetime(sta_pos['time']), inplace=True)
    sta_pos = sta_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([rdf, sta_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    stereoa=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    stereoa = stereoa.view(np.recarray) 

    stereoa.time=dt_lst
    stereoa.bx=comb_df['bx']
    stereoa.by=comb_df['by']
    stereoa.bz=comb_df['bz']
    stereoa.bt=comb_df['bt']
    stereoa.vx=comb_df['vx']
    stereoa.vy=comb_df['vy']
    stereoa.vz=comb_df['vz']
    stereoa.vt=comb_df['vt']
    stereoa.np=comb_df['np']
    stereoa.tp=comb_df['tp']
    stereoa.x=comb_df['x']
    stereoa.y=comb_df['y']
    stereoa.z=comb_df['z']
    stereoa.r=comb_df['r']
    stereoa.lat=comb_df['lat']
    stereoa.lon=comb_df['lon']

    #dump to pickle file
    header='Level 2 science solar wind magnetic field (MAG) and plasma (PLAS) data from IMPACT onboard STEREO-A, ' + \
    'Timerange: '+stereoa.time[0].strftime("%Y-%b-%d %H:%M")+' to '+stereoa.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by stereoa.time, stereoa.bx, stereoa.r etc. '+\
    'Total number of data points: '+str(stereoa.size)+'. '+\
    'Units are btxyz [nT, RTN], vtxy  [km s^-1], np[cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([stereoa,header], open(stereoa_path+'stereoa_rtn.p', "wb"))
