import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spiceypy
import os.path
import pickle
import scipy

from .functions_general import load_path



def format_path(fp):
    """Formatting required for CDF package."""
    return fp.replace('/', '\\')


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'Timestamp']
    df.loc[mask, cols] = np.nan
    return df


"""
JUNO DATA PATH
"""

juno_path=load_path(path_name='juno_path')
print(f"Juno path loaded: {juno_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
JUNO MAG DATA
# Cruise phase FGM data, https://pds-ppi.igpp.ucla.edu/search/view/?f=yes&id=pds://PPI/JNO-SS-3-FGM-CAL-V1.0
# 1 min resolution, SE coordinates (equivalent to RTN)
# .sts files, not .cdf
"""


def get_junomag(fp):
    """Get data and return pd.DataFrame."""
    cols = ['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond',
            'Decimal Day', 'bx', 'by', 'bz', 'Range', 'POS_X', 'POS_Y', 'POS_Z']
    try:
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if sum(c.isalpha() for c in line) == 0:
                    break

        df = pd.read_csv(fp, skiprows=i, sep=r'\s+', names=cols)
        df['time'] = df[['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond']]\
            .apply(lambda x: datetime.strptime(' '.join(str(y) for y in x),
                                               r'%Y %j %H %M %S %f'), axis=1)
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
        df.drop(columns = ['Year', 'DoY', 'Hour', 'Minute', 'Second', 'Millisecond', 'Decimal Day', 'Range', 'POS_X', 'POS_Y', 'POS_Z'], inplace=True)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_junomag_range(start_timestamp, end_timestamp, path=juno_path+'fgm/1min'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start < end:
        year = start.year
        doy = start.strftime('%j')
        fn = f'fgm_jno_l3_{year}{doy}se_r60s_v01.sts'
        _df = get_junomag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
JUNO PLASMA DATA
Juno plasma data provided by R. Wilson privately in .mat file
"""

def matlab2datetime(matlab_datenum):
    day = datetime.fromordinal(int(matlab_datenum))
    dayfrac = timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    return day + dayfrac


def get_junoplas(fp=f"{juno_path}"+"jade/JADE_Wilson.mat"):
    mat_jade = scipy.io.loadmat(fp)
    time = mat_jade['time'].reshape(-1)
    vt = mat_jade['vsw'].reshape(-1)
    np = mat_jade['np'].reshape(-1)
    tp = mat_jade['Tp'].reshape(-1)
    converted_time = [matlab2datetime(tval) for tval in time]
    data = {'time':converted_time, 'vt':vt, 'np':np, 'tp':tp}
    df = pd.DataFrame.from_dict(data)
    return df


def get_junoplas_range(start_timestamp, end_timestamp):
    df = get_junoplas()
    mask = (df['time'] >= start_timestamp) & (df['time'] <= end_timestamp)
    df_new = df[mask].reset_index(drop=True)
    return df_new


"""
JUNO POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
Currently set to HEEQ, but will implement options to change
"""


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def juno_furnish():
    """Main"""
    juno_path = kernels_path+'juno/'
    generic_path = kernels_path+'generic/'
    juno_kernels = os.listdir(juno_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in juno_kernels:
        spiceypy.furnsh(os.path.join(juno_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))
    

def get_juno_pos(t, frame="HEEQ"):
    if spiceypy.ktotal('ALL') < 1:
        juno_furnish()
    if frame == "HEEQ":
        try:
            pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0] #calls positions in HEEQ
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]
    elif frame == "HAE":
        try:
            pos = spiceypy.spkpos("JUNO", spiceypy.datetime2et(t), "ECLIPJ2000", "NONE", "SUN")[0] #calls positions in HAE or ECLIPJ2000
            r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
            position = t, pos[0], pos[1], pos[2], r, lat, lon
            return position
        except Exception as e:
            print(e)
            return [t, None, None, None, None, None, None]


def get_juno_positions(time_series, frame="HEEQ"):
    positions = []
    for t in time_series:
        position = get_juno_pos(t, frame)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def get_juno_positions_daily(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_juno_pos(t)
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


def get_juno_positions_hourly(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_juno_pos(t)
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


def get_juno_positions_minute(start, end, cadence, dist_unit='au', ang_unit='deg'):
    t = start
    positions = []
    while t < end:
        position = get_juno_pos(t)
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


def get_juno_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    if spiceypy.ktotal('ALL') < 1:
        juno_furnish()
    transform = spiceypy.pxform(base_frame, to_frame, spiceypy.datetime2et(epoch))
    return transform


def transform_data(df, to_frame):
    pass


"""
OUTPUT COMBINED PICKLE FILE
including MAG, empty PLAS, and POSITION data
"""


def create_juno_pkl(start_timestamp, end_timestamp, data_coord_sys = "RTN", pos_coord_sys = "HAE"):

    if data_coord_sys != "RTN":
        print("Juno data only available in RTN at the moment. Continuing to produce RTN data file...")

    #create mag df, resampled to nearest 1 min
    df_mag = get_junomag_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'Juno MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        mag_rdf = mag_rdf.dropna() #need to drop NaN values for plotly...

    #create plas df, resampled to nearest 1 min
    df_plas = get_junoplas_range(start_timestamp, end_timestamp)
    if df_plas is None:
        print(f'Note: Juno plasma data is unavailable during cruise phase, except 2016-05-15 to 2016-06-24')
        df_plas = pd.DataFrame({'time':[], 'vt':[], 'vx':[], 'vy':[], 'vz':[], 'np':[], 'tp':[]})
        plas_rdf = df_plas.drop(columns=['time'])
    else:
        plas_rdf = df_plas.set_index('time').resample('1min').mean().reset_index(drop=False)
        plas_rdf.set_index(pd.to_datetime(plas_rdf['time']), inplace=True)
        plas_rdf = plas_rdf.dropna() #need to drop NaN values for plotly...
        plas_rdf['vx'] = pd.NA
        plas_rdf['vy'] = pd.NA
        plas_rdf['vz'] = pd.NA

    #need to combine mag and plasma dfs to get complete set of timestamps for position calculation
    magplas_rdf = pd.concat([mag_rdf, plas_rdf], axis=1)
    #some timestamps may be NaT so after joining, drop time column and reinstate from combined index col
    magplas_rdf = magplas_rdf.drop(columns=['time'])
    magplas_rdf['time'] = magplas_rdf.index

    #get juno positions for corresponding timestamps
    juno_furnish()
    juno_pos = get_juno_positions(magplas_rdf['time'], frame=pos_coord_sys)
    juno_pos.set_index(pd.to_datetime(juno_pos['time']), inplace=True)
    juno_pos = juno_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, juno_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    juno=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    juno = juno.view(np.recarray) 

    juno.time=dt_lst
    juno.bx=comb_df['bx']
    juno.by=comb_df['by']
    juno.bz=comb_df['bz']
    juno.bt=comb_df['bt']
    juno.vx=comb_df['vx']
    juno.vy=comb_df['vy']
    juno.vz=comb_df['vz']
    juno.vt=comb_df['vt']
    juno.np=comb_df['np']
    juno.tp=comb_df['tp']
    juno.x=comb_df['x']
    juno.y=comb_df['y']
    juno.z=comb_df['z']
    juno.r=comb_df['r']
    juno.lat=comb_df['lat']
    juno.lon=comb_df['lon']

    #dump to pickle file
    header='Science level 2 solar wind magnetic field (FGM) from Juno Mission Cruise Phase,' + \
    'obtained from https://pds-ppi.igpp.ucla.edu/search/view/?f=yes&id=pds://PPI/JNO-SS-3-FGM-CAL-V1.0/DATA/CRUISE/SE/1MIN '+ \
    'Timerange: '+juno.time[0].strftime("%Y-%b-%d %H:%M")+' to '+juno.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by juno.time, juno.bx, juno.vt, etc. '+\
    'Total number of data points: '+str(juno.size)+'. '+\
    'Units are btxyz [nT, RTN], vtxy  [km s^-1], np[cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree]. '+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    fileid = start_timestamp.strftime("%Y-%m-%d")
    pickle.dump([juno,header], open(juno_path+f'juno_rtn_{fileid}.p', "wb"))


def create_juno_mag_pkl(start_timestamp, end_timestamp):

    #create mag df, resampled to nearest 1 min
    df_mag = get_junomag_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'Juno MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        mag_rdf = mag_rdf.dropna() #need to drop NaN values for plotly...

    #produce recarray with correct datatypes
    time_stamps = mag_rdf['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    juno=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    juno = juno.view(np.recarray) 

    juno.time=dt_lst
    juno.bx=mag_rdf['bx']
    juno.by=mag_rdf['by']
    juno.bz=mag_rdf['bz']
    juno.bt=mag_rdf['bt']

    #dump to pickle file
    header='Science level 2 solar wind magnetic field (FGM) from Juno Mission Cruise Phase, ' + \
    'obtained from https://pds-ppi.igpp.ucla.edu/search/view/?f=yes&id=pds://PPI/JNO-SS-3-FGM-CAL-V1.0/DATA/CRUISE/SE/1MIN '+ \
    'Timerange: '+juno.time[0].strftime("%Y-%b-%d %H:%M")+' to '+juno.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by juno.time, juno.bx, etc. '+\
    'Total number of data points: '+str(juno.size)+'. '+\
    'Units are btxyz [nT, RTN].'+\
    'Made with script by E.E. Davies (github @ee-davies, twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([juno,header], open(juno_path+'juno_20160520_rtn.p', "wb"))