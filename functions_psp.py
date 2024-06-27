import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy import pycdf
import cdflib
# import spiceypy
# import os
import glob
import urllib.request
import os.path
import pickle


"""
PARKER SOLAR PROBE SERVER DATA PATH
"""

psp_path='/Volumes/External/data/psp/'
kernels_path='/Volumes/External/data/kernels/'


"""
PSP BAD DATA FILTER
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
PSP MAG DATA
# 1 min and full resolution files from https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2
# 1min files naming convention: psp_fld_l2_mag_rtn_1min_{date_str}_v02.cdf
# full res files are split into daily files, 6 hourly intervals, so naming convention: psp_fld_l2_mag_rtn_{date_str}{time}_v02.cdf
"""


#DOWNLOAD FUNCTIONS for 1min or full res data


def download_pspmag_1min(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/1min'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_fld_l2_mag_rtn_1min_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def download_pspmag_full(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/full'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            data_item_id = f'psp_fld_l2_mag_rtn_{date_str}{t:02}_v02'
            if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
                print(f'{data_item_id}.cdf has already been downloaded.')
            else:
                try:
                    data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/{year}/{data_item_id}.cdf'
                    urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                    print(f'Successfully downloaded {data_item_id}.cdf')
                except Exception as e:
                    print('ERROR', e, data_item_id)
        start += timedelta(days=1)        
        

#LOAD FUNCTIONS for MAG data 


def get_pspmag_1min(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN_1min'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN_1min'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspmag_full(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['epoch_mag_RTN'], ['time'])}
        df = pd.DataFrame.from_dict(data)
        bx, by, bz = cdf['psp_fld_l2_mag_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


#Load range of files using specified start and end dates/ timestamps


def get_pspmag_range_1min(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/1min'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_fld_l2_mag_rtn_1min_{date_str}_v02.cdf'
        _df = get_pspmag_1min(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


def get_pspmag_range_full(start_timestamp, end_timestamp, path=f'{psp_path}'+'mag/l2/full'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        for t in [0, 6, 12, 18]:
            fn = f'{path}/psp_fld_l2_mag_rtn_{date_str}{t:02}_v02.cdf'
            _df = get_pspmag_full(fn)
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
PSP PLAS DATA
"""


def download_pspplas(start_timestamp, end_timestamp, path=f'{psp_path}'+'sweap/spc/l3i'):
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        data_item_id = f'psp_swp_spc_l3i_{date_str}_v02'
        if os.path.isfile(f"{path}/{data_item_id}.cdf") == True:
            print(f'{data_item_id}.cdf has already been downloaded.')
            start += timedelta(days=1)
        else:
            try:
                data_url = f'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/{year}/{data_item_id}.cdf'
                urllib.request.urlretrieve(data_url, f"{path}/{data_item_id}.cdf")
                print(f'Successfully downloaded {data_item_id}.cdf')
                start += timedelta(days=1)
            except Exception as e:
                print('ERROR', e, data_item_id)
                start += timedelta(days=1)


def get_pspspc_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_moment', 'wp_moment'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_moment_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np_fit', 'wp_fit'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp_fit_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_fit1(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'np1_fit', 'wp1_fit'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['vp1_fit_RTN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspi_mom(fp):
    """raw = rtn"""
    try:
        cdf = pycdf.CDF(fp)
        data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'DENS', 'TEMP'], ['timestamp', 'density', 'temperature'])}
        df = pd.DataFrame.from_dict(data)
        vx, vy, vz = cdf['VEL_RTN_SUN'][:].T
        df['v_x'] = vx
        df['v_y'] = vy
        df['v_z'] = vz
        df['v_bulk'] = np.linalg.norm(df[['v_x', 'v_y', 'v_z']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_pspspc_range_mom(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spc/l3i"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'temperature', -1E30)
    filter_bad_data(df, 'density', -1E30)
    filter_bad_data(df, 'v_bulk', -1E30)
    filter_bad_data(df, 'v_x', -1E30)
    filter_bad_data(df, 'v_y', -1E30)
    filter_bad_data(df, 'v_z', -1E30)
    return df


def get_pspspc_range_fit(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spc/l3i"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spc_l3i_{date_str}_v02.cdf'
        _df = get_pspspc_fit(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    filter_bad_data(df, 'temperature', -1E30)
    filter_bad_data(df, 'density', -1E30)
    filter_bad_data(df, 'v_bulk', -1E30)
    filter_bad_data(df, 'v_x', -1E30)
    filter_bad_data(df, 'v_y', -1E30)
    filter_bad_data(df, 'v_z', -1E30)
    return df


def get_pspspi_range_mom(start_timestamp, end_timestamp, path="/Volumes/External/Data/PSP/sweap/spi/sf00/mom"):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}/psp_swp_spi_sf00_l3_mom_{date_str}_v04.cdf'
        _df = get_pspspi_mom(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df


def create_psp_pkl(start_timestamp):

    # #download solo mag and plasma data up to now 
    download_pspmag_1min(start_timestamp)
    download_pspplas(start_timestamp)

    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_pspmag_range_1min(start_timestamp)
    if df_mag is None:
        print(f'PSP FIELDS data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)

    #load in plasma data to DataFrame and resample, create empty plasma and resampled DataFrame if no data
    #only drop time column if MAG DataFrame is not empty
    df_plas = get_pspspi_range_mom(start_timestamp)
    if df_plas is None:
        print(f'PSP SPI/MOM data is empty for this timerange')
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
    psp_pos = get_psp_positions(magplas_rdf['time'])
    psp_pos.set_index(pd.to_datetime(psp_pos['time']), inplace=True)
    psp_pos = psp_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([magplas_rdf, psp_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    psp=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('vx', float),('vy', float),('vz', float),('vt', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    psp = psp.view(np.recarray) 

    psp.time=dt_lst
    psp.bx=comb_df['bx']
    psp.by=comb_df['by']
    psp.bz=comb_df['bz']
    psp.bt=comb_df['bt']
    psp.vx=comb_df['vx']
    psp.vy=comb_df['vy']
    psp.vz=comb_df['vz']
    psp.vt=comb_df['vt']
    psp.np=comb_df['np']
    psp.tp=comb_df['tp']
    psp.x=comb_df['x']
    psp.y=comb_df['y']
    psp.z=comb_df['z']
    psp.r=comb_df['r']
    psp.lat=comb_df['lat']
    psp.lon=comb_df['lon']
    
    #dump to pickle file
    
    header='Science level 2 solar wind magnetic field (FIELDS) and plasma data (SWEAP/SPI/MOM) from Parker Solar Probe, ' + \
    'obtained from https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min and https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/  '+ \
    'Timerange: '+psp.time[0].strftime("%Y-%b-%d %H:%M")+' to '+psp.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by psp.time, psp.bx, psp.vt etc. '+\
    'Total number of data points: '+str(psp.size)+'. '+\
    'Units are btxyz [nT, RTN], vtxy  [km s^-1], np[cm^-3], tp [K], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with [...] by E. Davies (twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([psp,header], open(psp_path+'psp_rtn.p', "wb"))