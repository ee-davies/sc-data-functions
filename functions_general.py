import numpy as np
import pandas as pd
from scipy import constants

from pathlib import Path
import json


def resample_df(df, resample_min):
    rdf = df.set_index('time').resample(f'{resample_min}min').mean().reset_index(drop=False)
    return rdf


def merge_rdfs(df1, df2):
    df1.set_index(pd.to_datetime(df1['time']), inplace=True)
    df2.set_index(pd.to_datetime(df2['time']), inplace=True)
    mdf = pd.concat([df1, df2], axis=1)
    mdf = mdf.drop(['time'], axis=1)
    mdf = mdf.reset_index(drop=False)
    return mdf


def calc_pressure_params(plasmag_df):
# assuming Tpr is the (isotropic) temperature
# in reality is temperature in radial direction: https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H0_SWE
    plasmag_df['p_plas'] = (plasmag_df['density']*10**6)*constants.k*plasmag_df['temperature']
    plasmag_df['p_mag'] = 0.5*(plasmag_df['b_tot']*10**(-9))**2./constants.mu_0
    plasmag_df['beta'] = plasmag_df['p_plas']/plasmag_df['p_mag']
    plasmag_df['p_tot'] = plasmag_df['p_plas'] + plasmag_df['p_mag']
    return plasmag_df



def make_mag_recarray(df):
    #create rec array
    time_stamps = df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format
    rarr=np.zeros(len(dt_lst),dtype=[('time',object),('bt', float),('bx', float),('by', float),('bz', float)])
    rarr = rarr.view(np.recarray)
    rarr.time=dt_lst
    rarr.bt=df['bt']
    rarr.bx=df['bx']
    rarr.by=df['by']
    rarr.bz=df['bz']
    return rarr


def make_plas_recarray(df):
    #create rec array
    time_stamps = df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format
    rarr=np.zeros(len(dt_lst),dtype=[('time',object),('vt', float),('vx', float),('vy', float),('vz', float),\
                                     ('np', float),('tp', float)])
    rarr = rarr.view(np.recarray)
    rarr.time=dt_lst
    rarr.vt=df['vt']
    rarr.vx=df['vx']
    rarr.vy=df['vy']
    rarr.vz=df['vz']
    rarr.np=df['np']
    rarr.tp=df['tp']
    return rarr


def make_pos_recarray(df):
    #create rec array
    time_stamps = df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format
    rarr=np.zeros(len(dt_lst),dtype=[('time',object),('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    rarr = rarr.view(np.recarray)
    rarr.time=dt_lst
    rarr.x=df['x']
    rarr.y=df['y']
    rarr.z=df['z']
    rarr.r=df['r']
    rarr.lat=df['lat']
    rarr.lon=df['lon']
    return rarr


def make_combined_recarray(df):
    #create rec array
    time_stamps = df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format
    rarr=np.zeros(len(dt_lst),dtype=[('time',object),('bt', float),('bx', float),('by', float),('bz', float),\
                ('vt', float),('vx', float),('vy', float),('vz', float),('np', float),('tp', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    rarr = rarr.view(np.recarray)
    rarr.time=dt_lst
    rarr.bt=df['bt']
    rarr.bx=df['bx']
    rarr.by=df['by']
    rarr.bz=df['bz']
    rarr.vt=df['vt']
    rarr.vx=df['vx']
    rarr.vy=df['vy']
    rarr.vz=df['vz']
    rarr.np=df['np']
    rarr.tp=df['tp']
    rarr.x=df['x']
    rarr.y=df['y']
    rarr.z=df['z']
    rarr.r=df['r']
    rarr.lat=df['lat']
    rarr.lon=df['lon']
    return rarr


# === Load path from JSON config ===
def load_path(config_file=Path(__file__).resolve().parents[0] /'config.json', path_name='kernels_path'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[path_name]


def normalise_time(df, min_time, max_time):
    df['epoch'] = df['time'].apply(lambda x: float((x-min_time).total_seconds()))
    ## Normalise by max boundary time 
    max_epoch_time = (max_time-min_time).total_seconds()
    df['time_norm'] = df['epoch']/max_epoch_time
    return df