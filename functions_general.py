import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


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