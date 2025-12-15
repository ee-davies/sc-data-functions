import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import integrate
from scipy.stats import linregress
from scipy import stats

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio



"""
FUNCTIONS TO ANALYSE DIFFERENT PARTS OF ICMES
"""

"""
GENERAL
"""

def plotly_mag(df, save_fig=False):
    pio.renderers.default = 'browser'
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for column, color in zip(['bt', 'bx', 'by', 'bz'], ['black', 'red', 'green', 'blue']):
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[column],
                name=column.upper(),
                line_color=color
            ),
            row=1, col=1
        )
    fig.show()
    if save_fig == True:
        now = datetime.now().strftime('%Y%m%d%H%M')
        fig.write_html(f'plotly_mag_{now}.html') 


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


def get_gexp_power(r1,r2,b1,b2):
    power = (np.log(b2)-np.log(b1))/(np.log(r2)-np.log(r1))
    return power


def slope_fitting(df, start_time, end_time):
    time_mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    slope_df = df[time_mask]
    times = slope_df['time']
    X = (slope_df.loc[:, ('time')]-start_time).dt.total_seconds()
    x = np.array(X).reshape(1, -1)
    y = np.array(slope_df['vt']).reshape(1, -1) #rolling: rolling12_trip_filt2 or raw: trip_filt2_alt_masked
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    line = np.transpose(slope*x+intercept)
    return times, line, slope, intercept, r_value, p_value, std_err


def get_cruise_velocity(df_input):
    df = df_input.reset_index(drop=True)
    try:
        data = df[df['vt'].notnull()]
        min_time = df['time'].min()
        data['epoch'] = df['time'].apply(lambda x: float((x-min_time).total_seconds()))
        fit = linregress(data['epoch'], data['vt'])
        m = fit.slope
        c = fit.intercept
        v_c = m*data['epoch'].max()/2 + c
        v_te = m*data['epoch'].max() + c
        v_le = c
        delta_t = data['epoch'].max()
    except Exception as e:
        print(e)
        m = c = v_c = v_te = v_le = delta_t = pd.NA
    return pd.Series([v_c, m, v_le, v_te, delta_t])


def get_dep(v_c, v_le, v_te, delta_t, r):
    try:
        delta_v = v_le-v_te
        d = r*1.495978707E8
        v_c_squared = v_c**2
        dep = delta_v/delta_t * d/v_c_squared
    except Exception as e:
        print(e)
        dep = None
    return pd.Series([dep])


## OLD FUNCTIONS, NEED UPDATING


def make_icme_v_profile(icme_start, mo_start, mo_end, resampled_df):
    #fit velocity profile within mo 
    mo_fit = slope_fitting(resampled_df, mo_start, mo_end)
    mo_df = pd.DataFrame(mo_fit[0])
    mo_df['v_bulk'] = mo_fit[1]
    #create sheath df
    sheath_mask = (resampled_df['Timestamp']>= icme_start) & (resampled_df['Timestamp']< mo_start)
    sheath_df = resampled_df[sheath_mask]
    if mo_df['v_bulk'][0] > sheath_df['v_bulk'].mean():
        sheath_df = resampled_df[sheath_mask].assign(v_bulk=mo_df['v_bulk'][0])
    else:
        sheath_df = resampled_df[sheath_mask].assign(v_bulk=sheath_df['v_bulk'].mean()) #calculate mean of sheath and replace v_bulk column
    icme_df = pd.concat([sheath_df, mo_df])
    #drop position columns (to not get confused)
    icme_df = icme_df.drop(columns=['X', 'Y', 'Z'])
    return icme_df


def calc_time_delta(resampled_df, icme_df):
    #drop old v_bulk column from resampled df 
    resampled_df = resampled_df.drop(columns=['v_bulk'])
    #merge with new velocity profile, which also produces just values within ICME
    merge=pd.merge(icme_df, resampled_df, how='inner', left_index=True, right_index=True)
    #add columns with new variables needed for time_delta calculation and calculate
    merge['Ve'] = 30
    merge['W'] = np.tan(0.5*np.arctan(merge['v_bulk']/428))
    merge['Delta_t'] = (merge['X']/merge['v_bulk']) * (1 + ((merge['Y']*merge['W'])/merge['X']))/(1 - merge['Ve']*merge['W']/merge['v_bulk'])
    #make new df with just time delta and icme timestamps as index
    t_df = pd.DataFrame(merge['Delta_t'])
    return t_df


def apply_time_delta(df):
    t = []
    for i in range(len(df)):
        new_t = df['Timestamp'].iloc[i] + timedelta(seconds=df['Delta_t'].iloc[i])
        t = np.append(t, new_t)
    df['New_Timestamp'] = t
    return df


def time_shift_df(df, t_df, resample_min):
    #resample df and set timestamp as index
    rdf = df.set_index('Timestamp').resample(f'{resample_min}min').mean().reset_index(drop=False)
    rdf.set_index(pd.to_datetime(rdf['Timestamp']), inplace=True)
    #concatenate df and t_df 
    shifted_rdf = pd.concat([rdf, t_df], axis=1)
    #back and forward fill t_df values 
    shifted_rdf['Delta_t'] = shifted_rdf['Delta_t'].ffill().bfill()
    #apply time_delta to each timestamp
    final_df = apply_time_delta(shifted_rdf)
    return final_df