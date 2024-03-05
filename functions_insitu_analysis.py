import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def timeshift_dataframe(df, speed):
    df_ts = df.copy(deep=True)

    df_ts['r2'] = 0.99285 #create new column of distance to propagate to
    df_ts['r_sep'] = df_ts['r2'] - df_ts['r'] #column of r separation
    df_ts['v_prop'] = speed
    df_ts['t_delay'] = df_ts['r_sep']*1.495978707E8/df_ts['v_prop']
    #
    t = []
    for i in range(len(df)):
        new_t = df_ts['time'].iloc[i] + timedelta(seconds=df_ts['t_delay'].iloc[i])
        t = np.append(t, new_t)
    df_ts['time_shifted'] = t
    #could add column dropping lines
    return df_ts


def scale_B_field(df, power=-1.64, power_upper=-1, power_lower=-2): #requires timeshifted dataframe
    #default set to leitner scaling relationship for B field strength
    df['bt_scaled'] = df['bt']*(df['r2']/df['r'])**(power)
    df['bx_scaled'] = df['bx']*(df['r2']/df['r'])**(power)
    df['by_scaled'] = df['by']*(df['r2']/df['r'])**(power)
    df['bz_scaled'] = df['bz']*(df['r2']/df['r'])**(power)
    #lower bound
    df['bt_scaled_lb'] = df['bt']*(df['r2']/df['r'])**(power_lower)
    df['bx_scaled_lb'] = df['bx']*(df['r2']/df['r'])**(power_lower)
    df['by_scaled_lb'] = df['by']*(df['r2']/df['r'])**(power_lower)
    df['bz_scaled_lb'] = df['bz']*(df['r2']/df['r'])**(power_lower)
    #upper bound
    df['bt_scaled_ub'] = df['bt']*(df['r2']/df['r'])**(power_upper)
    df['bx_scaled_ub'] = df['bx']*(df['r2']/df['r'])**(power_upper)
    df['by_scaled_ub'] = df['by']*(df['r2']/df['r'])**(power_upper)
    df['bz_scaled_ub'] = df['bz']*(df['r2']/df['r'])**(power_upper)
    return df


def timeshift_boundary(datetime, df, speed, speed_uncertainty=50):

    df_timeshifted = timeshift_dataframe(df, speed)
    idx = df_timeshifted.set_index('time').index.get_loc(datetime, method='nearest')

    t_ts = df_timeshifted['time_shifted'].iloc[idx]

    upper_df = timeshift_dataframe(df, speed-speed_uncertainty)
    t_ts_ub = upper_df['time_shifted'].iloc[idx]

    lower_df = timeshift_dataframe(df, speed+speed_uncertainty)
    t_ts_lb = lower_df['time_shifted'].iloc[idx]
    
    return t_ts, t_ts_lb, t_ts_ub


# def calc_time_delta(resampled_df, icme_df):
#     #drop old v_bulk column from resampled df 
#     resampled_df = resampled_df.drop(columns=['v_bulk'])
#     #merge with new velocity profile, which also produces just values within ICME
#     merge=pd.merge(icme_df, resampled_df, how='inner', left_index=True, right_index=True)
#     #add columns with new variables needed for time_delta calculation and calculate
#     merge['Ve'] = 30
#     merge['W'] = np.tan(0.5*np.arctan(merge['v_bulk']/428))
#     merge['Delta_t'] = (merge['X']/merge['v_bulk']) * (1 + ((merge['Y']*merge['W'])/merge['X']))/(1 - merge['Ve']*merge['W']/merge['v_bulk'])
#     #make new df with just time delta and icme timestamps as index
#     t_df = pd.DataFrame(merge['Delta_t'])
#     return t_df


# def apply_time_delta(df):
#     t = []
#     for i in range(len(df)):
#         new_t = df['Timestamp'].iloc[i] + timedelta(seconds=df['Delta_t'].iloc[i])
#         t = np.append(t, new_t)
#     df['New_Timestamp'] = t
#     return df


# def time_shift_df(df, t_df, resample_min):
#     #resample df and set timestamp as index
#     rdf = df.set_index('Timestamp').resample(f'{resample_min}min').mean().reset_index(drop=False)
#     rdf.set_index(pd.to_datetime(rdf['Timestamp']), inplace=True)
#     #concatenate df and t_df 
#     shifted_rdf = pd.concat([rdf, t_df], axis=1)
#     #back and forward fill t_df values 
#     shifted_rdf['Delta_t'] = shifted_rdf['Delta_t'].ffill().bfill()
#     #apply time_delta to each timestamp
#     final_df = apply_time_delta(shifted_rdf)
#     return final_df




# ##########
# def timeshift_dataframe_old(df, speed=450, distance_of_object_au=0.992854, sc_name="SolO", ref_name="L1"):
#     df_ts = df.copy(deep=True)

#     #solo positions (most recent)
#     r = df['r'][df.shape[0]-1]
#     # lat = df['lat'][df.shape[0]-1]
#     # lon = df['lon'][df.shape[0]-1]

#     #Earth radial dist = 0.992854
#     r_sep = distance_of_object_au-r

#     print(f'Distance {sc_name} to {ref_name} = {r_sep:.2f} AU')
#     print(f'Constant speed {speed} kms/from {sc_name} to {ref_name}')

#     au = 1.495978707E11 #divide from au to metres

#     t_delay=r_sep*au/(speed*1e3)/3600  #m, m/s, convert seconds to hours
#     print(f'Time Delay = {t_delay:.2f} hours')

#     df_ts['time'] = df['time']+timedelta(hours=t_delay)
#     # final_ts_df = df_ts[0]
#     return df_ts, r_sep, t_delay