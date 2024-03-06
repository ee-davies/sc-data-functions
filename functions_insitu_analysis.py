import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def timeshift_dataframe_predspeed(df, speed): #uses predicted constant speed by ELEvoHI 
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


def timeshift_dataframe_predtime(df, t_shock, pred_arrival_time): #uses predicted arrival time at L1 ELEvoHI 
    df_ts = df.copy(deep=True)

    t_delta = pred_arrival_time - t_shock
    df_ts['t_delta'] = t_delta
    df_ts['time_shifted'] = df_ts['time'] + df_ts['t_delta']
    return df_ts


def expand_icme(df_timeshifted, def_ref_sc, t_le, t_te, power=0.8):

    df_s = df_timeshifted.copy(deep=True)

    mo_mask = (df_timeshifted['time'] >= t_le) & (df_timeshifted['time'] <= t_te)
    prior_mask = (df_timeshifted['time'] < t_le)
    post_mask = (df_timeshifted['time'] > t_te)

    D1 = (t_te - t_le).total_seconds()
    FR = df_timeshifted[mo_mask]
    r1 = FR['r'].mean()
    c = np.log(D1) - power*np.log(r1)

    idx = df_s.set_index('time').index.get_loc(t_le, method='nearest')
    ts_le = df_s['time_shifted'].iloc[idx]

    idx2 = def_ref_sc.set_index('time').index.get_loc(ts_le, method='nearest')
    r2 = def_ref_sc['r'].iloc[idx2]
    D2 = np.exp(c + power*np.log(r2))

    expansion_delta = np.linspace(0, len(df_timeshifted[mo_mask])-1, len(df_timeshifted[mo_mask]))*60*(D2/D1)

    df_timeshifted['expansion_delta'] = np.nan
    df_mo = df_timeshifted[mo_mask].assign(expansion_delta=expansion_delta)
    df_prior = df_timeshifted[prior_mask].assign(expansion_delta=expansion_delta.min())
    df_post = df_timeshifted[post_mask].assign(expansion_delta=expansion_delta.max())

    stitched_df = pd.concat([df_prior, df_mo, df_post])

    t = []
    for i in range(len(stitched_df)):
        new_t = stitched_df['time_shifted'].iloc[i] + timedelta(seconds=stitched_df['expansion_delta'].iloc[i])
        t = np.append(t, new_t)
    stitched_df['time_shifted_exp'] = t

    return stitched_df


def timeshift_boundary_predspeed(datetime, df, speed, speed_uncertainty=50):

    df_timeshifted = timeshift_dataframe_predspeed(df, speed)
    idx = df_timeshifted.set_index('time').index.get_loc(datetime, method='nearest')

    t_ts = df_timeshifted['time_shifted_exp'].iloc[idx]

    upper_df = timeshift_dataframe_predspeed(df, speed-speed_uncertainty)
    t_ts_ub = upper_df['time_shifted_exp'].iloc[idx]

    lower_df = timeshift_dataframe_predspeed(df, speed+speed_uncertainty)
    t_ts_lb = lower_df['time_shifted_exp'].iloc[idx]
    
    return t_ts, t_ts_lb, t_ts_ub


def timeshift_boundary_predtime(df_timeshifted, boundary_datetime, boundary_uncertainty):

    df_s = df_timeshifted.copy(deep=True)

    idx = df_s.set_index('time').index.get_loc(boundary_datetime, method='nearest')
    t_ts = df_s['time_shifted_exp'].iloc[idx]

    upper_bound = boundary_datetime - timedelta(hours=boundary_uncertainty)
    idx2 = df_s.set_index('time').index.get_loc(upper_bound, method='nearest')
    t_ts_ub = df_s['time_shifted_exp'].iloc[idx2]

    lower_bound = boundary_datetime + timedelta(hours=boundary_uncertainty)
    idx3 = df_s.set_index('time').index.get_loc(lower_bound, method='nearest')
    t_ts_lb = df_s['time_shifted_exp'].iloc[idx3]
    
    return t_ts, t_ts_lb, t_ts_ub


def scale_B_field(df1, df2, power=-1.64, power_upper=-1, power_lower=-2): #requires timeshifted dataframe
    #observing spacecraft e.g. solo, round timeshifted times to nearest min, set as index to join with reference spaecraft e.g. dscovr
    df_timeshifted = df1.copy(deep=True)
    df_timeshifted['time_shifted_round'] = df_timeshifted['time_shifted_exp'].round('1min')
    df_timeshifted.set_index(pd.to_datetime(df_timeshifted['time_shifted_round']), inplace=True)
    # reference spacecraft df e.g dscovr -> get r2
    df_reference = df2.copy(deep=True)
    df_reference.set_index(pd.to_datetime(df_reference['time']), inplace=True)
    df_reference = df_reference.rename(columns={"r": "r2", "time":"time2"})
    df_reference = df_reference.drop(['bx', 'by', 'bz', 'bt', 'vx', 'vy', 'vz', 'vt', 'np', 'tp', 'x', 'y', 'z', 'lat', 'lon'], axis=1)
    df_reference = df_reference[df_reference['r2'].notna()]
    #combine dataframes at timeshifted index
    df = pd.concat([df_timeshifted, df_reference], axis=1)
    df = df[df['time_shifted_round'].notna()]
    df = df.reset_index(drop=True)
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
    #filter data for nans (ruins later plotly shading if not removed)
    df = df[df['bt_scaled'].notna()]
    return df


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

# def scale_B_field_outdated(df, power=-1.64, power_upper=-1, power_lower=-2): #requires timeshifted dataframe
#     #default set to leitner scaling relationship for B field strength
#     df['bt_scaled'] = df['bt']*(df['r2']/df['r'])**(power)
#     df['bx_scaled'] = df['bx']*(df['r2']/df['r'])**(power)
#     df['by_scaled'] = df['by']*(df['r2']/df['r'])**(power)
#     df['bz_scaled'] = df['bz']*(df['r2']/df['r'])**(power)
#     #lower bound
#     df['bt_scaled_lb'] = df['bt']*(df['r2']/df['r'])**(power_lower)
#     df['bx_scaled_lb'] = df['bx']*(df['r2']/df['r'])**(power_lower)
#     df['by_scaled_lb'] = df['by']*(df['r2']/df['r'])**(power_lower)
#     df['bz_scaled_lb'] = df['bz']*(df['r2']/df['r'])**(power_lower)
#     #upper bound
#     df['bt_scaled_ub'] = df['bt']*(df['r2']/df['r'])**(power_upper)
#     df['bx_scaled_ub'] = df['bx']*(df['r2']/df['r'])**(power_upper)
#     df['by_scaled_ub'] = df['by']*(df['r2']/df['r'])**(power_upper)
#     df['bz_scaled_ub'] = df['bz']*(df['r2']/df['r'])**(power_upper)
#     return df