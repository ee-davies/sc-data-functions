import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
import spiceypy
import os.path


#input datetime to return T1, T2 and T3 based on Hapgood 1992
#http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_geocentric_transformation_matrices(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours    
    #define position of geomagnetic pole in GEO coordinates
    pgeo=78.8+4.283*((mjd-46066)/365.25)*0.01 #in degrees
    lgeo=289.1-1.413*((mjd-46066)/365.25)*0.01 #in degrees
    #GEO vector
    Qg=[np.cos(pgeo*np.pi/180)*np.cos(lgeo*np.pi/180), np.cos(pgeo*np.pi/180)*np.sin(lgeo*np.pi/180), np.sin(pgeo*np.pi/180)]
    #now move to equation at the end of the section, which goes back to equations 2 and 4:
    #CREATE T1; T0, UT is known from above
    zeta=(100.461+36000.770*T0+15.04107*UT)*np.pi/180
    ################### theta und z
    T1=np.matrix([[np.cos(zeta), np.sin(zeta),  0], [-np.sin(zeta) , np.cos(zeta) , 0], [0,  0,  1]]) #angle for transpose
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180 #lamda sun
    #CREATE T2, LAMBDA, M, lt2 known from above
    ##################### lamdbda und Z
    t2z=np.matrix([[np.cos(lt2), np.sin(lt2),  0], [-np.sin(lt2) , np.cos(lt2) , 0], [0,  0,  1]])
    et2=(23.439-0.013*T0)*np.pi/180
    ###################### epsilon und x
    t2x=np.matrix([[1,0,0],[0,np.cos(et2), np.sin(et2)], [0, -np.sin(et2), np.cos(et2)]])
    T2=np.dot(t2z,t2x)  #equation 4 in Hapgood 1992
    #matrix multiplications   
    T2T1t=np.dot(T2,np.matrix.transpose(T1))
    ################
    Qe=np.dot(T2T1t,Qg) #Q=T2*T1^-1*Qq
    psigsm=np.arctan(Qe.item(1)/Qe.item(2)) #arctan(ye/ze) in between -pi/2 to +pi/2
    T3=np.matrix([[1,0,0],[0,np.cos(-psigsm), np.sin(-psigsm)], [0, -np.sin(-psigsm), np.cos(-psigsm)]])
    return T1, T2, T3


def get_heliocentric_transformation_matrices(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours
    #equation 12
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    #lamda sun in radians
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
    #S1 matrix
    S1=np.matrix([[np.cos(lt2+np.pi), np.sin(lt2+np.pi),  0], [-np.sin(lt2+np.pi) , np.cos(lt2+np.pi) , 0], [0,  0,  1]])
    #equation 13
    #create S2 matrix with angles with reversed sign for transformation HEEQ to HAE
    iota=7.25*np.pi/180
    omega=(73.6667+0.013958*((mjd+3242)/365.25))*np.pi/180 #in rad         
    theta=np.arctan(np.cos(iota)*np.tan(lt2-omega))
    #quadrant of theta must be opposite lt2 - omega; Hapgood 1992 end of section 5   
    #get lambda-omega angle in degree mod 360   
    lambda_omega_deg=np.mod(lt2-omega,2*np.pi)*180/np.pi
    x = np.cos(np.deg2rad(lambda_omega_deg))
    y = np.sin(np.deg2rad(lambda_omega_deg))
    #get theta_node in deg
    x_theta = np.cos(theta)
    y_theta = np.sin(theta)
    #if in same quadrant, then theta_node = theta_node +pi  
    if (x>=0 and y>=0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi
        elif (x_theta<=0 and y_theta<=0): theta = theta
        elif (x_theta>=0 and y_theta<=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta + np.pi/2
        
    elif (x<=0 and y<=0):
        if (x_theta>=0 and y_theta>=0): theta = theta
        elif (x_theta<=0 and y_theta<=0): theta = theta + np.pi
        elif (x_theta>=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta - np.pi/2
        
    elif (x>=0 and y<=0):
        if (x_theta>=0 and y_theta>=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = theta - np.pi/2 
        elif (x_theta>=0 and y_theta<=0): theta = theta + np.pi
        elif (x_theta<=0 and y_theta>=0): theta = theta

    elif (x<0 and y>0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta>=0 and y_theta<=0): theta = theta
        elif (x_theta<=0 and y_theta>=0): theta = theta - np.pi   

    s2_theta = np.matrix([[np.cos(theta), np.sin(theta),  0], [-np.sin(theta) , np.cos(theta) , 0], [0,  0,  1]])
    s2_iota = np.matrix([[1,  0,  0], [0, np.cos(iota), np.sin(iota)], [0, -np.sin(iota) , np.cos(iota)]])
    s2_omega = np.matrix([[np.cos(omega), np.sin(omega),  0], [-np.sin(omega) , np.cos(omega) , 0], [0,  0,  1]])
    S2 = np.dot(np.dot(s2_theta,s2_iota),s2_omega)

    return S1, S2


"""
Geocentric frame conversions
"""


def GSE_to_GSM_mag(df):
    # Get all transformation matrices at once
    times = df['time'].values
    T3_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in times])
    # Create coordinate matrix (N x 3) where N is number of rows
    coords = np.column_stack([df['bx'].values, df['by'].values, df['bz'].values])
    # Vectorized matrix multiplication using einsum
    # 'ijk,ik->ij' means: for each i, multiply matrix T3_matrices[i] with vector coords[i,:]
    GSM_coords = np.einsum('ijk,ik->ij', T3_matrices, coords)
    bx, by, bz = GSM_coords[:, 0], GSM_coords[:, 1], GSM_coords[:, 2]
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': df['bt'],
        'bx': bx,
        'by': by, 
        'bz': bz,
    })
    return df_transformed


def GSE_to_GSM_plas(df):
    # Get all transformation matrices at once
    times = df['time'].values
    T3_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in times])
    # Create coordinate matrix (N x 3) where N is number of rows
    coords = np.column_stack([df['vx'].values, df['vy'].values, df['vz'].values])
    # Vectorized matrix multiplication using einsum
    # 'ijk,ik->ij' means: for each i, multiply matrix T3_matrices[i] with vector coords[i,:]
    GSM_coords = np.einsum('ijk,ik->ij', T3_matrices, coords)
    vx, vy, vz = GSM_coords[:, 0], GSM_coords[:, 1], GSM_coords[:, 2]
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': df['vt'],
        'vx': vx,
        'vy': vy, 
        'vz': vz,
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed


def GSM_to_GSE_mag(df):
    # Get all transformation matrices at once
    times = df['time'].values
    T3_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in times])
    # Compute inverse matrices for all T3 matrices at once
    T3_inv_matrices = np.linalg.inv(T3_matrices)
    # Create coordinate matrix (N x 3) where N is number of rows
    coords = np.column_stack([df['bx'].values, df['by'].values, df['bz'].values])
    # Vectorized matrix multiplication using einsum
    # 'ijk,ik->ij' means: for each i, multiply matrix T3_inv_matrices[i] with vector coords[i,:]
    GSE_coords = np.einsum('ijk,ik->ij', T3_inv_matrices, coords)
    bx, by, bz = GSE_coords[:, 0], GSE_coords[:, 1], GSE_coords[:, 2]
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': df['bt'],
        'bx': bx,
        'by': by, 
        'bz': bz,
    })
    return df_transformed


def GSM_to_GSE_plas(df):
    # Get all transformation matrices at once
    times = df['time'].values
    T3_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in times])
    # Compute inverse matrices for all T3 matrices at once
    T3_inv_matrices = np.linalg.inv(T3_matrices)
    # Create coordinate matrix (N x 3) where N is number of rows
    coords = np.column_stack([df['vx'].values, df['vy'].values, df['vz'].values])
    # Vectorized matrix multiplication using einsum
    # 'ijk,ik->ij' means: for each i, multiply matrix T3_inv_matrices[i] with vector coords[i,:]
    GSE_coords = np.einsum('ijk,ik->ij', T3_inv_matrices, coords)
    vx, vy, vz = GSE_coords[:, 0], GSE_coords[:, 1], GSE_coords[:, 2]
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': df['vt'],
        'vx': vx,
        'vy': vy, 
        'vz': vz,
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed


"""
Geocentric to RTN frame conversions
"""


def GSE_to_RTN_approx_mag(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': df['bt'],
        'bx': -df['bx'],
        'by': -df['by'], 
        'bz': df['bz'],
    })
    return df_transformed


def GSE_to_RTN_approx_plas(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': df['vt'],
        'vx': -df['vx'],
        'vy': -df['vy'], 
        'vz': df['vz'],
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed

    
def GSM_to_RTN_approx_mag(df):
    df_gse = GSM_to_GSE_mag(df)
    df_transformed = GSE_to_RTN_approx_mag(df_gse)
    return df_transformed


def GSM_to_RTN_approx_plas(df):
    df_gse = GSM_to_GSE_plas(df)
    df_transformed = GSE_to_RTN_approx_plas(df_gse)
    return df_transformed


"""
Geocentric - Heliocentric frame conversions
"""


def GSE_to_HEE_mag(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': df['bt'],
        'bx': -df['bx'],
        'by': -df['by'], 
        'bz': df['bz'],
    })
    return df_transformed


def GSE_to_HEE_plas(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': df['vt'],
        'vx': -df['vx'],
        'vy': -df['vy'], 
        'vz': df['vz'],
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed


def HEE_to_GSE_mag(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': df['bt'],
        'bx': -df['bx'],
        'by': -df['by'], 
        'bz': df['bz'],
    })
    return df_transformed


def HEE_to_GSE_plas(df):
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': df['vt'],
        'vx': -df['vx'],
        'vy': -df['vy'], 
        'vz': df['vz'],
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed


"""
Heliocentric frame conversions
"""


def HEE_to_HAE(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[i])
        S1_inv = np.linalg.inv(S1)
        B_HEE_i = np.matrix([[df['bx'].iloc[i]],[df['by'].iloc[i]],[df['bz'].iloc[i]]]) 
        B_HEA_i = np.dot(S1_inv,B_HEE_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        B_HAE.append(flat_B_HAE_i)
    df_transformed = pd.DataFrame(B_HAE, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def HAE_to_HEE(df):
    B_HEE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[i])
        B_HAE_i = np.matrix([[df['bx'].iloc[i]],[df['by'].iloc[i]],[df['bz'].iloc[i]]]) 
        B_HEE_i = np.dot(S1,B_HAE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        B_HEE.append(flat_B_HEE_i)
    df_transformed = pd.DataFrame(B_HEE, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def HAE_to_HEEQ(df):
    B_HEEQ = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[i])
        B_HAE_i = np.matrix([[df['bx'].iloc[i]],[df['by'].iloc[i]],[df['bz'].iloc[i]]]) 
        B_HEEQ_i = np.dot(S2,B_HAE_i)
        B_HEEQ_i_list = B_HEEQ_i.tolist()
        flat_B_HEEQ_i = list(itertools.chain(*B_HEEQ_i_list))
        B_HEEQ.append(flat_B_HEEQ_i)
    df_transformed = pd.DataFrame(B_HEEQ, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def HEEQ_to_HAE(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[i])
        S2_inv = np.linalg.inv(S2)
        B_HEEQ_i = np.matrix([[df['bx'].iloc[i]],[df['by'].iloc[i]],[df['bz'].iloc[i]]]) 
        B_HEA_i = np.dot(S2_inv,B_HEEQ_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        B_HAE.append(flat_B_HAE_i)
    df_transformed = pd.DataFrame(B_HAE, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def HEE_to_HEEQ(df):
    df_hae = HEE_to_HAE(df)
    df_transformed = HAE_to_HEEQ(df_hae)
    return df_transformed


def HEEQ_to_HEE(df):
    df_hae = HEEQ_to_HAE(df)
    df_transformed = HAE_to_HEE(df_hae)
    return df_transformed


"""
Heliocentric to RTN frame conversions

# RTN transforms require spacecraft position information
# Often position files will not have same timestamps as datafiles, 
# so first need to be interpolated to the same timestamps as the data being transformed
# input DataFrames are required to be pre-combined with position xyz
"""


def interp_to_newtimes(df1, df2):
    #set time as index of dataframe you want to interpolate
    df1.set_index(df1.time, inplace=True)
    df1 = df1.drop(columns=['time'])
    #define new indices using different dataframe
    new_indices_mag = pd.DatetimeIndex(df2.time)
    #perform interpolation of df1 to df2 times
    out = (df1.reindex(df1.index.union(new_indices_mag))
         .interpolate('time').loc[new_indices_mag])
    df_new = out.reset_index(drop=False)
    return df_new


def combine_dataframes(df1,df2):
        df1.set_index(df1.time, inplace=True)
        df1 = df1.drop(columns=['time'])
        df2.set_index(df2.time, inplace=True)
        df2 = df2.drop(columns=['time'])
        combined_df = pd.concat([df1, df2], axis=1)
        combined_df = combined_df.reset_index(drop=False)
        return combined_df


#function to transform mag data from HEEQ to RTN, using a DataFrame which has already combined mag, plas and position with same timestamps
def HEEQ_to_RTN(df):
    #unit vectors of HEEQ basis
    heeq_x=[1,0,0]
    heeq_y=[0,1,0]
    heeq_z=[0,0,1]
    B_RTN = []
    for i in range(df.shape[0]):
        #make unit vectors of RTN in basis of HEEQ
        rtn_r = [df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]]/np.linalg.norm([df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]])
        rtn_t=np.cross(heeq_z,rtn_r)
        rtn_n=np.cross(rtn_r,rtn_t)

        br_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_r)+df['by'].iloc[i]*np.dot(heeq_y,rtn_r)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_r)
        bt_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_t)+df['by'].iloc[i]*np.dot(heeq_y,rtn_t)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_t)
        bn_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_n)+df['by'].iloc[i]*np.dot(heeq_y,rtn_n)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_n)
        B_RTN_i = [br_i, bt_i, bn_i]
        B_RTN.append(B_RTN_i)
    df_transformed = pd.DataFrame(B_RTN, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def HEEQ_to_RTN_mag(df):
    #unit vectors of HEEQ basis
    heeq_x=[1,0,0]
    heeq_y=[0,1,0]
    heeq_z=[0,0,1]
    B_RTN = []
    for i in range(df.shape[0]):
        #make unit vectors of RTN in basis of HEEQ
        rtn_r = [df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]]/np.linalg.norm([df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]])
        rtn_t=np.cross(heeq_z,rtn_r)/np.linalg.norm(np.cross(heeq_z,rtn_r))
        rtn_n=np.cross(rtn_r,rtn_t)/np.linalg.norm(np.cross(rtn_r, rtn_t))

        br_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_r)+df['by'].iloc[i]*np.dot(heeq_y,rtn_r)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_r)
        bt_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_t)+df['by'].iloc[i]*np.dot(heeq_y,rtn_t)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_t)
        bn_i=df['bx'].iloc[i]*np.dot(heeq_x,rtn_n)+df['by'].iloc[i]*np.dot(heeq_y,rtn_n)+df['bz'].iloc[i]*np.dot(heeq_z,rtn_n)
        B_RTN_i = [br_i, bt_i, bn_i]
        B_RTN.append(B_RTN_i)

    B_RTN = np.array(B_RTN)
    bx, by, bz = B_RTN[:, 0], B_RTN[:, 1], B_RTN[:, 2]
    bt = np.linalg.norm([bx,by,bz], axis=0)
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'bt': bt,
        'bx': bx,
        'by': by, 
        'bz': bz,
    })
    return df_transformed


def HEEQ_to_RTN_plas(df):
    #unit vectors of HEEQ basis
    heeq_x=[1,0,0]
    heeq_y=[0,1,0]
    heeq_z=[0,0,1]
    V_RTN = []
    for i in range(df.shape[0]):
        #make unit vectors of RTN in basis of HEEQ
        rtn_r = [df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]]/np.linalg.norm([df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]])
        rtn_t=np.cross(heeq_z,rtn_r)/np.linalg.norm(np.cross(heeq_z,rtn_r))
        rtn_n=np.cross(rtn_r,rtn_t)/np.linalg.norm(np.cross(rtn_r, rtn_t))

        vr_i=df['vx'].iloc[i]*np.dot(heeq_x,rtn_r)+df['vy'].iloc[i]*np.dot(heeq_y,rtn_r)+df['vz'].iloc[i]*np.dot(heeq_z,rtn_r)
        vt_i=df['vx'].iloc[i]*np.dot(heeq_x,rtn_t)+df['vy'].iloc[i]*np.dot(heeq_y,rtn_t)+df['vz'].iloc[i]*np.dot(heeq_z,rtn_t)
        vn_i=df['vx'].iloc[i]*np.dot(heeq_x,rtn_n)+df['vy'].iloc[i]*np.dot(heeq_y,rtn_n)+df['vz'].iloc[i]*np.dot(heeq_z,rtn_n)
        V_RTN_i = [vr_i, vt_i, vn_i]
        V_RTN.append(V_RTN_i)

    V_RTN = np.array(V_RTN)
    vx, vy, vz = V_RTN[:, 0], V_RTN[:, 1], V_RTN[:, 2]
    vt = np.linalg.norm([vx,vy,vz], axis=0)
    # Create result DataFrame
    df_transformed = pd.DataFrame({
        'time': df['time'].values,
        'vt': vt,
        'vx': vx,
        'vy': vy, 
        'vz': vz,
        'tp': df['tp'],
        'np': df['np'],
    })
    return df_transformed


def RTN_to_HEEQ(df):
    #HEEQ unit vectors (same as spacecraft xyz position)
    heeq_x=[1,0,0]
    heeq_y=[0,1,0]
    heeq_z=[0,0,1]
    B_HEEQ = []
    # go through all data points    
    for i in range(df.shape[0]):                
        #make normalized RTN unit vectors from spacecraft position in HEEQ basis
        rtn_x=[df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]]/np.linalg.norm([df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]])
        rtn_y=np.cross(heeq_z,rtn_x)/np.linalg.norm(np.cross(heeq_z,rtn_x))
        rtn_z=np.cross(rtn_x, rtn_y)/np.linalg.norm(np.cross(rtn_x, rtn_y))
        
        #project into new system (HEEQ)
        bx_i=np.dot(np.dot(df['bx'].iloc[i],rtn_x)+np.dot(df['by'].iloc[i],rtn_y)+np.dot(df['bz'].iloc[i],rtn_z),heeq_x)
        by_i=np.dot(np.dot(df['bx'].iloc[i],rtn_x)+np.dot(df['by'].iloc[i],rtn_y)+np.dot(df['bz'].iloc[i],rtn_z),heeq_y)
        bz_i=np.dot(np.dot(df['bx'].iloc[i],rtn_x)+np.dot(df['by'].iloc[i],rtn_y)+np.dot(df['bz'].iloc[i],rtn_z),heeq_z)
        B_HEEQ_i = [bx_i, by_i, bz_i]
        B_HEEQ.append(B_HEEQ_i)
    df_transformed = pd.DataFrame(B_HEEQ, columns=['bx', 'by', 'bz'])
    df_transformed['bt'] = np.linalg.norm(df_transformed[['bx', 'by', 'bz']], axis=1)
    df_transformed['time'] = df['time']
    df_transformed['vx'] = df['vx']
    df_transformed['vy'] = df['vy']
    df_transformed['vz'] = df['vz']
    df_transformed['vt'] = df['vt']
    df_transformed['np'] = df['np']
    df_transformed['tp'] = df['tp']
    df_transformed['x'] = df['x']
    df_transformed['y'] = df['y']
    df_transformed['z'] = df['z']
    df_transformed['y'] = df['y']
    df_transformed['r'] = df['r']
    df_transformed['lat'] = df['lat']
    df_transformed['lon'] = df['lon']
    return df_transformed


def RTN_to_HAE(df_rtn):
    df_heeq = RTN_to_HEEQ(df_rtn)
    df_hae = HEEQ_to_HAE(df_heeq)
    return df_hae


def RTN_to_HEE(df_rtn):
    df_hae = RTN_to_HAE(df_rtn)
    df_hee = HAE_to_HEE(df_hae)
    return df_hee


def RTN_to_GSE(df_rtn):
    df_hee = RTN_to_HEE(df_rtn)
    df_gse = HEE_to_GSE(df_hee)
    return df_gse


def RTN_to_GSM(df_rtn):
    df_gse = RTN_to_GSE(df_rtn)
    df_gsm = GSE_to_GSM(df_gse)
    return df_gsm


def generic_furnish():
    """Main"""
    kernels_path='/Volumes/External/data/kernels/'
    generic_path = kernels_path+'generic/'
    generic_kernels = os.listdir(generic_path)
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_transform(epoch: datetime, base_frame: str, to_frame: str):
    """Return transformation matrix at a given epoch."""
    transform = spiceypy.pxform(base_frame, to_frame, spiceypy.datetime2et(epoch))
    return transform


def perform_mag_transform(df, base_frame: str, to_frame: str):
    generic_furnish()
    timeseries = df.time
    BASE = np.vstack((df.bx, df.by, df.bz)).T
    transformation_matrices = np.array([get_transform(t, base_frame, to_frame) for t in timeseries])
    TO = np.einsum('ijk,ik->ij', transformation_matrices, BASE)
    df_transformed = pd.concat([timeseries], axis=1)
    df_transformed['bt'] = df.bt
    df_transformed['bx'] = TO[:,0]
    df_transformed['by'] = TO[:,1]
    df_transformed['bz'] = TO[:,2]
    spiceypy.kclear()
    return df_transformed


def perform_plas_transform(df, base_frame: str, to_frame: str):
    generic_furnish()
    timeseries = df.time
    BASE = np.vstack((df.vx, df.vy, df.vz)).T
    transformation_matrices = np.array([get_transform(t, base_frame, to_frame) for t in timeseries])
    TO = np.einsum('ijk,ik->ij', transformation_matrices, BASE)
    df_transformed = pd.concat([timeseries], axis=1)
    df_transformed['vt'] = df.vt
    df_transformed['vx'] = TO[:,0]
    df_transformed['vy'] = TO[:,1]
    df_transformed['vz'] = TO[:,2]
    df_transformed['tp'] = df.tp
    df_transformed['np'] = df.np
    spiceypy.kclear()
    return df_transformed