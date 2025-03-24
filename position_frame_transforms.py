import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


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
Geocentric position conversions
"""


def GSE_to_GSM(df):
    B_GSM = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_geocentric_transformation_matrices(df['time'].iloc[0])
        B_GSE_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_GSM_i = np.dot(T3,B_GSE_i)
        B_GSM_i_list = B_GSM_i.tolist()
        flat_B_GSM_i = list(itertools.chain(*B_GSM_i_list))
        r, lat, lon = cart2sphere(flat_B_GSM_i[0], flat_B_GSM_i[1], flat_B_GSM_i[2])
        position = flat_B_GSM_i[0], flat_B_GSM_i[1], flat_B_GSM_i[2], r, lat, lon
        B_GSM.append(position)
    df_transformed = pd.DataFrame(B_GSM, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    df_transformed['time'] = df['time']
    return df_transformed


def GSM_to_GSE(df):
    B_GSE = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_geocentric_transformation_matrices(df['time'].iloc[0])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_GSE_i = np.dot(T3_inv,B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        r, lat, lon = cart2sphere(flat_B_GSE_i[0], flat_B_GSE_i[1], flat_B_GSE_i[2])
        position = flat_B_GSE_i[0], flat_B_GSE_i[1], flat_B_GSE_i[2], r, lat, lon
        B_GSE.append(position)
    df_transformed = pd.DataFrame(B_GSE, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    df_transformed['time'] = df['time']
    return df_transformed


"""
Heliocentric position conversions
"""


def HEE_to_HAE_old(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[0])
        S1_inv = np.linalg.inv(S1)
        B_HEE_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_HEA_i = np.dot(S1_inv,B_HEE_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        r, lat, lon = cart2sphere(flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2])
        position = flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2], r, lat, lon
        B_HAE.append(position)
    df_transformed = pd.DataFrame(B_HAE, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    # df_transformed['time'] = df['time']
    return df_transformed


def HEE_to_HAE(df):
    timeseries = df.time
    HEE = np.vstack((df.x, df.y, df.z)).T
    transformation_matrices = np.array([np.linalg.inv(get_heliocentric_transformation_matrices(t)[0]) for t in timeseries])
    HAE = np.einsum('ijk,ik->ij', transformation_matrices, HEE)
    r, lat, lon = cart2sphere(HAE[:,0],HAE[:,1],HAE[:,2])
    df_transformed = pd.concat([timeseries], axis=1)
    df_transformed['x'] = HAE[:,0]
    df_transformed['y'] = HAE[:,1]
    df_transformed['z'] = HAE[:,2]
    df_transformed['r'] = r
    df_transformed['lat'] = lat
    df_transformed['lon'] = lon
    return df_transformed


def HAE_to_HEE(df):
    B_HEE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[0])
        B_HAE_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_HEE_i = np.dot(S1,B_HAE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        r, lat, lon = cart2sphere(flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2])
        position = flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2], r, lat, lon
        B_HEE.append(position)
    df_transformed = pd.DataFrame(B_HEE, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    df_transformed['time'] = df['time']
    return df_transformed


def HAE_to_HEEQ_old(df):
    B_HEEQ = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[0])
        B_HAE_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_HEEQ_i = np.dot(S2,B_HAE_i)
        B_HEEQ_i_list = B_HEEQ_i.tolist()
        flat_B_HEEQ_i = list(itertools.chain(*B_HEEQ_i_list))
        r, lat, lon = cart2sphere(flat_B_HEEQ_i[0], flat_B_HEEQ_i[1], flat_B_HEEQ_i[2])
        position = flat_B_HEEQ_i[0], flat_B_HEEQ_i[1], flat_B_HEEQ_i[2], r, lat, lon
        B_HEEQ.append(position)
    df_transformed = pd.DataFrame(B_HEEQ, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    # df_transformed['time'] = df['time']
    return df_transformed


def HAE_to_HEEQ(df):
    timeseries = df.time
    HAE = np.vstack((df.x, df.y, df.z)).T
    transformation_matrices = np.array([get_heliocentric_transformation_matrices(t)[1] for t in timeseries])
    HEEQ = np.einsum('ijk,ik->ij', transformation_matrices, HAE)
    r, lat, lon = cart2sphere(HEEQ[:,0],HEEQ[:,1],HEEQ[:,2])
    df_transformed = pd.concat([timeseries], axis=1)
    df_transformed['x'] = HEEQ[:,0]
    df_transformed['y'] = HEEQ[:,1]
    df_transformed['z'] = HEEQ[:,2]
    df_transformed['r'] = r
    df_transformed['lat'] = lat
    df_transformed['lon'] = lon
    return df_transformed


def HEEQ_to_HAE(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df['time'].iloc[0])
        S2_inv = np.linalg.inv(S2)
        B_HEEQ_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_HEA_i = np.dot(S2_inv,B_HEEQ_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        r, lat, lon = cart2sphere(flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2])
        position = flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2], r, lat, lon
        B_HAE.append(position)
    df_transformed = pd.DataFrame(B_HAE, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    df_transformed['time'] = df['time']
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
Geocentric to heliocentric position conversions
#requires extra step in the conversion of GSE to HEE i.e. adding position vector of Sun
"""

def get_rsun_position_vector(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180 #lamda sun
    #section 6.1
    r_0 = 1.495985E8 #units km
    e = 0.016709 - 0.0000418*T0
    omega_bar = 282.94 + 1.72*T0
    v = lt2 - omega_bar
    #final r_sun equation
    r_sun = (r_0*(1 - e**2)) / (1 + e*np.cos(v))
    R_sun = np.matrix([[r_sun],[0],[0]])
    return R_sun


def get_rsun_position(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180 #lamda sun
    #section 6.1
    r_0 = 1.495985E8 #units km
    e = 0.016709 - 0.0000418*T0
    omega_bar = 282.94 + 1.72*T0
    v = lt2 - omega_bar
    #final r_sun equation
    r_sun = (r_0*(1 - e**2)) / (1 + e*np.cos(v))
    return r_sun


def GSE_to_HEE_old(df):
    B_HEE = []
    z_rot_180 = np.matrix([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
    for i in range(df.shape[0]):
        R_sun = get_rsun_position_vector(df['time'].iloc[0])
        B_GSE_i = np.matrix([[df['x'].iloc[i]],[df['y'].iloc[i]],[df['z'].iloc[i]]]) 
        B_HEE_i = R_sun + np.dot(z_rot_180,B_GSE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        r, lat, lon = cart2sphere(flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2])
        position = flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2], r, lat, lon
        B_HEE.append(position)
    df_transformed = pd.DataFrame(B_HEE, columns=['x', 'y', 'z', 'r', 'lat', 'lon'])
    #df time replication no longer works
    return df_transformed


def GSE_to_HEE(df):
    timeseries = df.time
    r_suns = []
    for t in timeseries:
        r_sun = get_rsun_position(t)
        r_suns.append(r_sun)
    x = -df.x + r_suns
    y = -df.y
    z = df.z
    r, lat, lon = cart2sphere(x,y,z)
    df_transformed = pd.concat([timeseries, x, y, z], axis=1)
    df_transformed['r'] = r
    df_transformed['lat'] = lat
    df_transformed['lon'] = lon
    return df_transformed