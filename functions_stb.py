def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'Timestamp']
    df.loc[mask, cols] = np.nan
    return df


def get_stereomag(fp):
    cdf = pycdf.CDF(fp)
    data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(['Epoch', 'BTOTAL'], ['Timestamp', 'B_TOT'])}
    df = pd.DataFrame.from_dict(data)
    bx, by, bz = cdf['BFIELDRTN'][:].T
    df['B_R'] = bx
    df['B_T'] = by
    df['B_N'] = bz
    return filter_bad_data(df, 'B_TOT', -9.99e+29)


# def get_stereoplas(fp):
#     cdf = pycdf.CDF(fp)
#     cols_raw = ['Epoch', 'Vp_RTN', 'Vr_Over_V_RTN', 'Vt_Over_V_RTN', 'Vn_Over_V_RTN', 'Tp', 'Np']
#     cols_new = ['Timestamp', 'v_bulk', 'v_x', 'v_y', 'v_z', 'v_therm', 'density']
#     data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
#     df = pd.DataFrame.from_dict(data)
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     for col in cols_new[1:]:
#         df[col] = df[col].astype('float32')
#     return filter_bad_data(df, 'v_bulk', -9.99e+04)


def get_stereoplas(fp):
    cdf = pycdf.CDF(fp)
    cols_raw = ['epoch', 'proton_bulk_speed', 'proton_Vr_RTN', 'proton_Vt_RTN', 'proton_Vn_RTN', 'proton_temperature', 'proton_number_density']
    cols_new = ['Timestamp', 'v_bulk', 'v_x', 'v_y', 'v_z', 'v_therm', 'density']
    data = {df_col: cdf[cdf_col][:] for cdf_col, df_col in zip(cols_raw, cols_new)}
    df = pd.DataFrame.from_dict(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in cols_new[1:]:
        df[col] = df[col].astype('float32')
    return filter_bad_data(df, 'v_bulk', -9.99e+04)



def get_stereobmag_range(start_timestamp, end_timestamp, path=r'D:/STEREO_B'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date()
    while start < end:
        year = start.year
        fn = f'stb_l2_magplasma_1m_{year}0101_v01.cdf'
        _df = get_stereomag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=366)
    return df



# def get_stereobplas_range(start_timestamp, end_timestamp, path=r'D:/STEREO_B'):
#     """Pass two datetime objects and grab .STS files between dates, from
#     directory given."""
#     df = None
#     start = start_timestamp.date()
#     end = end_timestamp.date()
#     while start < end:
#         year = start.year
#         fn = f'stb_l2_magplasma_1m_{year}0101_v01.cdf'
#         _df = get_stereoplas(f'{path}/{fn}')
#         if _df is not None:
#             if df is None:
#                 df = _df.copy(deep=True)
#             else:
#                 df = df.append(_df.copy(deep=True))
#         start += timedelta(days=366)
#     return df


def get_stereobplas_range(start_timestamp, end_timestamp, path=r'D:/STEREO_B/PLA'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'stb_l2_pla_1dmax_1min_{date_str}_v09.cdf'
        _df = get_stereoplas(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df