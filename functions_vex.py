def get_vexmag(fp):
    """Reformats a .TAB file to .csv equivalent."""
    if not fp.endswith('.TAB'):
        raise Exception('Wrong filetype passed, must end with .TAB...')
    cols = ['Timestamp', 'B_R', 'B_T', 'B_N', 'B_TOT', 'X_POS', 'Y_POS', 'Z_POS', 'R_POS']
    i = 0  # instantiate
    i_stop = 500  # initial
    check_table = True
    data = []
    try:
        with open(fp, 'r') as f:
            lines = f.readlines()
            while i < i_stop:
                if check_table:
                    if lines[i].startswith('^TABLE'):
                        i_stop = int(lines[i].split('=')[-1].strip()) - 1
                        check_table = False
                i += 1
            for line in lines[i:]:
                data.append(line.split())
        df = pd.DataFrame(data, columns=cols)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        for col in cols[1:]:
            df[col] = df[col].astype('float32')
        df['B_R'] = -1 * df['B_R']
        df['B_T'] = -1 * df['B_T']
        df = filter_bad_data(df, 'B_TOT', 9.99e+04)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_vexmag_range(start_timestamp, end_timestamp, path=r'D:/VEX'):
    """Pass two datetime objects and grab .STS files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        year = start.year
        doy = start.strftime('%j')
        date_str = f'{year}{start.month:02}{start.day:02}'
        fn = f'MAG_{date_str}_DOY{doy}_S004_V1.TAB'
        _df = get_vexmag(f'{path}/{fn}')
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = df.append(_df.copy(deep=True))
        start += timedelta(days=1)
    return df