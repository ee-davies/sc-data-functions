import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import urllib.request


#input station as 'SOPO' or 'MCMU'
def download_nm_data(i, station, icme_start, mo_end, output_dir = '/Users/emmadavies/Library/CloudStorage/OneDrive-USNH/FDs_BDEs_complexity/WIND_CAT/'):
    start = icme_start.date()-timedelta(days=5)
    end = mo_end.date()+timedelta(days=10)

    # station = 'SOPO' #choice of station #MCMU #SOPO
    tabchoice = 'revori' #revori #ori
    data_type = 'corr_for_efficiency'
    date_choice = 'bydate' #last
    tresolution = 10
    start_year = start.year
    start_month = start.month #2d.p.
    start_day = start.day #2d.p.
    start_hour = 1 #2d.p.
    start_min = 1 #2d.p.
    end_year = end.year
    end_month = end.month #2d.p.
    end_day = end.day #2d.p.
    end_hour = 1 #2d.p.
    end_min = 1 #2d.p.
    
    plot_url = f'http://nest.nmdb.eu/draw_graph.php?formchk=1&stations[]={station}&tabchoice={tabchoice}&dtype={data_type}&tresolution=5&yunits=0&shift=2&date_choice={date_choice}&start_day={start_day}&start_month={start_month}&start_year={start_year}&start_hour={start_hour}&start_min={start_min}&end_day={end_day}&end_month={end_month}&end_year={end_year}&end_hour={end_hour}&end_min={end_min}&output=plot&ygrid=1&mline=1&transp=0&fontsize=1&text_color=222222&background_color=FFFFFF&margin_color=FFFFFF'
    data_url = f'http://nest.nmdb.eu/draw_graph.php?formchk=1&stations[]={station}&tabchoice={tabchoice}&dtype={data_type}&tresolution={tresolution}&yunits=0&date_choice={date_choice}&start_day={start_day}&start_month={start_month}&start_year={start_year}&start_hour={start_hour}&start_min={start_min}&end_day={end_day}&end_month={end_month}&end_year={end_year}&end_hour={end_hour}&end_min={end_min}&output=ascii'
    
    urllib.request.urlretrieve(data_url, f"{output_dir}/{station}_data_{i}.txt")


def get_mcmu_data(i, output_dir = '/Users/emmadavies/Library/CloudStorage/OneDrive-USNH/FDs_BDEs_complexity/WIND_CAT/'):
    table = pd.read_table(f'{output_dir}/MCMU_data_{i}.txt', header=168, sep=';', names=['Timestamp','rcorr_e'])
    table = table[:-1].reset_index(drop=True)
    table['Timestamp'] = pd.to_datetime(table['Timestamp'])
    df = table
    return df


def get_sopo_data(i, output_dir = '/Users/emmadavies/Library/CloudStorage/OneDrive-USNH/FDs_BDEs_complexity/WIND_CAT/'):
    table = pd.read_table(f'{output_dir}/SOPO_data_{i}.txt', header=168, sep=';', names=['Timestamp','rcorr_e'])
    table = table[:-1].reset_index(drop=True)
    table['Timestamp'] = pd.to_datetime(table['Timestamp'])
    df = table
    return df