
from datetime import timedelta, timezone, datetime
import requests
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import urllib
import urllib.error
from collections import defaultdict
import re

import os
import json
import pickle


from .functions_noaa import (
    get_dscovrpositions
)

from .functions_ace import (
    get_acepos_frommag_range
)

from .functions_general import load_path

from .position_frame_transforms import (
    GSE_to_HEE,
    HEE_to_HEEQ
)


"""
NOAA/DSCOVR and RTSW DATA PATH
"""


dscovr_path=load_path(path_name='dscovr_path')
print(f"DSCOVR path loaded: {dscovr_path}")

rtsw_path=load_path(path_name='rtsw_path')
print(f"RTSW path loaded: {rtsw_path}")

ace_path=load_path(path_name='ace_path')
print(f"ACE path loaded: {ace_path}")

# Load path once globally
kernels_path = load_path(path_name='kernels_path')
print(f"Kernels path loaded: {kernels_path}")


"""
RTSW DOWNLOAD FUNCTIONS
"""


def download_rtsw(types = ["mag", "plasma", "kp"], timespan = "1-day", base_url = "http://services.swpc.noaa.gov/text/rtsw/data/"):

    os.makedirs(rtsw_path, exist_ok=True)

    error_count = 0
    stopflag = False
    counter = 0

    while stopflag == False:
        try:
            for filetyp in types:
                filename = f"{filetyp}-{timespan}.{counter}.json"
                local_path = Path(rtsw_path + "/" + filetyp + timespan + "." + str(counter) + ".json")

                if local_path.exists():
                    print(f"File {filename} already exists. Skipping download.")
                    continue  # Skip to the next iteration if file exists

                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(
                    base_url + filetyp + "-" + timespan + "." + str(counter) + ".json",
                    rtsw_path + "/" + filetyp + timespan + "." + str(counter) + ".json",
                )

        except urllib.error.URLError as e:
            print(f"Error downloading RTSW: {e.reason}")
            error_count += 1
            print(type(e))
        else:
            # Reset error count if successful request
            error_count = 0
        finally:
            counter = counter + 1
            if counter >= 100000:
                stopflag = True


def get_rtsw_data(json_file):
    data = defaultdict(dict)
    with open(json_file, "r") as jdata:
        dp = json.load(jdata)
        columns = dp[0]  # Extract column names
        for entry in dp[1:]:  # Skip the first entry which contains column names
            entry_dict = dict(
                zip(columns, entry)
            )  # Map each entry to its corresponding column name
            time_tag = entry_dict.get("time_tag")
            active = int(entry_dict.get("active"))
            if time_tag is not None and active is not None:
                for key, value in entry_dict.items():
                    if key != "time_tag":
                        entry_dict[key] = float(value) if value is not None else np.nan
                # print(active)
                if (
                    time_tag not in data and active == 1
                ):  # or float(source) > float(data[time_tag].get('source', float('-inf'))):
                    data[time_tag] = entry_dict

    return list(data.values())


"""
RTSW DATA SAVING FUNCTIONS:
"""

# Use vectorized lookup for positions
def get_positions(time, source_mag, df_dscovr_pos, df_ace_pos):
    if source_mag == 1.0:
        return (
            df_dscovr_pos.loc[time] if time in df_dscovr_pos.index else [np.nan] * 6
        )
    else:
        return df_ace_pos.loc[time] if time in df_ace_pos.index else [np.nan] * 6

def extract_index_mag(path):
    # Extract numeric index from filename like 'mag-1-day.0.json'
    match = re.search(r'mag-1-day\.(\d+)\.json', path.name)
    return int(match.group(1)) if match else float('inf')

def extract_index_plasma(path):
    # Extract numeric index from filename like 'plasma-1-day.0.json'
    match = re.search(r'plasma-1-day\.(\d+)\.json', path.name)
    return int(match.group(1)) if match else float('inf')

def create_rtsw_mag_pkl(output_path = rtsw_path):
    out_path = Path(output_path)
    
    # find all matching files automatically
    mag_files = sorted(out_path.glob('mag-1-day.*.json'), key=extract_index_mag)

    if not mag_files:
        raise FileNotFoundError(f"No mag-1-day JSON files found in {output_path}")
    
    print(f"Found {len(mag_files)} mag-1-day JSON files.")
    
    # Load mag data from all files
    mag_data = []

    for mag_file in tqdm(mag_files, desc="Loading mag data"):
        try:
            data = get_rtsw_data(mag_file)
            mag_data.extend(data)
        except Exception as e:
            print(f"Error processing {mag_file}: {e}")
            continue
    
    mag_data.sort(key=lambda x: x['time_tag'])

    rtsw_mag = np.zeros(len(mag_data), dtype=[
        ("time_tag", "O"),
        ("bt", float),
        ("bx", float),
        ("by", float),
        ("bz", float),
        ("lat", float),
        ("lon", float),
        ("quality", float),
        ("source", float),
        ("active", float),
    ])

    rtsw_mag.view(np.recarray)

    rtsw_mag.time = [entry["time_tag"] for entry in mag_data]
    rtsw_mag.bt = [entry["bt"] for entry in mag_data]
    rtsw_mag.bx = [entry["bx"] for entry in mag_data]
    rtsw_mag.by = [entry["by"] for entry in mag_data]
    rtsw_mag.bz = [entry["bz"] for entry in mag_data]
    rtsw_mag.lat_gsm = [entry["lat"] for entry in mag_data]
    rtsw_mag.lon_gsm = [entry["lon"] for entry in mag_data]
    rtsw_mag.quality = [entry["quality"] for entry in mag_data]
    rtsw_mag.source = [entry["source"] for entry in mag_data]
    rtsw_mag.active = [entry["active"] for entry in mag_data]


    header='Mag data from RTSW, called from http://services.swpc.noaa.gov/text/rtsw/data/'+\
    ' Timerange: '+ rtsw_mag.time[0]+' to '+rtsw_mag.time[-1]+'.'+\
    ', Units are btxyz [nT, GSM], lat/lon [deg,GSM], quality, source [0=ACE, 1=DSCOVR, 2=mixed], active [ 0-no, 1-yes].'+\
    ' Made with script by H.T. Ruedisser (github @hruedisser). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    
    # dump to pkl
    pickle.dump([rtsw_mag, header], open(Path(out_path) / 'rtsw_mag.p', 'wb'))



def create_rtsw_plasma_pkl(output_path = rtsw_path):
    out_path = Path(output_path)
    
    # find all matching files automatically
    plasma_files = sorted(out_path.glob('plasma-1-day.*.json'), key=extract_index_plasma)

    if not plasma_files:
        raise FileNotFoundError(f"No plasma-1-day JSON files found in {output_path}")
    
    print(f"Found {len(plasma_files)} plasma-1-day JSON files.")
    
    # Load plasma data from all files
    plasma_data = []

    for plasma_file in tqdm(plasma_files, desc="Loading plasma data"):
        try:
            data = get_rtsw_data(plasma_file)
            plasma_data.extend(data)
        except Exception as e:
            print(f"Error processing {plasma_file}: {e}")
            continue
    
    plasma_data.sort(key=lambda x: x['time_tag'])

    rtsw_plasma = np.zeros(len(plasma_data), dtype=[
        ("time_tag", "O"),
        ("speed", float),
        ("density", float),
        ("temperature", float),
        ("quality", float),
        ("source", float),
        ("active", float),
    ])

    rtsw_plasma.view(np.recarray)

    rtsw_plasma.time = [entry["time_tag"] for entry in rtsw_plasma]
    rtsw_plasma.speed = [entry["speed"] for entry in rtsw_plasma]
    rtsw_plasma.density = [entry["density"] for entry in rtsw_plasma]
    rtsw_plasma.temperature = [entry["temperature"] for entry in rtsw_plasma]
    rtsw_plasma.quality = [entry["quality"] for entry in rtsw_plasma]
    rtsw_plasma.source = [entry["source"] for entry in rtsw_plasma]
    rtsw_plasma.active = [entry["active"] for entry in rtsw_plasma]


    header='Plasma data from RTSW, called from http://services.swpc.noaa.gov/text/rtsw/data/'+\
    ' Timerange: '+ rtsw_plasma.time[0]+' to '+rtsw_plasma.time[-1]+'.'+\
    ', Units are speed [km/s], density [1/cm^3], temperature [K], quality, source [0=ACE, 1=DSCOVR, 2=mixed], active [ 0-no, 1-yes].'+\
    ' Made with script by H.T. Ruedisser (github @hruedisser). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    
    # dump to pkl
    pickle.dump([rtsw_plasma, header], open(Path(out_path) / f'rtsw_plasma.p', 'wb'))


def create_rtsw_realtime_archive(output_path = rtsw_path):

    # load mag and plasma pkls
    out_path = Path(output_path)
    rtsw_mag = pickle.load(open(Path(out_path) / 'rtsw_mag.p', 'rb'))[0]
    rtsw_plasma = pickle.load(open(Path(out_path) / 'rtsw_plasma.p', 'rb'))[0]

    mag_df = pd.DataFrame(rtsw_mag)
    plasma_df = pd.DataFrame(rtsw_plasma)

    combined_df = pd.merge(
        mag_df, plasma_df, on="time_tag", how="outer", suffixes=('_mag', '_plasma')
    )

    combined_df = combined_df[
        [
            "time_tag",
            "bt",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "lat_gsm",
            "lon_gsm",
            "source_mag",
            "speed",
            "density",
            "temperature",
            "source_plasma"
        ]
    ]

    datetime_objects = [
        datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in combined_df["time_tag"]
    ]

    # Make array

    rtsw = np.zeros(
        len(combined_df),
        dtype=[
            ("time", object),
            ("bx", float),
            ("by", float),
            ("bz", float),
            ("bt", float),
            ("np", float),
            ("vt", float),
            ("tp", float),
            ("source_mag", float),
            ("source_plasma", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("r", float),
            ("lat", float),
            ("lon", float),
        ]
    )

    rtsw = rtsw.view(np.recarray)

    # Fill with data
    rtsw.time = datetime_objects
    rtsw.bx = combined_df["bx_gsm"].values
    rtsw.by = combined_df["by_gsm"].values
    rtsw.bz = combined_df["bz_gsm"].values
    rtsw.bt = combined_df["bt"].values
    rtsw.np = combined_df["density"].values
    rtsw.vt = combined_df["speed"].values
    rtsw.tp = combined_df["temperature"].values
    rtsw.source_mag = combined_df["source_mag"].values
    rtsw.source_plasma = combined_df["source_plasma"].values

    if np.any(rtsw.source_mag == 0.0):
        print("ACE data found")
    else:
        print("No ACE data")

    if np.any(rtsw.source_mag == 1.0):
        print("DSCOVR data found")
    else:
        print("No DSCOVR data")

    if np.any(rtsw.source_mag == 2.0):
        print("Mixed data found")
    else:
        print("No mixed data")
        
    start_timestamp = rtsw.time[0].strftime("%Y-%m-%d %H:%M:%S")
    end_timestamp = rtsw.time[-1].strftime("%Y-%m-%d %H:%M:%S")

    # Get DSCOVR positions for the time range in GSE
    dscovr_positions = get_dscovrpositions(start_timestamp, end_timestamp, coord_sys="GSE")

    # Get ACE positions for the time range in GSE
    ace_positions = get_acepos_frommag_range(start_timestamp, end_timestamp, coord_sys="GSE")

    # Apply position transforms
    dscovr_pos_HEE = GSE_to_HEE(dscovr_positions)
    dscovr_pos_HEEQ = HEE_to_HEEQ(dscovr_pos_HEE)

    ace_pos_HEE = GSE_to_HEE(ace_positions)
    ace_pos_HEEQ = HEE_to_HEEQ(ace_pos_HEE)

    # Set index and resample

    dscovr_pos = (dscovr_pos_HEEQ.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False))

    ace_pos = (ace_pos_HEEQ.set_index('time').resample('1min').interpolate(method='linear').reset_index(drop=False))

    positions_HEEQ = np.array(
        [
            get_positions(time, source)
            for time, source in zip(rtsw.time, rtsw.source_mag, dscovr_pos, ace_pos)
        ]
    )

    rtsw.x = positions_HEEQ[:, 0]
    rtsw.y = positions_HEEQ[:, 1]
    rtsw.z = positions_HEEQ[:, 2]
    rtsw.r = positions_HEEQ[:, 3]
    rtsw.lat = positions_HEEQ[:, 4]
    rtsw.lon = positions_HEEQ[:, 5]

    header='Real time solar wind magnetic field and plasma data from NOAA http://services.swpc.noaa.gov/text/rtsw/data/.'+\
    ' Position data from DSCOVR, called from https://www.ngdc.noaa.gov/dscovr-data-access/ or https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download.' +\
    ' Orbit data from ACE, sourced from mag or plasma data files from https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/ or https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/.'+\
    ' Timerange: '+ rtsw.time[0]+' to '+rtsw.time[-1]+'.'+\
    ', Units: bxyz [nT, GSM], np [1/cm^3], vt [km/s], tp [K], source_mag [0=ACE, 1=DSCOVR, 2=mixed], source_plasma [0=ACE, 1=DSCOVR, 2=mixed], xyz [AU, HEEQ], lat/lon [deg, HEEQ].'+\
    ' Made with script by H.T. Ruedisser (github @hruedisser). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    
    # dump to pkl
    pickle.dump([rtsw, header], open(Path(out_path) / f'rtsw_realtime_archive.p', 'wb'))





