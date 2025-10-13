
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


from functions_noaa import (
    get_dscovrpositions
)

from functions_ace import (
    get_acepos_frommag_range
)

from functions_general import load_path

from position_frame_transforms import (
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

    # calculate number of days since Jan 1 1998 to today
    start_date = datetime(1998, 1, 1)
    end_date = datetime.now()
    delta = end_date - start_date
    total_days = delta.days

    print(f"Assuming the ACE mission to start on Jan 1 1998, there should be a total of {total_days} days of data to download until today ({end_date.strftime('%Y-%m-%d')}).")

    while stopflag == False:
        try:
            for filetyp in types:
                filename = f"{filetyp}-{timespan}.{counter}.json"
                local_path = Path(rtsw_path + filetyp + timespan + "." + str(counter) + ".json")

                if local_path.exists():
                    print(f"File {filename} already exists. Skipping download.")
                    continue  # Skip to the next iteration if file exists

                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(
                    base_url + filetyp + "-" + timespan + "." + str(counter) + ".json",
                    rtsw_path + filetyp + timespan + "." + str(counter) + ".json",
                )

        except urllib.error.URLError as e:
            print(f"Error downloading RTSW {filetyp} data for day {counter}: {e}")
            error_count += 1
            print(type(e))
        else:
            # Reset error count if successful request
            error_count = 0
        finally:
            counter = counter + 1
            if counter >= total_days + 50:  # Adding a small buffer to ensure all days are covered
                stopflag = True
                print("Reached the estimated total number of days, stopping.")
            if error_count >= 500:
                stopflag = True
                print("Too many errors, stopping.")


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
    match = re.search(r'mag1-day\.(\d+)\.json', path.name)
    return int(match.group(1)) if match else float('inf')

def extract_index_plasma(path):
    # Extract numeric index from filename like 'plasma-1-day.0.json'
    match = re.search(r'plasma-1-day\.(\d+)\.json', path.name)
    return int(match.group(1)) if match else float('inf')

def create_rtsw_mag_pkl(output_path = rtsw_path, filesave_path = rtsw_path):
    out_path = Path(output_path)
    
    # find all matching files automatically
    mag_files = sorted(out_path.glob('mag1-day.*.json'), key=extract_index_mag)

    if not mag_files:
        raise FileNotFoundError(f"No mag1-day JSON files found in {output_path}")
    
    print(f"Found {len(mag_files)} mag1-day JSON files.")
    
    # Load mag data from all files
    mag_data = []

    for mag_file in tqdm(mag_files, desc="Loading mag data"):
        try:
            data = get_rtsw_data(mag_file)
            mag_data.extend(data)
        except Exception as e:
            print(f"Error processing {mag_file}: {e}")
            continue

    print("Loaded individual mag_files, now sorting and creating recarray.")
    
    mag_data.sort(key=lambda x: x['time_tag'])

    rtsw_mag = np.zeros(len(mag_data), dtype=[
        ("time", "O"),
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

    rtsw_mag = rtsw_mag.view(np.recarray)

    rtsw_mag.time = [entry["time_tag"] for entry in mag_data]
    rtsw_mag.bt = [entry["bt"] for entry in mag_data]
    rtsw_mag.bx = [entry["bx_gsm"] for entry in mag_data]
    rtsw_mag.by = [entry["by_gsm"] for entry in mag_data]
    rtsw_mag.bz = [entry["bz_gsm"] for entry in mag_data]
    rtsw_mag.lat_gsm = [entry["lat_gsm"] for entry in mag_data]
    rtsw_mag.lon_gsm = [entry["lon_gsm"] for entry in mag_data]
    rtsw_mag.quality = [entry["quality"] for entry in mag_data]
    rtsw_mag.source = [entry["source"] for entry in mag_data]
    rtsw_mag.active = [entry["active"] for entry in mag_data]


    header='Mag data from RTSW, called from http://services.swpc.noaa.gov/text/rtsw/data/'+\
    ' Timerange: '+ rtsw_mag.time[0]+' to '+rtsw_mag.time[-1]+'.'+\
    ', Units are btxyz [nT, GSM], lat/lon [deg,GSM], quality, source [0=ACE, 1=DSCOVR, 2=mixed], active [ 0-no, 1-yes].'+\
    ' Made with script by H.T. Ruedisser (github @hruedisser). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    
    # dump to pkl
    pickle.dump([rtsw_mag, header], open(Path(filesave_path) / 'rtsw_mag.p', 'wb'))



def create_rtsw_plasma_pkl(output_path = rtsw_path, filesave_path = rtsw_path):
    out_path = Path(output_path)
    
    # find all matching files automatically
    plasma_files = sorted(out_path.glob('plasma1-day.*.json'), key=extract_index_plasma)

    if not plasma_files:
        raise FileNotFoundError(f"No plasma1-day JSON files found in {output_path}")
    
    print(f"Found {len(plasma_files)} plasma1-day JSON files.")
    
    # Load plasma data from all files
    plasma_data = []

    for plasma_file in tqdm(plasma_files, desc="Loading plasma data"):
        try:
            data = get_rtsw_data(plasma_file)
            plasma_data.extend(data)
        except Exception as e:
            print(f"Error processing {plasma_file}: {e}")
            continue

    print("Loaded individual plasma_files, now sorting and creating recarray.")
    
    plasma_data.sort(key=lambda x: x['time_tag'])

    rtsw_plasma = np.zeros(len(plasma_data), dtype=[
        ("time", "O"),
        ("speed", float),
        ("density", float),
        ("temperature", float),
        ("quality", float),
        ("source", float),
        ("active", float),
    ])

    rtsw_plasma = rtsw_plasma.view(np.recarray)

    rtsw_plasma.time = [entry["time_tag"] for entry in plasma_data]
    rtsw_plasma.speed = [entry["speed"] for entry in plasma_data]
    rtsw_plasma.density = [entry["density"] for entry in plasma_data]
    rtsw_plasma.temperature = [entry["temperature"] for entry in plasma_data]
    rtsw_plasma.quality = [entry["quality"] for entry in plasma_data]
    rtsw_plasma.source = [entry["source"] for entry in plasma_data]
    rtsw_plasma.active = [entry["active"] for entry in plasma_data]


    header='Plasma data from RTSW, called from http://services.swpc.noaa.gov/text/rtsw/data/'+\
    ' Timerange: '+ rtsw_plasma.time[0]+' to '+rtsw_plasma.time[-1]+'.'+\
    ', Units are speed [km/s], density [1/cm^3], temperature [K], quality, source [0=ACE, 1=DSCOVR, 2=mixed], active [ 0-no, 1-yes].'+\
    ' Made with script by H.T. Ruedisser (github @hruedisser). File creation date: '+\
    datetime.now(timezone.utc).strftime("%Y-%b-%d %H:%M")+' UTC'
    
    # dump to pkl
    pickle.dump([rtsw_plasma, header], open(Path(filesave_path) / f'rtsw_plasma.p', 'wb'))


def create_rtsw_realtime_archive(output_path = rtsw_path, start_date = None, end_date = None):

    # load mag and plasma pkls
    out_path = Path(output_path)
    
    rtsw_mag = pickle.load(open(Path(output_path) / 'rtsw_mag.p', 'rb'))[0]
    print(f"Loaded rtsw_mag from {Path(output_path) / 'rtsw_mag.p'}")
    
    rtsw_plasma = pickle.load(open(Path(output_path) / 'rtsw_plasma.p', 'rb'))[0]
    print(f"Loaded rtsw_plasma from {Path(output_path) / 'rtsw_plasma.p'}")

    mag_df = pd.DataFrame(rtsw_mag)
    plasma_df = pd.DataFrame(rtsw_plasma)

    # Convert time columns to datetime
    mag_df["time"] = pd.to_datetime(mag_df["time"])
    plasma_df["time"] = pd.to_datetime(plasma_df["time"])

    if start_date is not None:
        mag_df = mag_df[mag_df['time'] >= start_date]
        plasma_df = plasma_df[plasma_df['time'] >= start_date]
        print(f"Filtered data from {start_date} onwards.")
    if end_date is not None:
        mag_df = mag_df[mag_df['time'] <= end_date]
        plasma_df = plasma_df[plasma_df['time'] <= end_date]
        print(f"Filtered data up to {end_date}.")
    
    print("Merging mag and plasma dataframes...")
    combined_df = pd.merge(
        mag_df, plasma_df, on="time", how="outer", suffixes=('_mag', '_plasma')
    )

    combined_df = combined_df[
        [
            "time",
            "bt",
            "bx",
            "by",
            "bz",
            "lat",
            "lon",
            "source_mag",
            "source_plasma",
            "speed",
            "density",
            "temperature"
        ]
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
            ("vx", float),
            ("vy", float),
            ("vz", float),
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
    rtsw.time = combined_df["time"]
    rtsw.bx = combined_df["bx"].values
    rtsw.by = combined_df["by"].values
    rtsw.bz = combined_df["bz"].values
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
        
    start_timestamp = rtsw.time[0]
    end_timestamp = rtsw.time[-1]

    print(f"Getting DSCOVR and ACE positions from {start_timestamp} to {end_timestamp}...")

    # Get DSCOVR positions directly in GSE
    dscovr_positions = get_dscovrpositions(start_timestamp, end_timestamp, coord_sys="GSE")

    dscovr_positions =  dscovr_positions.set_index('time')

    full_dscovr_index = pd.date_range(
        dscovr_positions.index.min(),
        dscovr_positions.index.max(),
        freq='1min'
    )

    dscovr_positions = pd.merge_asof(full_dscovr_index.to_frame(name='time'), dscovr_positions, on='time', direction='nearest', tolerance=pd.Timedelta('30s'))
    
    # dscovr_positions are only available in 1 day resolution, so we need to linearly interpolate to 1 min resolution
    dscovr_positions = dscovr_positions.set_index('time').interpolate(method='time').reset_index()
    

    # Transform to HEE and then to HEEQ in-place style
    print("Transforming DSCOVR positions (GSE → HEE → HEEQ)...")
    dscovr_positions = GSE_to_HEE(dscovr_positions)
    dscovr_positions = HEE_to_HEEQ(dscovr_positions)

    dscovr_positions = dscovr_positions.set_index("time")

    # Get ACE positions directly in GSE
    ace_positions = get_acepos_frommag_range(start_timestamp, end_timestamp, coord_sys="GSE")

    ace_positions = ace_positions.sort_values("time").set_index('time')

    full_ace_index = pd.date_range(
        ace_positions.index.min(),
        ace_positions.index.max(),
        freq='1min'
    )

    ace_positions = pd.merge_asof(full_ace_index.to_frame(name='time'), ace_positions, on='time', direction='nearest', tolerance=pd.Timedelta('30s'))

    # Transform to HEE and then to HEEQ in-place style
    print("Transforming ACE positions (GSE → HEE → HEEQ)...")
    ace_positions = GSE_to_HEE(ace_positions)
    ace_positions = HEE_to_HEEQ(ace_positions)

    ace_positions = ace_positions.set_index("time")

    positions_HEEQ = np.array(
        [
            get_positions(time, source, dscovr_positions, ace_positions)
            for time, source in zip(combined_df.time, combined_df.source_mag)
        ]
    )

    rtsw.x = positions_HEEQ[:, 0]
    rtsw.y = positions_HEEQ[:, 1]
    rtsw.z = positions_HEEQ[:, 2]
    rtsw.r = positions_HEEQ[:, 3]
    rtsw.lat = positions_HEEQ[:, 4]
    rtsw.lon = positions_HEEQ[:, 5]

    header = (
        f"Real time solar wind magnetic field and plasma data from NOAA http://services.swpc.noaa.gov/text/rtsw/data/."
        f" Position data from DSCOVR, called from https://www.ngdc.noaa.gov/dscovr-data-access/ or https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download."
        f" Orbit data from ACE, sourced from mag or plasma data files from https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0/ or https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/."
        f" Timerange: {rtsw.time[0]:%Y-%m-%d %H:%M} to {rtsw.time[-1]:%Y-%m-%d %H:%M}."
        f" Units: bxyz [nT, GSM], np [1/cm^3], vt [km/s], tp [K], source_mag [0=ACE, 1=DSCOVR, 2=mixed], source_plasma [0=ACE, 1=DSCOVR, 2=mixed], xyz [AU, HEEQ], lat/lon [deg, HEEQ]."
        f" Made with script by H.T. Ruedisser (github @hruedisser). File creation date: {datetime.now(timezone.utc):%Y-%b-%d %H:%M} UTC"
    )
    
    # dump to pkl
    pickle.dump([rtsw, header], open(Path(output_path) / f'rtsw_realtime_archive_gsm.p', 'wb'))

    print(f"Saved rtsw_realtime_archive_gsm to {Path(output_path) / 'rtsw_realtime_archive_gsm.p'}")





