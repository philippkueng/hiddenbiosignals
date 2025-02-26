# extracts and aligns timestamps from Unix timestamp in the filename of the wav file recorded and saved with the Oxocard and stored in a folder per user
# this is then aligned with the went to bed timestamp obtained from the smartphone sleep tracking app storeld in an Excel file

import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta

# Directories
excel_dir = "sleep_labels"
wav_dir = "Audio_Data/Jasper"

# Function to extract and convert timestamp from WAV filename
def extract_wav_timestamp(filename):
    match = re.search(r"Jasper_142Hz_(\d+)\.wav$", filename)
    if match:
        timestamp_ms = int(match.group(1))  # Extract timestamp in milliseconds
        timestamp_s = timestamp_ms / 1000   # Convert to seconds
        return int(timestamp_s)  # Return as integer (UNIX timestamp in GMT)
    return None

# Load all WAV files and extract their timestamps
wav_files = {}
for filename in os.listdir(wav_dir):
    if filename.endswith(".wav"):
        timestamp = extract_wav_timestamp(filename)
        if timestamp is not None:
            wav_files[timestamp] = filename  # Store in dict: {timestamp: filename}

# Function to convert "Went to bed" timestamp from German time to UNIX (GMT)
def convert_to_unix_gmt(timestamp):
    if isinstance(timestamp, pd.Timestamp):  # If already a Timestamp object
        timestamp = timestamp.to_pydatetime()  # Convert to datetime object
    elif isinstance(timestamp, str):  # If it's a string, parse it
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    # German time is UTC+1 (Winter) or UTC+2 (Summer). Assume UTC+1 for now.
    german_tz = timezone(timedelta(hours=1))  # Adjust if needed
    timestamp = timestamp.replace(tzinfo=german_tz)  # Set German time zone
    timestamp_gmt = timestamp.astimezone(timezone.utc)  # Convert to GMT/UTC
    return int(timestamp_gmt.timestamp())  # Return UNIX timestamp

# Max allowed difference (1 hour = 3600 seconds)
MAX_TIME_DIFF = 3600

# Process each Excel file
for excel_file in os.listdir(excel_dir):
    if excel_file.startswith("updated_") or not excel_file.endswith(".xlsx"):
        print(f"Skipping file: {excel_file}")
        continue  # Skip already processed or non-Excel files
    # Skip hidden/system files (like .DS_Store)
    if excel_file.startswith("."):
        print(f"Skipping hidden file: {file_path}")
        continue
        
    file_path = os.path.join(excel_dir, excel_file)
    df = pd.read_excel(file_path, engine="openpyxl")

    # Convert "Went to bed" to UNIX timestamp in GMT
    df["Unix Went to bed (GMT)"] = df["Went to bed"].apply(convert_to_unix_gmt)

    # Find closest matching WAV file (if within 1 hour)
    matched_wav_files = []
    wav_start_times = []
    
    for _, row in df.iterrows():
        went_to_bed_unix = row["Unix Went to bed (GMT)"]

        if wav_files:
            # Find the closest WAV file timestamp
            closest_wav_start_time = min(wav_files.keys(), key=lambda t: abs(t - went_to_bed_unix))
            time_diff = abs(closest_wav_start_time - went_to_bed_unix)

            # Only add the WAV file if the difference is <= 1 hour
            if time_diff <= MAX_TIME_DIFF:
                matched_wav_files.append(wav_files[closest_wav_start_time])
                formatted_wav_time = datetime.utcfromtimestamp(closest_wav_start_time).strftime("%Y-%m-%d %H:%M:%S")
                wav_start_times.append(formatted_wav_time)  # Add formatted timestamp
            else:
                matched_wav_files.append(None)
                wav_start_times.append(None)  # No close match found
        else:
            matched_wav_files.append(None)
            wav_start_times.append(None)

    # Add matched WAV filename and formatted timestamp as new columns
    df["Matched WAV File"] = matched_wav_files
    df["wav_recording_started"] = wav_start_times

    # Save the updated Excel file
    updated_file_path = os.path.join(excel_dir, f"updated_{excel_file}")
    df.to_excel(updated_file_path, index=False)
    print(f"Processed and saved: {updated_file_path}")

