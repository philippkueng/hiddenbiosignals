# this trims the wav files in a folder based on the "went to bed" timestamp

import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from pydub import AudioSegment

# Directories
excel_dir = "sleep_labels"
wav_dir = "Audio_Data/Jasper"
trimmed_wav_dir = "trimmed_wavs_jasper"

# Ensure output directory exists
os.makedirs(trimmed_wav_dir, exist_ok=True)

# Function to extract timestamp from WAV filename
def extract_wav_timestamp(filename):
    match = re.search(r"Jasper_142Hz_(\d+)\.wav$", filename)
    if match:
        timestamp_ms = int(match.group(1))  # Extract timestamp in milliseconds
        timestamp_s = timestamp_ms / 1000   # Convert to seconds
        return int(timestamp_s)  # Return as integer (UNIX timestamp in GMT)
    return None

# Load all WAV files and extract timestamps
wav_files = {}
for filename in os.listdir(wav_dir):
    if filename.endswith(".wav"):
        timestamp = extract_wav_timestamp(filename)
        if timestamp is not None:
            wav_files[timestamp] = filename  # Store in dictionary: {timestamp: filename}

# Function to convert "Went to bed" timestamp from German time to UNIX (GMT)
def convert_to_unix_gmt(timestamp):
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    elif isinstance(timestamp, str):
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    german_tz = timezone(timedelta(hours=1))  # Adjust for daylight savings if needed
    timestamp = timestamp.replace(tzinfo=german_tz)
    timestamp_gmt = timestamp.astimezone(timezone.utc)
    return int(timestamp_gmt.timestamp())

# Process each Excel file
for excel_file in os.listdir(excel_dir):
    if not excel_file.endswith(".xlsx"):
        print(f"Skipping file: {excel_file}")
        continue  # Skip already processed or non-Excel files

    file_path = os.path.join(excel_dir, excel_file)
    df = pd.read_excel(file_path, engine="openpyxl")

    # Convert "Went to bed" to UNIX timestamp in GMT
    df["Unix Went to bed (GMT)"] = df["Went to bed"].apply(convert_to_unix_gmt)

    # Process each row to trim the corresponding WAV file
    for _, row in df.iterrows():
        went_to_bed_unix = row["Unix Went to bed (GMT)"]

        if wav_files:
            # Find the closest WAV file
            closest_wav_start_time = min(wav_files.keys(), key=lambda t: abs(t - went_to_bed_unix))
            wav_file_name = wav_files[closest_wav_start_time]
            wav_file_path = os.path.join(wav_dir, wav_file_name)

            # Calculate trim start time
            trim_start_time = went_to_bed_unix - closest_wav_start_time  # Time in seconds

            if trim_start_time > 0:
                # Load the WAV file
                audio = AudioSegment.from_wav(wav_file_path)

                # Convert to milliseconds and trim
                trim_start_ms = trim_start_time * 1000
                trimmed_audio = audio[trim_start_ms:]

                # Save the trimmed WAV file
                trimmed_wav_path = os.path.join(trimmed_wav_dir, f"trimmed_{wav_file_name}")
                trimmed_audio.export(trimmed_wav_path, format="wav")
                print(f"Trimmed and saved: {trimmed_wav_path}")

            else:
                print(f"Skipping trimming for {wav_file_name} (Went to bed timestamp is before WAV recording start)")

print("Processing complete.")

