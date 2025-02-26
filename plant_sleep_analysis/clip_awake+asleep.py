# this clips the trimmed wav files into two segments awake and asleep, for asleep the first 30 minutes of sleep are taken

import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from pydub import AudioSegment

# Directories
excel_dir = "sleep_labels"
trimmed_wav_dir = "trimmed_wavs_jasper"
wavs_awake_dir = "wavs_awake_jasper"  # First part until "Asleep after (seconds)"
asleep_30min_dir = "30mins_asleep_jasper"  # 30 minutes after "Asleep after (seconds)"

# Ensure output directories exist
os.makedirs(wavs_awake_dir, exist_ok=True)
os.makedirs(asleep_30min_dir, exist_ok=True)

# Process each Excel file
for excel_file in os.listdir(excel_dir):
    if not excel_file.endswith(".xlsx"):
        print(f"Skipping file: {excel_file}")
        continue  # Skip already processed or non-Excel files

    file_path = os.path.join(excel_dir, excel_file)
    df = pd.read_excel(file_path, engine="openpyxl")

    # Process each row to create awake and asleep segments
    for _, row in df.iterrows():
        asleep_after_seconds = row["Asleep after (seconds)"]
        matched_wav_file = row["Matched WAV File"]  # Get the original filename

        if pd.isna(matched_wav_file) or not isinstance(matched_wav_file, str):
            print(f"Skipping row - No matched WAV file found")
            continue

        # Construct the expected trimmed WAV filename
        trimmed_wav_name = f"trimmed_{matched_wav_file}"
        trimmed_wav_path = os.path.join(trimmed_wav_dir, trimmed_wav_name)

        # Ensure the trimmed WAV file exists
        if not os.path.exists(trimmed_wav_path):
            print(f"Trimmed WAV not found: {trimmed_wav_path}")
            continue

        # Load the trimmed WAV file
        audio = AudioSegment.from_wav(trimmed_wav_path)

        # Convert asleep duration to milliseconds
        asleep_after_ms = asleep_after_seconds * 1000
        asleep_30min_ms = 30 * 60 * 1000  # 30 minutes in milliseconds

        # Extract the UNIX timestamp from the filename for naming
        match = re.search(r"Jasper_142Hz_(\d+)\.wav$", matched_wav_file)
        if not match:
            print(f"Skipping {matched_wav_file} - Unable to extract timestamp")
            continue

        original_timestamp = int(match.group(1))  # Extracted Unix timestamp (milliseconds)
        asleep_after_unix = original_timestamp + asleep_after_seconds * 1000  # Convert to UNIX ms

        # Format timestamps for filenames
        original_time_str = datetime.utcfromtimestamp(original_timestamp / 1000).strftime("%Y-%m-%d_%H-%M-%S")
        asleep_after_str = datetime.utcfromtimestamp(asleep_after_unix / 1000).strftime("%Y-%m-%d_%H-%M-%S")

        # **1️⃣ Create Wavs Awake (before sleep)**
        clipped_awake_audio = audio[:asleep_after_ms]
        clipped_awake_name = f"awake_Jasper_142Hz_{original_time_str}_until_{asleep_after_str}.wav"
        clipped_awake_path = os.path.join(wavs_awake_dir, clipped_awake_name)
        clipped_awake_audio.export(clipped_awake_path, format="wav")
        print(f"Saved awake segment: {clipped_awake_path}")

        # **2️⃣ Create 30mins Asleep (after sleep)**
        clipped_asleep_audio = audio[asleep_after_ms:asleep_after_ms + asleep_30min_ms]
        clipped_asleep_name = f"asleep_Jasper_142Hz_{asleep_after_str}_30min.wav"
        clipped_asleep_path = os.path.join(asleep_30min_dir, clipped_asleep_name)
        clipped_asleep_audio.export(clipped_asleep_path, format="wav")
        print(f"Saved 30-min asleep segment: {clipped_asleep_path}")

print("Processing complete.")

