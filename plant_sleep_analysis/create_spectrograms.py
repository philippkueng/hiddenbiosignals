# this creates all spectrograms from the awake and asleep wav files, one image per minute, and stores them in a folder, name appropriately for further training

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Directories
wavs_awake_dir = "wavs_awake_jasper"
asleep_30min_dir = "30mins_asleep_jasper"
spectrogram_dir = "spectrograms_jasper"
os.makedirs(spectrogram_dir, exist_ok=True)

# Spectrogram Parameters
SR = 142  # Sample rate
N_MELS = 128
N_FFT = 256
HOP_LENGTH = 128
SEGMENT_DURATION = 60  # Each segment = 1 min

# **Function to Extract Spectrograms per Minute**
def extract_spectrograms(wav_path, label, output_dir):
    y, sr = librosa.load(wav_path, sr=SR)
    duration = librosa.get_duration(y=y, sr=sr)

    spectrograms = []
    labels = []

    for i in range(0, int(duration), SEGMENT_DURATION):
        start_sample = i * sr
        end_sample = min((i + SEGMENT_DURATION) * sr, len(y))

        # Extract 1-minute segment
        segment = y[start_sample:end_sample]
        if len(segment) < sr * SEGMENT_DURATION:
            continue  # Skip short segments

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels

        # Save as Image
        spectrogram_file = f"{output_dir}/spec_{os.path.basename(wav_path).replace('.wav', '')}_{i}.png"
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH, cmap='magma')
        plt.axis("off")
        plt.savefig(spectrogram_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        spectrograms.append(spectrogram_file)
        labels.append(label)

    return spectrograms, labels

# **Process WAV Files**
spectrograms = []
labels = []

# Awake (Label = 0)
for file in os.listdir(wavs_awake_dir):
    if file.endswith(".wav"):
        path = os.path.join(wavs_awake_dir, file)
        awake_specs, awake_labels = extract_spectrograms(path, label=0, output_dir=spectrogram_dir)
        spectrograms.extend(awake_specs)
        labels.extend(awake_labels)

# Asleep (Label = 1)
for file in os.listdir(asleep_30min_dir):
    if file.endswith(".wav"):
        path = os.path.join(asleep_30min_dir, file)
        asleep_specs, asleep_labels = extract_spectrograms(path, label=1, output_dir=spectrogram_dir)
        spectrograms.extend(asleep_specs)
        labels.extend(asleep_labels)

print(f"Total spectrograms generated: {len(spectrograms)}")

