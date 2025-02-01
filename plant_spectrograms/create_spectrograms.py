aimport librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def analyze_emotion_spectrograms_with_timestamps(wav_file, excel_file, output_dir, wav_start_time_str, sr=142, n_fft=256, interval_seconds=5):
    """
    Creates spectrograms from a WAV file and labels them with the strongest emotion based on aligned timestamps.
    
    Args:
        wav_file: Path to the input WAV file.
        excel_file: Path to Excel file with columns for emotions and 'timestamp' (as datetime).
        output_dir: Directory to save the spectrograms.
        wav_start_time_str: Start time of the WAV file in '%Y-%m-%d %H:%M:%S.%f' format.
        sr: Sample rate (default: 142 Hz).
        n_fft: Size of the FFT window (default: 256).
        interval_seconds: Time interval between spectrograms (default: 5 seconds).
    """
    # Load the WAV file
    y, _ = librosa.load(wav_file, sr=sr)
    
    # Load the emotion data
    emotion_df = pd.read_excel(excel_file)
    emotion_columns = ['happy', 'surprised', 'neutral', 'sad', 'angry', 'disgusted', 'fearful']
    
    # Calculate the strongest emotion for each row
    emotion_df['strongest_emotion'] = emotion_df[emotion_columns].idxmax(axis=1)
    emotion_df['timestamp'] = pd.to_datetime(emotion_df['timestamp'])
    
    # Convert WAV start time to datetime
    wav_start_time = datetime.strptime(wav_start_time_str, "%Y-%m-%d %H:%M:%S.%f")
    
    # Calculate the number of frames
    duration = librosa.get_duration(y=y, sr=sr)
    num_frames = int(duration // interval_seconds)
    
    # Calculate frame parameters
    samples_per_frame = sr * interval_seconds
    hop_length = n_fft // 4
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate spectrograms for each frame
    name_counter_offset = 2745
    for i in range(num_frames):
        # Calculate the current time in the WAV file
        current_time = i * interval_seconds
        current_wav_time = wav_start_time + timedelta(seconds=current_time)
        
        # Find the strongest emotion for the current WAV time
        emotion_row = emotion_df[emotion_df['timestamp'] <= current_wav_time].iloc[-1] \
            if not emotion_df[emotion_df['timestamp'] <= current_wav_time].empty else None
        
        current_emotion = emotion_row['strongest_emotion'] if emotion_row is not None else "unknown"
        
        # Extract the audio segment
        start_sample = i * samples_per_frame
        end_sample = min((i + 1) * samples_per_frame, len(y))
        y_frame = y[start_sample:end_sample]
        
        # Compute the melspectrogram
        S = librosa.feature.melspectrogram(
            y=y_frame,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=64
        )
        
        # Convert to log scale
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Add spectrogram
        plt.subplot(1, 1, 1)
        librosa.display.specshow(
            log_S,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Time: {current_wav_time} - Strongest Emotion: {current_emotion}")
        
        # Save the plot
        output_file = os.path.join(output_dir, f"spectrogram_{current_time+name_counter_offset:04d}s_{current_emotion}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Generated spectrogram {i+1}/{num_frames} - Time: {current_wav_time} - Emotion: {current_emotion}")

if __name__ == "__main__":
    # File paths
    wav_file = "Bananenpalme-jan19-2_142Hz_1737283669461.wav"
    excel_file = "output_with_strongest_emotion-jan19.xlsx"
    output_dir = "emotion_spectrograms_jan19"
    wav_start_time = "2025-01-19 11:47:49.461000"  # Start time of the WAV file
    
    try:
        analyze_emotion_spectrograms_with_timestamps(wav_file, excel_file, output_dir, wav_start_time)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


