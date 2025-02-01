import os
import time
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from flask import Flask, render_template_string, send_from_directory, jsonify

# Settings
DIRECTORY = "oxo_data"  # Directory to monitor for new .wav files
MODEL_PATH = "emotion_classifier_model.keras"  # Path to your pre-trained model
PREDICTION_INTERVAL = 5  # Time interval (in seconds) to check for new files
PROCESSED_FILES = set()  # Keep track of processed files
IMAGE_FOLDER = "oxo_spectral_images"  # Folder for spectrogram images

# Flask app
app = Flask(__name__)
LATEST_PREDICTION = {"image_path": None, "emotion": None, "confidence": None, "timestamp": None}

def extract_creation_time(wav_file):
    try:
        timestamp = int(os.path.basename(wav_file).split("_")[1].split(".")[0])
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    except (IndexError, ValueError):
        return "Unknown Time"

def generate_mel_spectrogram(wav_file, output_image, sr=142, n_fft=256, n_mels=64):
    y, _ = librosa.load(wav_file, sr=sr)
    hop_length = n_fft // 4
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    output_image_path = os.path.join(IMAGE_FOLDER, os.path.basename(output_image))
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(log_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return output_image_path

def preprocess_spectrogram(image_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_emotion(model, spectrogram_image):
    spectrogram_input = preprocess_spectrogram(spectrogram_image)
    predictions = model.predict(spectrogram_input)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_label, confidence

def monitor_and_predict():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load('label_classes.npy', allow_pickle=True)

    while True:
        wav_files = [f for f in os.listdir(DIRECTORY) if f.endswith(".wav")]
        for wav_file in wav_files:
            if wav_file not in PROCESSED_FILES:
                try:
                    full_path = os.path.join(DIRECTORY, wav_file)
                    creation_time = extract_creation_time(full_path)
                    spectrogram_image_path = generate_mel_spectrogram(full_path, f"{os.path.splitext(wav_file)[0]}_spectrogram.png")
                    predicted_label, confidence = predict_emotion(model, spectrogram_image_path)
                    predicted_emotion = class_names[predicted_label]

                    LATEST_PREDICTION.update({
                        "image_path": spectrogram_image_path,
                        "emotion": predicted_emotion,
                        "confidence": confidence,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    })

                    PROCESSED_FILES.add(wav_file)

                except Exception as e:
                    print(f"Error processing file {wav_file}: {e}")

        time.sleep(PREDICTION_INTERVAL)

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Emotion Prediction</title>
        <script>
            function fetchUpdates() {
                fetch('/latest')
                .then(response => response.json())
                .then(data => {
                    if (data.image_url) {
                        document.getElementById('spectrogram').src = data.image_url;
                        document.getElementById('emotion').innerText = data.emotion;
                        document.getElementById('confidence').innerText = data.confidence.toFixed(2);
                        document.getElementById('timestamp').innerText = data.timestamp;
                    }
                });
            }

            setInterval(fetchUpdates, 2000);
        </script>
    </head>
    <body>
        <h1>Emotion Prediction</h1>
        <img id="spectrogram" src="" alt="Spectrogram" style="max-width: 100%; height: auto;">
        <p><strong>Emotion:</strong> <span id="emotion">No data</span></p>
        <p><strong>Confidence:</strong> <span id="confidence">0.00</span>%</p>
        <p><strong>Last Updated:</strong> <span id="timestamp">N/A</span></p>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/latest')
def latest():
    image_path = LATEST_PREDICTION.get("image_path")
    if image_path:
        image_url = f"/images/{os.path.basename(image_path)}"
    else:
        image_url = None

    return jsonify({
        "image_url": image_url,
        "emotion": LATEST_PREDICTION.get("emotion"),
        "confidence": float(LATEST_PREDICTION.get("confidence") or 0.0),  # Convert to float
        "timestamp": LATEST_PREDICTION.get("timestamp")
    })

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    from threading import Thread

    monitor_thread = Thread(target=monitor_and_predict, daemon=True)
    monitor_thread.start()

    app.run(debug=True, port=5001, use_reloader=False)

