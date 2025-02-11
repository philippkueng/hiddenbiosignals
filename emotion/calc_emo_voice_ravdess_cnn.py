# train and test model to predict emotions from voice, using the RAVDESS dataset 
# https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # For nicer confusion matrix plots
import tensorflow as tf  # Import TensorFlow
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # For better training
from sklearn.preprocessing import StandardScaler

# Check and set TensorFlow device (Metal GPU if available)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # Check for GPU
try:
    # For TensorFlow versions that support Metal
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)  # Enable memory growth
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')  # Use Metal GPU
    logical_device = tf.config.list_logical_devices('GPU')[0] # Get logical device
    print(f"Using {logical_device} for training.")
except RuntimeError as e:
    print(e) # Print any errors encountered
    print("Using CPU for training.") # Fallback to CPU if Metal is not working

data_dir = "sorted_audio"  # Replace with actual path
emotions = {  # Dictionary mapping emotion labels to numerical indices
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
sample_rate = 16000  # Standard for audio processing
input_shape = (224, 224, 3)  # Input shape for VGGish (after preprocessing)
n_classes = len(emotions)

def extract_features(file_path, save_spectrograms=False, spectrogram_dir="save_spectro_vgg"):
    """Loads audio, extracts mel spectrogram, and preprocesses for VGGish."""
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        # Extract Mel spectrogram (adjust n_mels as needed)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Example: 128 mel bands
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) # Convert to dB scale
        
        # Resize/pad to VGGish input size (224x224)
        import cv2 # Make sure you have opencv-python installed
        img = cv2.resize(mel_spectrogram, (224, 224))
        img = np.stack([img]*3, axis=-1) # Convert to 3 channels (RGB-like)
        img = img/255.0 # Normalize pixel values

        if save_spectrograms:  # Save if requested
            if spectrogram_dir is None:
                raise ValueError("spectrogram_dir must be specified if save_spectrograms is True")
            file_name = os.path.basename(file_path)  # Get the file name
            spectrogram_name = os.path.splitext(file_name)[0] + ".png"  # Create spectrogram name
            spectrogram_path = os.path.join(spectrogram_dir, spectrogram_name)

            # Create the directory if it doesn't exist
            os.makedirs(spectrogram_dir, exist_ok=True)

            plt.figure(figsize=(10, 4))  # Adjust figure size as needed
            librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time') # Display the spectrogram
            plt.colorbar(format='%+2.0f dB') # Add a colorbar
            plt.title(f"Mel Spectrogram: {file_name}")
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plt.savefig(spectrogram_path)  # Save the spectrogram
            plt.close()  # Close the figure to free memory
        return img
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def load_data(data_dir):
    """Loads audio files, extracts features, and creates labels."""
    features = []
    labels = []

    for emotion_code, emotion_name in emotions.items():
        emotion_dir = os.path.join(data_dir, emotion_name) # RAVDESS folder structure
        print(f"Data directory: {emotion_dir}")  # Add this line
        if os.path.exists(emotion_dir):  # Check if directory exists
            for file_name in os.listdir(emotion_dir):
                if file_name.endswith(".wav"):  # Only process wav files
                    file_path = os.path.join(emotion_dir, file_name)
                    feature = extract_features(file_path)
                    if feature is not None:  # Skip files with errors
                        features.append(feature)
                        labels.append(emotion_code)  # Store the emotion code as label

    # Convert labels to numerical indices
    labels = [list(emotions.keys()).index(label) for label in labels]
    return np.array(features), np.array(labels)

# Load data
X, y = load_data(data_dir)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  # Your features (X) and labels (y)
    test_size=0.2,  # Or whatever your test size is
    random_state=42,  # For reproducibility (important!)
    shuffle=True  # Explicitly set shuffle to True (it's the default, but good practice)
)

scaler = StandardScaler()

# Reshape for StandardScaler (important!)
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # Reshape to 2D
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

X_train = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape) # Fit and transform training data
X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape) # Transform test data using the fitted scaler

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

# Load pre-trained VGG16 (or VGGish if you adapt it)
input_shape = (224, 224, 3)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)),  # Add L2 regularization
    keras.layers.Dropout(0.5),  # Increase dropout if needed
    keras.layers.Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop if val_loss doesn't improve
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)  # Reduce LR if val_loss plateaus

# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,  # Increased epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices
y_true = np.argmax(y_test, axis=1)  # Get true class indices

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(emotions.values()),  # Use emotion names
            yticklabels=list(emotions.values()))  # Use emotion names
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save("emotion_recognition_model.h5")
