import os
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Check for MPS support
if tf.config.list_physical_devices('GPU'):
    print("Using MPS")
    tf.keras.backend.set_floatx('float32')
else:
    print("Using CPU")


def load_and_preprocess_data(spectrogram_dir):
    """
    Load spectrograms and their labels from the directory.
    """
    images = []
    labels = []
    
    for filename in os.listdir(spectrogram_dir):
        if filename.endswith('.png'):
            # Extract emotion type from filename
            emotion_type = filename.split('_')[-1].replace('.png', '')
            
            # Load and preprocess image
            img_path = os.path.join(spectrogram_dir, filename)
            img = Image.open(img_path)
            # Convert grayscale to RGB by duplicating channels
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            
            # Preprocess for ResNet
            img_array = preprocess_input(img_array)
            
            images.append(img_array)
            labels.append(emotion_type)
    
    return np.array(images), np.array(labels)


def create_resnet_model(num_classes):
    """
    Create a ResNet50 model with transfer learning
    """
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create new model with custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


from sklearn.utils import resample
from collections import Counter
import numpy as np

def ensure_stratification(X, y, min_samples_per_class):
    """
    Adjust dataset by:
    - Downsampling the largest class to the size of the second-largest class.
    - Oversampling smaller classes to meet `min_samples_per_class`.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): One-hot encoded labels.
        min_samples_per_class (int): Minimum number of samples per class.

    Returns:
        X_resampled (ndarray): Resampled feature matrix.
        y_resampled (ndarray): Resampled one-hot encoded labels.
    """
    # Calculate class counts
    classes, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    class_counts = Counter(np.argmax(y, axis=1))
    print("Original class distribution:", class_counts)

    # Determine target size for downsampling
    counts_sorted = sorted(class_counts.values(), reverse=True)
    target_size = counts_sorted[1] if len(counts_sorted) > 1 else counts_sorted[0]
    print(f"Target size for downsampling: {target_size}")

    X_resampled, y_resampled = [], []

    for cls in classes:
        X_cls = X[np.argmax(y, axis=1) == cls]
        y_cls = y[np.argmax(y, axis=1) == cls]
        
        if len(X_cls) > target_size:  # Downsample if larger than the second-largest class
            X_cls, y_cls = resample(X_cls, y_cls, n_samples=target_size, random_state=42, replace=False)
        elif len(X_cls) < min_samples_per_class:  # Oversample if smaller than `min_samples_per_class`
            X_cls, y_cls = resample(X_cls, y_cls, n_samples=min_samples_per_class, random_state=42, replace=True)

        X_resampled.append(X_cls)
        y_resampled.append(y_cls)

    # Combine all resampled classes
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.vstack(y_resampled)

    # Print final class distribution
    final_class_counts = Counter(np.argmax(y_resampled, axis=1))
    print("Final class distribution after stratification:", final_class_counts)

    return X_resampled, y_resampled


def filter_classes(X, y_onehot, min_samples_per_class):
    """
    Filters out classes with fewer than `min_samples_per_class` samples.

    Args:
        X (ndarray): Feature matrix.
        y_onehot (ndarray): One-hot encoded labels.
        min_samples_per_class (int): Minimum number of samples for a class to be retained.

    Returns:
        X_filtered (ndarray): Filtered feature matrix.
        y_filtered (ndarray): Filtered one-hot encoded labels.
        valid_classes (list): List of valid class indices.
    """
    # Calculate class distribution
    class_counts = Counter(np.argmax(y_onehot, axis=1))
    print("Original class distribution:", class_counts)
    
    # Determine valid classes
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
    print(f"Valid classes (at least {min_samples_per_class} samples): {valid_classes}")
    
    # Validate the filtering condition
    if not valid_classes:
        raise ValueError(f"No classes have at least {min_samples_per_class} samples.")
    
    # Filter indices of valid classes
    valid_indices = np.isin(np.argmax(y_onehot, axis=1), valid_classes)
    X_filtered = X[valid_indices]
    y_filtered = y_onehot[valid_indices]
    
    # Print filtered class distribution for verification
    filtered_class_counts = Counter(np.argmax(y_filtered, axis=1))
    print("Filtered class distribution:", filtered_class_counts)
    
    return X_filtered, y_filtered, valid_classes





def preprocess_data(X, y_onehot, min_samples_per_class=70):
    """
    Preprocess and balance the dataset, ensuring stratification is possible.
    """
    # Step 1: Filter out classes with fewer than `min_samples_per_class` members
    X_filtered, y_filtered, valid_classes = filter_classes(X, y_onehot, min_samples_per_class)

    # Double-check that filtering was applied correctly
    print("Filtered class distribution:", Counter(np.argmax(y_filtered, axis=1)))

    # Step 2: Oversample to ensure sufficient samples for stratification
    X_resampled, y_resampled = ensure_stratification(X_filtered, y_filtered, min_samples_per_class)

    # Step 3: Split the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=np.argmax(y_resampled, axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    
    # Verify class distribution after splitting
    print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
    print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))
    print("Test set class distribution:", Counter(np.argmax(y_test, axis=1)))

    return X_train, X_val, X_test, y_train, y_val, y_test, valid_classes



def train_emotion_classifier(spectrogram_dir, output_model_path, epochs=50):
    """
    Train a ResNet model to classify emotion types from spectrograms
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(spectrogram_dir)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    print("Shape of y_onehot:", y_onehot.shape)
    
    # Preprocess and retrieve valid classes
    X_train, X_val, X_test, y_train, y_val, y_test, valid_classes = preprocess_data(X, y_onehot, min_samples_per_class=70)

    print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
    print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))
    print("Test set class distribution:", Counter(np.argmax(y_test, axis=1)))
    
    # Create and compile model
    print("Creating model...")
    model, base_model = create_resnet_model(num_classes)
    
    # First phase: Train only the top layers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_model_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # First training phase
    print("Training top layers...")
    history1 = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Second phase: Fine-tune the last few layers of ResNet
    print("\nFine-tuning ResNet layers...")
    # Unfreeze the last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Second training phase
    history2 = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,  # Smaller batch size for fine-tuning
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Combine histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]
    
    # Evaluate model on unseen test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nUnseen test accuracy: {test_accuracy:.4f}")
    
    # Save the LabelEncoder classes to a file
    np.save('label_classes.npy', label_encoder.classes_)
    # Generate metrics for the unseen test set
    generate_metrics(model, X_test, y_test, label_encoder, valid_classes)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_model_path), 'training_history.png'))
    plt.close()

    # Confusion Matrix
    

    return model, label_encoder, history

# Confusion Matrix and Classification Report
def generate_metrics(model, X_test, y_test, label_encoder, valid_classes):
    """
    Generate confusion matrix and classification report for the model.
    """
    # Predict on the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Map valid class indices to their original names
    valid_class_names = [label_encoder.classes_[cls] for cls in valid_classes]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=valid_class_names, labels=valid_classes
    ))

def predict_emotion(model, image_path, label_encoder):
    """
    Predict emotion type for a single spectrogram
    """
    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
    confidence = np.max(prediction[0])
    
    return predicted_class, confidence


if __name__ == "__main__":
    # Set paths
    spectrogram_dir = "emotion_spectrograms"
    model_path = "emotion_classifier_model.keras"
    
    # Train model
    try:
        model, label_encoder, history = train_emotion_classifier(
            spectrogram_dir,
            model_path,
            epochs=30  # Reduced epochs since we're doing two training phases
        )
        print("Training completed successfully!")
        
        # Example prediction
        sample_image = os.path.join(spectrogram_dir, os.listdir(spectrogram_dir)[0])
        predicted_class, confidence = predict_emotion(model, sample_image, label_encoder)
        print(f"\nSample prediction for {os.path.basename(sample_image)}:")
        print(f"Predicted emotion: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
