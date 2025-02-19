# testing the face emotion recognition model with some face pictures

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from PIL import Image

# Define emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the CNN model class
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, len(EMOTION_LABELS))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Function to predict emotion from an uploaded image
def predict_emotion(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        emotion = EMOTION_LABELS[predicted_class]
    
    # Display image and prediction
    plt.imshow(Image.open(image_path), cmap="gray")
    plt.title(f"Predicted Emotion: {emotion}")
    plt.axis("off")
    plt.show()
    return emotion

# Example usage - Upload and test with images
image_paths = ["test1.png", "test2.png","test3.png"]  # Replace with your actual image paths
for img_path in image_paths:
    if os.path.exists(img_path):
        predict_emotion(img_path)
    else:
        print(f"Image {img_path} not found. Please upload and try again.")

