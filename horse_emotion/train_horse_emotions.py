import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Select device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
DATASET_PATH = "trainingimages"  # Change this to your dataset path

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 224  # Image size for ResNet

# Data transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Class names
class_names = full_dataset.classes
print(f"Classes: {class_names}")

# Split dataset into Train (70%), Validation (15%), Test (15%)
train_size = int(0.7 * len(full_dataset))   # 280 images
val_size = int(0.15 * len(full_dataset))    # 60 images
test_size = len(full_dataset) - train_size - val_size  # 60 images

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))

# Move model to Mac GPU (Metal)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track losses and accuracies
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct/total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, return_loss=True)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_horse_emotions_model.pth")
            print("Saved best model!")

# Evaluation function
def evaluate_model(model, data_loader, return_loss=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    if return_loss:
        return running_loss / len(data_loader), acc
    return acc

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Load best model and evaluate on the test set
model.load_state_dict(torch.load("best_horse_emotions_model.pth"))
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")

# Plot Loss and Accuracy
def plot_metrics():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax[0].plot(range(1, EPOCHS+1), train_losses, label="Train Loss", marker="o")
    ax[0].plot(range(1, EPOCHS+1), val_losses, label="Val Loss", marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].legend()

    # Accuracy plot
    ax[1].plot(range(1, EPOCHS+1), train_accuracies, label="Train Accuracy", marker="o")
    ax[1].plot(range(1, EPOCHS+1), val_accuracies, label="Val Accuracy", marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Accuracy over Epochs")
    ax[1].legend()

    plt.show()

plot_metrics()

# Generate confusion matrix
def plot_confusion_matrix():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix on Test Set")
    plt.show()

plot_confusion_matrix()

