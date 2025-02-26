#trains a model to predict if user is asleep or awake from the spectrograms

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from PIL import Image

# Directories
spectrogram_dir = "spectrograms"

# **Load Spectrograms & Labels**
spectrograms = []
labels = []

for file in os.listdir(spectrogram_dir):
    if file.endswith(".png"):
        label = 0 if "awake" in file else 1  # Awake = 0, Asleep = 1
        spectrograms.append(os.path.join(spectrogram_dir, file))
        labels.append(label)

# **Dataset Class for Spectrogram Images**
class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# **Transformations (Simple, No Extra Noise)**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# **Split into Train & Validation Sets**
train_images, val_images, train_labels, val_labels = train_test_split(
    spectrograms, labels, test_size=0.2, random_state=42
)

train_dataset = SpectrogramDataset(train_images, train_labels, transform=transform)
val_dataset = SpectrogramDataset(val_images, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# **Set Device: Use Metal GPU (MPS) if available**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# **Load Pretrained ResNet18 Model**
model = resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

# **Loss Function**
criterion = nn.CrossEntropyLoss()

# **Optimizer & LR Scheduler**
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)  # Reduce LR if no improvement in 3 epochs

# **Early Stopping Parameters**
patience = 5
best_val_loss = float("inf")
patience_counter = 0
best_model_path = "best_model.pth"

# **Tracking Metrics**
num_epochs = 50
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# **Training Loop**
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # **Validation Phase**
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(100 * correct / total)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

    # **Adjust Learning Rate if No Improvement**
    scheduler.step(val_loss)

    # **Check for Early Stopping**
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("🔥 Best model saved!")
    else:
        patience_counter += 1
        print(f"Early stopping patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("⏹️ Early stopping triggered. Training stopped.")
        break

# **Load Best Model for Final Evaluation**
model.load_state_dict(torch.load(best_model_path))
print("✅ Loaded best model.")

# **Testing & Evaluation**
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# **Plot Loss and Accuracy**
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='-', color='r', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', linestyle='-', color='g', label="Train Acc")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='s', linestyle='-', color='orange', label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# **Confusion Matrix**
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Awake", "Asleep"])
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels, all_preds, target_names=["Awake", "Asleep"]))
