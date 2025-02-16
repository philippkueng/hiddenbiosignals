import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Check for MPS (Metal) GPU availability
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load data
ants_df = pd.read_excel("ants.xlsx", header=None, names=["text"])
bees_df = pd.read_excel("bees.xlsx", header=None, names=["text"])
leeches_df = pd.read_excel("leeches.xlsx", header=None, names=["text"])

# Assign labels
ants_df["labels"] = 0  # ant
bees_df["labels"] = 1  # bee
leeches_df["labels"] = 2  # leech

# Combine data
df = pd.concat([ants_df, bees_df, leeches_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["labels"], test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": X_train.tolist(), "labels": y_train.tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "labels": y_test.tolist()})

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove non-tensor columns
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Set PyTorch format
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Load model and move to MPS
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)  # Move model to MPS

# Function to move dataset to MPS
def to_mps(batch):
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=False  # MPS doesn't support fp16 properly yet
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=lambda data: to_mps({key: torch.stack([f[key] for f in data]) for key in data[0]})  # Move data to MPS
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ant", "bee", "leech"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ant", "bee", "leech"], yticklabels=["ant", "bee", "leech"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
