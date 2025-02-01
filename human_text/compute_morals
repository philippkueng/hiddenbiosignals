import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

class CustomRegressionModel(nn.Module):
    def __init__(self, transformer_name, aux_input_size, output_size):
        super(CustomRegressionModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(transformer_name)
        self.aux_fc = nn.Linear(aux_input_size, 64)
        self.fc_combined = nn.Linear(self.text_encoder.config.hidden_size + 64, 128)
        self.regression_head = nn.Linear(128, output_size)

    def forward(self, input_ids, attention_mask, aux_features):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_embedding = torch.mean(text_output, dim=1)
        aux_embedding = self.aux_fc(aux_features)
        combined = torch.cat([text_embedding, aux_embedding], dim=1)
        combined = torch.relu(self.fc_combined(combined))
        outputs = self.regression_head(combined)
        return outputs

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)



    # Add default value for `is_augmented` if not present (for backward compatibility)
    if 'is_augmented' not in df.columns:
        df['is_augmented'] = 0

    # Features for input
    input_texts = df['body'].tolist()
    auxiliary_features = df[['num_nouns', 'num_verbs', 'num_adjectives', 'num_adverbs',
                             'first_person_pronoun_score', 'second_person_pronoun_score','sentiment_score',
                             'age_user', 'gender_user']].fillna(0).values

    # Targets for classification (convert to integer labels)
    targets = df[['fairness_reciprocity_score_moral', 'authority_respect_score_moral', 'purity_sanctity_score_moral']].astype(float).values

    # `is_augmented` column
    is_augmented = df['is_augmented'].values

    return input_texts, auxiliary_features, targets, is_augmented

def prepare_datasets(input_texts, auxiliary_features, targets, is_augmented, tokenizer_name, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Convert is_augmented to boolean mask for selecting only original records for testing
    is_original_mask = (is_augmented == 0)

    # Split data into train/val/test sets
    total_size = len(input_texts)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)

    # Ensure test set uses only original records
    original_indices = np.where(is_original_mask)[0]
    if len(original_indices) < (total_size - train_size - val_size):
        raise ValueError("Not enough original samples for the test set.")
    
    np.random.shuffle(original_indices)
    test_indices = original_indices[-(total_size - train_size - val_size):]

    train_indices = slice(0, train_size)
    val_indices = slice(train_size, train_size + val_size)

    train_encodings = {k: v[train_indices].to(device) for k, v in encodings.items()}
    val_encodings = {k: v[val_indices].to(device) for k, v in encodings.items()}
    test_encodings = {k: v[test_indices].to(device) for k, v in encodings.items()}

    train_aux = torch.tensor(auxiliary_features[train_indices], dtype=torch.float32, device=device)
    val_aux = torch.tensor(auxiliary_features[val_indices], dtype=torch.float32, device=device)
    test_aux = torch.tensor(auxiliary_features[test_indices], dtype=torch.float32, device=device)

    # Convert targets to float for regression
    train_targets = torch.tensor(targets[train_indices], dtype=torch.float32, device=device)
    val_targets = torch.tensor(targets[val_indices], dtype=torch.float32, device=device)
    test_targets = torch.tensor(targets[test_indices], dtype=torch.float32, device=device)

    # Create datasets
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_aux, train_targets)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_aux, val_targets)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_aux, test_targets)

    return train_dataset, val_dataset, test_dataset




def train_model(model, train_loader, val_loader, device, num_epochs=100, early_stopping_patience=20):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    # Define target labels
    target_labels = ['fairness_reciprocity_score_moral', 'authority_respect_score_moral', 'purity_sanctity_score_moral']

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")

        for input_ids, attention_mask, aux_features, targets in train_progress:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, aux_features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_progress.set_postfix({"Loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_loss, r2, accuracies, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, R2={r2:.4f}")
        
        # Print accuracies with correct labels
        for label, acc in zip(target_labels, accuracies):
            print(f"Accuracy for {label}: {acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for input_ids, attention_mask, aux_features, targets in data_loader:
            outputs = model(input_ids, attention_mask, aux_features)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # Move predictions and labels to CPU for numpy conversion
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    r2 = r2_score(all_labels, all_preds)
    
    accuracies = [r2_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    
    return avg_loss, r2, accuracies, all_preds, all_labels
    


def export_predictions(model, data_loader, input_texts, auxiliary_features, original_df, file_path, device):
    """Exports model predictions along with original input data and user IDs to a CSV file."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (input_ids, attention_mask, aux_features, targets) in enumerate(data_loader):
            outputs = model(input_ids, attention_mask, aux_features)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Create a DataFrame with predictions and true values
    pred_df = pd.DataFrame(all_preds, columns=['Pred_fairness_reciprocity_score_moral', 'Pred_authority_respect_score_moral', 'Pred_purity_sanctity_score_moral'])
    true_df = pd.DataFrame(all_labels, columns=['True_fairness_reciprocity_score_moral', 'True_authority_respect_score_moral', 'True_purity_sanctity_score_moral'])

    # Combine with the original user_id and input features
    final_df = pd.concat([original_df[['mail_user']], pred_df, true_df], axis=1)

    # Export to CSV
    final_df.to_csv(file_path, index=False)
    print(f"Predictions exported to {file_path}")


def main():
    # Configuration
    file_path = 'updated_aggregated_body_aug_anno_moral.xlsx'    
    # path name to excel file with moral value labels and text
    # user_id	mail_user	age_user	gender_user	harm_care_score_moral	fairness_reciprocity_score_moral	in_group_loyality_score_moral	authority_respect_score_moral	purity_sanctity_score_moral	body	is_augmented	Text	num_nouns	num_verbs	num_adjectives	num_adverbs	first_person_pronoun_score	second_person_pronoun_score	third_person_pronoun_score	sentiment_score	sentiment_label
    # 2044	user@mail.com		0	15	18	6	12	7	text	15	4	2	2	0	0	0	-0.966681838	negative
  
    transformer_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable MPS optimizations if available
    if device.type == "mps":
        torch.backends.mps.enable_fallback_implementations = True
    
    # Load and preprocess data
    input_texts, auxiliary_features, targets, is_augmented = load_and_preprocess_data(file_path)
    
    # Prepare datasets with device awareness
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        input_texts, auxiliary_features, targets, is_augmented, transformer_name, device
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model and move to device
    model = CustomRegressionModel(
        transformer_name,
        aux_input_size=auxiliary_features.shape[1],
        output_size=3
    ).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples (original only): {len(test_dataset)}")

    # Check for test set size
    if len(test_loader) == 0:
        print("No samples available in the test set.")
    else:
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load("best_model.pth"))
        loss_fn = nn.MSELoss()
        test_loss, r2, accuracies, test_preds, test_labels  = evaluate_model(model, test_loader, loss_fn, device)

        mse = mean_squared_error(test_labels, test_preds)
        mae = mean_absolute_error(test_labels, test_preds)

        target_labels = ['fairness_reciprocity_score_moral', 'authority_respect_score_moral', 'purity_sanctity_score_moral']

        # Export predictions to CSV (only for the test set)
        export_predictions(
            model,
            test_loader,
            input_texts[len(input_texts) - len(test_dataset):],  # Only test set texts
            auxiliary_features[len(auxiliary_features) - len(test_dataset):],
            pd.read_excel(file_path),
            "test_predictions_FFI_aug.csv",
            device
        )

        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")

        for label, acc in zip(target_labels, accuracies):
            print(f"Test Accuracy for {label}: {acc:.4f}")

if __name__ == "__main__":
    main()
