# duplicate dataset by rountrip translation from German to Spanish and back

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm  # For the progress bar
import re  # For cleaning text

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize and remove non-printable characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())  # Replace multiple spaces/newlines with a single space
    return cleaned_text
    
    
import time

def safe_translate(text, source, target, retries=3):
    for attempt in range(retries):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Pause before retrying
            else:
                raise e


# Function to perform round-trip translation with truncation and cleaning   - mid_lang was 'fr' before
def round_trip_translation(text, src_lang='de', mid_lang='es', back_lang='de', max_length=4900):
    try:
        cleaned_text = clean_text(text)
        truncated_text = cleaned_text[:max_length]

        if len(truncated_text) == 0:
            raise ValueError("Text length is zero after cleaning.")

        # Safe translation
        french_text = safe_translate(truncated_text, src_lang, mid_lang)
        german_text = safe_translate(french_text, mid_lang, back_lang)

        return german_text
    except Exception as e:
        print(f"Translation failed for text: {text}\nError: {e}")
        return text  # Fallback to original if translation fails


# Function to augment dataset
def augment_training_data(file_path):
    # Load Excel file
    df = pd.read_excel(file_path)

    # Add `is_augmented` column for original records
    df['is_augmented'] = 0

    # Translate and double dataset
    new_rows = []
    print("Starting translation...")

    # Add progress bar using tqdm
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Translating rows"):
        original_text = row.get('body', "")
        translated_text = round_trip_translation(original_text)

        # Create a new row with translated text and mark as augmented
        new_row = row.copy()
        new_row['body'] = translated_text
        new_row['is_augmented'] = 1  # Mark as augmented

        new_rows.append(new_row)

    # Create a new DataFrame with translated entries
    augmented_df = pd.DataFrame(new_rows)

    # Concatenate original and augmented DataFrames
    doubled_df = pd.concat([df, augmented_df], ignore_index=True)

    return doubled_df

# Usage example
input_file_path = 'updated_aggregated_body.xlsx'
output_file_path = 'updated_aggregated_body_aug.xlsx'

# Show progress while saving the file
doubled_data = augment_training_data(input_file_path)
print("Saving augmented dataset...")
doubled_data.to_excel(output_file_path, index=False)
print(f"Augmented dataset saved to {output_file_path}")

