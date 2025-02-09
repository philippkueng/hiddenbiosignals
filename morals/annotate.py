# Import necessary libraries
import spacy
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
# python -m spacy download de_core_news_sm

# Load German NLP model and stopwords
nlp = spacy.load("de_core_news_sm")
german_stop_words = stopwords.words('german')

# Initialize sentiment analysis pipeline with a German model
#sentiment_analyzer = pipeline(
#    "sentiment-analysis",
#    model="oliverguhr/german-sentiment-bert",
#    tokenizer="oliverguhr/german-sentiment-bert",
#    framework="pt"
#)
# Load the model and tokenizer explicitly
model_name = "oliverguhr/german-sentiment-bert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # PyTorch version
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create sentiment-analysis pipeline
sentiment_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt")


# Load tokenizer for pronoun analysis
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Define pronoun sets for analysis
pronouns = {
    'first_person': ['ich', 'wir'],  # "I", "we" in German
    'second_person': ['du', 'ihr', 'sie'],  # "You" in German
    'third_person': ['er', 'sie', 'es'],  # "He", "She", "It" in German
}

# Define the function for linguistic feature extraction
def count_features(text):
    doc = nlp(text)
    num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
    num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
    num_adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
    num_adverbs = sum(1 for token in doc if token.pos_ == "ADV")
    return {
        "num_nouns": num_nouns,
        "num_verbs": num_verbs,
        "num_adjectives": num_adjectives,
        "num_adverbs": num_adverbs,
    }

# Define the function for pronoun usage analysis
def calculate_pronoun_usage(text):
    tokens = tokenizer.tokenize(text.lower(), truncation=True, max_length=512)  # Tokenize and convert to lowercase
    pronoun_counts = {key: sum(tokens.count(p) for p in pronouns[key]) for key in pronouns}
    total_tokens = len(tokens)
    normalized_counts = {f"{key}_pronoun_score": count / total_tokens if total_tokens > 0 else 0
                         for key, count in pronoun_counts.items()}
    return normalized_counts

# Define the function for sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens for performance
    sentiment_label = result['label'].lower()  # e.g., "positive", "neutral", or "negative"
    sentiment_score = result['score'] if sentiment_label == "positive" else -result['score']
    return {"sentiment_score": sentiment_score, "sentiment_label": sentiment_label}

# Combine everything into a pipeline
def process_texts(texts):
    results = []
    for i, text in enumerate(texts):
        # Extract linguistic features
        linguistic_features = count_features(text)

        # Analyze pronoun usage
        pronoun_features = calculate_pronoun_usage(text)

        # Analyze sentiment
        sentiment_features = analyze_sentiment(text)

        # Combine all features and include the text
        result = {"Text": text, **linguistic_features, **pronoun_features, **sentiment_features}
        results.append(result)

    return pd.DataFrame(results)

# Input texts
# Load dataset
file_path = 'updated_aggregated_body_aug.xlsx'  # Update with correct path
df = pd.read_excel(file_path)
# Select textual columns to analyze (adjust as needed)
text_column = 'body'  # Example column
df[text_column] = df[text_column].fillna("")  # Fill NaNs with empty strings
texts = df[text_column]

# Run the pipeline
results_df = process_texts(texts)

# Merge results with original DataFrame
df = pd.concat([df, results_df], axis=1)

# Save results to Excel
output_path = "updated_aggregated_body_aug_anno.xlsx"
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")

