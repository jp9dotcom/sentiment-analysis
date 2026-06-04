"""
Phase 1: Data Ingestion and Advanced Preprocessing Pipeline
Handles: HTML removal, URL removal, contraction expansion, elongated word normalization,
         emoji removal, digit removal, tokenization, POS tagging, lemmatization, stop word removal
"""

import logging
import re
import time
import os
import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data")
RAW_DATA_PATH = Path("data\dataset.csv")  # Update this to your actual file path

# ─────────────────────────────────────────────────────────────────────────────
# NLP Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_nltk():
    """Download required NLTK resources."""
    # Added 'averaged_perceptron_tagger_eng' and 'punkt_tab' for newer NLTK versions
    resources = [
        'punkt', 
        'punkt_tab',                      # Added for newer NLTK tokenization
        'stopwords', 
        'wordnet', 
        'omw-1.4', 
        'averaged_perceptron_tagger',     # Kept for backwards compatibility
        'averaged_perceptron_tagger_eng'  # <--- THIS IS THE FIX
    ]
    
    for resource in resources:
        nltk.download(resource, quiet=True)
    logger.info("NLTK resources downloaded ✓") 

# Initialize NLP tools
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# Preserve negation words for sentiment analysis
NEGATIONS = {'not', 'no', 'nor', 'never', "n't", 'don', 'doesn', 'didn', 'won', 'wouldn', 'shouldn'}
STOP_WORDS = STOP_WORDS - NEGATIONS

# Comprehensive contraction mapping
CONTRACTION_MAP = {
    # Negative contractions
    "aren't": "are not", "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "can't": "cannot", "cannot": "cannot",
    "shan't": "shall not", "mustn't": "must not", "needn't": "need not",
    
    # Pronoun + verb contractions
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "it'll": "it will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would",
    
    # Question contractions
    "what's": "what is", "who's": "who is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
    
    # Other common contractions
    "that's": "that is", "there's": "there is", "here's": "here is",
    "let's": "let us", "who's": "who is", "what're": "what are",
    "might've": "might have", "must've": "must have", "could've": "could have",
    "would've": "would have", "should've": "should have"
}

# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def expand_contractions(text: str) -> str:
    """
    Expand contractions in text.
    Examples: "aren't" -> "are not", "soooo" -> "so"
    """
    # Handle contractions
    for contraction, expansion in CONTRACTION_MAP.items():
        text = text.replace(contraction, expansion)
    
    return text

def remove_elongated_chars(text: str) -> str:
    """
    Remove elongated characters.
    Examples: "sooooo" -> "so", "gooooood" -> "good", "loooove" -> "love"
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text: str) -> str:
    """
    Complete text cleaning pipeline for IMDb reviews.
    
    Steps:
    1. Handle null/invalid input
    2. Remove HTML tags
    3. Remove URLs
    4. Expand contractions (aren't -> are not)
    5. Remove elongated characters (sooooo -> so)
    6. Lowercase
    7. Remove emojis (non-ASCII characters)
    8. Remove digits
    9. Remove special characters & punctuation
    10. Normalize whitespace
    11. Tokenize
    12. POS tagging
    13. Lemmatization with POS awareness
    14. Remove stop words (preserve negations)
    15. Remove short tokens (length <= 2)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # 3. Expand contractions (aren't -> are not, etc.)
    text = expand_contractions(text)
    
    # 4. Remove elongated characters (sooooo -> so)
    text = remove_elongated_chars(text)
    
    # 5. Lowercase
    text = text.lower()
    
    # 6. Remove emojis (non-ASCII characters)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 7. Remove digits
    text = re.sub(r'\d+', ' ', text)
    
    # 8. Remove special characters & punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 10. Tokenize
    tokens = word_tokenize(text)
    
    # 11. POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # 12. Lemmatization, stop word removal, and short token removal
    cleaned_tokens = []
    for word, tag in pos_tags:
        # Remove stop words and short tokens
        if len(word) > 2 and word not in STOP_WORDS:
            # Lemmatize with POS tag
            wn_pos = get_wordnet_pos(tag)
            lemma = LEMMATIZER.lemmatize(word, pos=wn_pos)
            cleaned_tokens.append(lemma)
    
    return ' '.join(cleaned_tokens)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading and Validation
# ────────────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Load dataset and perform basic validation."""
    logger.info(f"Loading data from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please update RAW_DATA_PATH in the script.")
    
    df = pd.read_csv(path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        df = df.dropna()
        logger.info(f"After dropping nulls: {df.shape}")
    
    # Remove duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.info(f"Removing {n_dupes} duplicate rows")
        df = df.drop_duplicates().reset_index(drop=True)
    
    logger.info(f"Final dataset shape: {df.shape}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning pipeline to dataset."""
    logger.info("Starting text cleaning pipeline...")
    logger.info("This may take a few minutes for large datasets...")
    
    # Label encoding
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    logger.info(f"Label encoding: positive=1, negative=0")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Calculate word counts before cleaning
    df['word_count'] = df['review'].str.split().str.len()
    
    # Apply cleaning pipeline
    t0 = time.time()
    df['cleaned'] = df['review'].apply(clean_text)
    elapsed = time.time() - t0
    logger.info(f"Cleaning completed in {elapsed:.1f} seconds")
    
    # Calculate word counts after cleaning
    df['cleaned_word_count'] = df['cleaned'].str.split().str.len()
    
    # Remove empty reviews after cleaning
    empty_count = (df['cleaned'].str.strip() == '').sum()
    if empty_count > 0:
        logger.warning(f"Removing {empty_count} empty reviews after cleaning")
        df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)
    
    # Calculate text length feature
    df['text_length'] = df['cleaned_word_count']
    
    # Log statistics
    logger.info(f"Average word count - Raw: {df['word_count'].mean():.1f}, Cleaned: {df['cleaned_word_count'].mean():.1f}")
    vocab_reduction = (1 - df['cleaned_word_count'].mean() / df['word_count'].mean()) * 100
    logger.info(f"Vocabulary reduction: {vocab_reduction:.1f}%")
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Data Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Split data into train, validation, and test sets.
    Strategy: 70% train, 15% validation, 15% test with stratification.
    """
    logger.info("Splitting data into train/val/test sets...")
    
    X = df['cleaned'].to_numpy(dtype=str)
    y = df['label'].to_numpy(dtype=int)
    
    # Step 1: 85% train+val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    
    # Step 2: Split train+val into 70% train, 15% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15/0.85, random_state=SEED, stratify=y_trainval
    )
    
    # Log split summary
    for name, X_s, y_s in [('Train', X_train, y_train), 
                           ('Validation', X_val, y_val), 
                           ('Test', X_test, y_test)]:
        pos = y_s.sum()
        neg = len(y_s) - pos
        ratio = pos / len(y_s)
        logger.info(f"{name:10s}: {len(X_s):6,} samples | positive: {pos:,} | negative: {neg:,} | ratio: {ratio:.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────────────────────────────────────
# Data Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save processed data to disk."""
    logger.info("Saving processed data...")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full cleaned dataframe
    df.to_csv(DATA_DIR / 'imdb_cleaned.csv', index=False)
    logger.info(f"✓ Saved: {DATA_DIR / 'imdb_cleaned.csv'}")
    
    # Save train/val/test splits as numpy arrays
    np.save(DATA_DIR / 'X_train.npy', X_train)
    np.save(DATA_DIR / 'X_val.npy', X_val)
    np.save(DATA_DIR / 'X_test.npy', X_test)
    np.save(DATA_DIR / 'y_train.npy', y_train)
    np.save(DATA_DIR / 'y_val.npy', y_val)
    np.save(DATA_DIR / 'y_test.npy', y_test)
    logger.info(f"✓ Saved: {DATA_DIR / '*.npy'} (6 files)")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1 Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Train: {len(X_train):,}")
    logger.info(f"Validation: {len(X_val):,}")
    logger.info(f"Test: {len(X_test):,} (locked - do not touch until final evaluation)")

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_1():
    """Main pipeline execution."""
    logger.info("="*60)
    logger.info("Phase 1: Data Ingestion and Advanced Preprocessing")
    logger.info("="*60)
    
    # Setup
    setup_nltk()
    
    # Load data
    df = load_data(RAW_DATA_PATH)
    
    # Process data
    df = process_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Save data
    save_data(df, X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("\nReady for Phase 2: Vectorization and Model Training")

if __name__ == "__main__":
    run_phase_1()