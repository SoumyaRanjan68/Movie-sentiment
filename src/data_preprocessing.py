"""
Data Preprocessing Module for Movie Review Sentiment Analysis
-------------------------------------------------------------
Objective: Clean, normalize, and prepare raw movie review texts
while rigorously preserving sentiment markers and negations.
"""

import re
import html
import os
import pandas as pd

# Contraction mapping to expand short forms and keep negation words explicit
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "must've": "must have",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "where's": "where is", "who's": "who is",
    "won't": "will not", "would've": "would have", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have"
}

HTML_TAG_REGEX = re.compile(r'<.*?>')
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
NON_ALPHANUM_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
WHITESPACE_REGEX = re.compile(r'\s+')
CONTRACTIONS_REGEX = re.compile(
    r'\b(' + r'|'  .join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r')\b',
    flags=re.IGNORECASE
)


def expand_contractions(text: str) -> str:
    """
    Expands common contractions (e.g., 'don\'t' -> 'do not').
    Preserving 'not' prevents 'not good' from turning into 'good' during tokenization.
    """
    def replace(match):
        matched_text = match.group(0).lower()
        return CONTRACTION_MAP.get(matched_text, matched_text)
    return CONTRACTIONS_REGEX.sub(replace, text)


def clean_text(text: str) -> str:
    """
    Applies standard, explainable NLP cleaning to raw review text:
    1. Unescapes HTML entities (&amp;, &lt;, etc.)
    2. Strips HTML tags (<br />, <p>, etc.)
    3. Strips web URLs
    4. Expands contractions to preserve negation words
    5. Converts to lowercase for vocabulary consistency
    6. Removes special characters / punctuation (preserves letters and numbers)
    7. Collapses consecutive whitespace characters into a single space
    """
    if not isinstance(text, str):
        return ''
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Remove HTML tags
    text = HTML_TAG_REGEX.sub(' ', text)
    
    # 3. Remove URLs
    text = URL_REGEX.sub(' ', text)
    
    # 4. Expand contractions
    text = expand_contractions(text)
    
    # 5. Convert to lowercase
    text = text.lower()
    
    # 6. Remove non-alphanumeric characters
    text = NON_ALPHANUM_REGEX.sub(' ', text)
    
    # 7. Collapse whitespace
    text = WHITESPACE_REGEX.sub(' ', text).strip()
    
    return text


def load_and_preprocess_data(raw_data_path: str, processed_data_path: str = None) -> pd.DataFrame:
    """
    Loads raw IMDB CSV data, validates integrity, removes duplicate reviews,
    applies clean_text, encodes binary labels, and optionally persists
    the processed DataFrame.
    """
    print(f'[*] Loading raw data from: {raw_data_path}')
    df = pd.read_csv(raw_data_path)
    print(f'[*] Raw dataset shape: {df.shape}')
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f'[!] Found {missing_count} missing values. Dropping rows.')
        df = df.dropna().reset_index(drop=True)
    else:
        print('[+] Zero missing values found.')
        
    # Check and remove duplicates
    duplicate_count = df.duplicated(subset=['review']).sum()
    print(f'[*] Duplicate reviews found: {duplicate_count}')
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
        print(f'[+] Dataset shape after deduplication: {df.shape}')
        
    # Apply text cleaning pipeline
    print('[*] Cleaning review text (HTML removal, contraction expansion, normalization)...')
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Target encoding: positive -> 1, negative -> 0
    df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Review length metrics for EDA
    df['char_length'] = df['cleaned_review'].apply(len)
    df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    
    if processed_data_path:
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        print(f'[*] Saving processed data to: {processed_data_path}')
        df.to_csv(processed_data_path, index=False)
        print('[+] Processed data saved successfully.')
        
    return df


if __name__ == '__main__':
    raw_p = os.path.join('data', 'raw', 'imdb_reviews.csv')
    proc_p = os.path.join('data', 'processed', 'cleaned_reviews.csv')
    
    sample_review = "I hated this film! It wasn't good at all. <br /> http://sample.com"
    print(f"Raw: {sample_review}")
    print(f"Cleaned: {clean_text(sample_review)}")
    
    df = load_and_preprocess_data(raw_p, proc_p)
    print(dfhead(3))
