"""
Model Training and Vectorization Module
---------------------------------------
Objective:
1. Split dataset into Train (80%) and Test (20%) using stratification.
2. Fit TF-IDF Vectorizer ONLY on training set (preventing Data Leakage).
3. Train Logistic Regression and Multinomial Naive Bayes.
4. Evaluate both models on the unseen Test set.
5. Select and persist the best performing model and fitted vectorizer.
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.data_preprocessing import load_and_preprocess_data


def train_and_evaluate_pipeline(
    data_path: str = 'data/processed/cleaned_reviews.csv',
    raw_path: str = 'data/raw/imdb_reviews.csv',
    models_dir: str = 'models',
    random_state: int = 42
):
    """
    Executes the end-to-end training and model selection pipeline.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f'[*] Processed data not found at {data_path}. Running preprocessing on {raw_path}...')
        df = load_and_preprocess_data(raw_path, data_path)
    else:
        print(f'[*] Loading preprocessed data from {data_path}...')
        df = pd.read_csv(data_path)
        df['cleaned_review'] = df['cleaned_review'].fillna('')

    print(f'[*] Dataset shape: {df.shape}')
    print(f'[*] Class distribution:\n{df["sentiment"].value_counts()}')

    # 2. Train / Test Split (Preventing Data Leakage)
    print('\n[*] Performing Stratified Train/Test Split (80% Train, 20% Test)...')
    X = df['cleaned_review']
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f'[+] Training set size: {len(X_train)} samples')
    print(f'[+] Test set size:     {len(X_test)} samples')

    # 3. TF-IDF Vectorization
    # CRITICAL: Fit ONLY on X_train. Transform X_train and X_test.
    print('\n[*] Fitting TF-IDF Vectorizer on Training Data...')
    print('    - ngram_range: (1, 2) [captures both unigrams and bigrams like "not good", "must watch"]')
    print('    - max_features: 10,000 [limits vocabulary to top 10,000 informative terms]')
    print('    - min_df: 3 [filters terms appearing in fewer than 3 documents]')
    print('    - sublinear_tf: True [uses logarithmic term frequency 1 + log(tf)]')

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=3,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f'[+] TF-IDF Training Matrix Shape: {X_train_tfidf.shape}')
    print(f'[+] TF-IDF Test Matrix Shape:     {X_test_tfidf.shape}')

    # 4. Model Training & Comparison
    results = {}
    trained_models = {}

    # --- Model A: Logistic Regression ---
    print('\n' + '='*50)
    print('  Training Model 1: Logistic Regression (C=1.0, max_iter=1000)')
    print('='*50)
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_tfidf, y_train)
    
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_metrics = {
        'model_name': 'Logistic Regression',
        'accuracy': float(accuracy_score(y_test, lr_preds)),
        'precision': float(precision_score(y_test, lr_preds)),
        'recall': float(recall_score(y_test, lr_preds)),
        'f1_score': float(f1_score(y_test, lr_preds))
    }
    results['Logistic Regression'] = lr_metrics
    trained_models['Logistic Regression'] = lr_model
    
    print(f"Accuracy:  {lr_metrics['accuracy']:.4f}")
    print(f"Precision: {lr_metrics['precision']:.4f}")
    print(f"Recall:    {lr_metrics['recall']:.4f}")
    print(f"F1-Score:  {lr_metrics['f1_score']:.4f}")
    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, lr_preds, target_names=['Negative (0)', 'Positive (1)']))

    # --- Model B: Multinomial Naive Bayes ---
    print('='*50)
    print('  Training Model 2: Multinomial Naive Bayes (alpha=1.0)')
    print('='*50)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_metrics = {
        'model_name': 'Multinomial Naive Bayes',
        'accuracy': float(accuracy_score(y_test, nb_preds)),
        'precision': float(precision_score(y_test, nb_preds)),
        'recall': float(recall_score(y_test, nb_preds)),
        'f1_score': float(f1_score(y_test, nb_preds))
    }
    results['Multinomial Naive Bayes'] = nb_metrics
    trained_models['Multinomial Naive Bayes'] = nb_model
    
    print(f"Accuracy:  {nb_metrics['accuracy']:.4f}")
    print(f"Precision: {nb_metrics['precision']:.4f}")
    print(f"Recall:    {nb_metrics['recall']:.4f}")
    print(f"F1-Score:  {nb_metrics['f1_score']:.4f}")
    print("\nClassification Report (Multinomial Naive Bayes):")
    print(classification_report(y_test, nb_preds, target_names=['Negative (0)', 'Positive (1)']))

    # 5. Compare and Select Best Model
    best_model_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print('\n' + '='*50)
    print(f'  WINNING MODEL: {best_model_name}')
    print(f'  Accuracy: {best_metrics["accuracy"]:.4f} | F1-Score: {best_metrics["f1_score"]:.4f}')
    print('='*50)

    # 6. Save Artifacts
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')

    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save both models for comparative evaluations
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression_model.joblib'))
    joblib.dump(nb_model, os.path.join(models_dir, 'naive_bayes_model.joblib'))

    metadata = {
        'best_model': best_model_name,
        'metrics_summary': results,
        'vectorizer_params': {
            'ngram_range': [1, 2],
            'max_features': 10000,
            'min_df': 3,
            'sublinear_tf': True
        },
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f'\n[+] Saved Best Model -> {best_model_path}')
    print(f'[+] Saved TF-IDF Vectorizer -> {vectorizer_path}')
    print(f'[+] Saved Metadata Summary -> {metadata_path}')

    return results, metadata


if __name__ == '__main__':
    train_and_evaluate_pipeline()
