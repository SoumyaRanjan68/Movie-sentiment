"""
Model Evaluation and Visualization Module
-----------------------------------------
Objective:
1. Load test data, trained models, and vectorizer.
2. Generate Confusion Matrix plots for Logistic Regression and Naive Bayes.
3. Generate Comparative Performance Metric charts.
4. Generate EDA visualizations (Class distribution, review length distribution).
5. Save all plots to reports/figures/.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src.data_preprocessing import clean_text

# Set consistent aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def plot_eda_figures(df: pd.DataFrame, figures_dir: str = 'reports/figures'):
    """
    Creates and saves clear, interview-ready EDA visualizations.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x='sentiment',
        data=df,
        order=['positive', 'negative'],
        palette={'positive': '#2ecc71', 'negative': '#e74c3c'}
    )
    plt.title('Movie Review Sentiment Class Distribution (Target Variable)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Sentiment Class', fontsize=11)
    plt.ylabel('Review Count', fontsize=11)
    
    # Add count labels on bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    plt.tight_layout()
    dist_path = os.path.join(figures_dir, 'class_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f'[+] Saved EDA Plot -> {dist_path}')

    # 2. Word Count Distribution by Sentiment
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df[df['word_count'] <= 600],
        x='word_count',
        hue='sentiment',
        kde=True,
        bins=35,
        palette={'positive': '#2ecc71', 'negative': '#e74c3c'},
        alpha=0.45
    )
    plt.title('Distribution of Review Word Counts by Sentiment Class', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Word Count (Cleaned Review)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.tight_layout()
    len_path = os.path.join(figures_dir, 'review_length_distribution.png')
    plt.savefig(len_path, dpi=300)
    plt.close()
    print(f'[+] Saved EDA Plot -> {len_path}')


def plot_confusion_matrix_custom(y_true, y_pred, model_name: str, save_path: str):
    """
    Generates an annotated, readable confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted Negative (0)', 'Predicted Positive (1)'],
        yticklabels=['Actual Negative (0)', 'Actual Positive (1)'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Ground Truth Label', fontsize=11)
    plt.xlabel('Model Prediction', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'[+] Saved Confusion Matrix -> {save_path}')


def plot_model_comparison(metrics_dict: dict, save_path: str):
    """
    Generates a comparative bar chart comparing Accuracy, Precision, Recall, and F1-Score.
    """
    df_metrics = pd.DataFrame(metrics_dict).T.reset_index()
    df_metrics = df_metrics.rename(columns={'index': 'Model'})
    
    df_melted = pd.melt(
        df_metrics,
        id_vars=['Model'],
        value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
        var_name='Metric',
        value_name='Score'
    )
    df_melted['Metric'] = df_melted['Metric'].map({
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    })
    
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(
        data=df_melted,
        x='Metric',
        y='Score',
        hue='Model',
        palette=['#3498db', '#9b59b6']
    )
    plt.title('Model Performance Comparison: Logistic Regression vs. Naive Bayes', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
    plt.xlabel('Evaluation Metric', fontsize=11)
    plt.ylim(0.70, 1.0)
    
    # Annotate values on bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height + 0.005),
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            
    plt.legend(title='Algorithm', loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'[+] Saved Model Comparison Plot -> {save_path}')


def run_full_evaluation(
    data_path: str = 'data/processed/cleaned_reviews.csv',
    models_dir: str = 'models',
    figures_dir: str = 'reports/figures'
):
    """
    Loads models, computes metrics on stratified test split, and generates all visual reports.
    """
    print('[*] Starting Full Evaluation and Report Generation...')
    df = pd.read_csv(data_path)
    df['cleaned_review'] = df['cleaned_review'].fillna('')
    
    # 1. Generate EDA Figures
    plot_eda_figures(df, figures_dir)
    
    # 2. Re-create stratified test split
    X = df['cleaned_review']
    y = df['sentiment_label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 3. Load Vectorizer and Models
    vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.joblib'))
    nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.joblib'))
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Predictions and Confusion Matrices
    lr_preds = lr_model.predict(X_test_tfidf)
    nb_preds = nb_model.predict(X_test_tfidf)
    
    plot_confusion_matrix_custom(
        y_test, lr_preds,
        model_name='Logistic Regression',
        save_path=os.path.join(figures_dir, 'confusion_matrix_logistic_regression.png')
    )
    
    plot_confusion_matrix_custom(
        y_test, nb_preds,
        model_name='Multinomial Naive Bayes',
        save_path=os.path.join(figures_dir, 'confusion_matrix_naive_bayes.png')
    )
    
    # 5. Comparative Metrics Chart
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            plot_model_comparison(
                meta['metrics_summary'],
                save_path=os.path.join(figures_dir, 'model_comparison_metrics.png')
            )
            
    print('[+] Full Evaluation & Visual Figures successfully completed.')


if __name__ == '__main__':
    run_full_evaluation()
