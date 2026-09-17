"""
Prediction and Inference Pipeline with Model Explainability
------------------------------------------------------------
Objective:
Provides a clean, production-ready inference engine that accepts
any raw text, applies the exact same preprocessing and vectorization pipeline,
and outputs predicted sentiment, confidence probabilities, sentiment intensity,
and local feature contributions (Explainable AI / XAI).
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from typing import Dict, Any, List, Tuple

from src.data_preprocessing import clean_text


class SentimentPredictor:
    """
    Production-ready Sentiment Predictor encapsulating preprocessing,
    vectorization, calibrated inference, and token-level explainability.
    """
    def __init__(self, models_dir: str = 'models'):
        self.model_path = os.path.join(models_dir, 'best_model.joblib')
        self.vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Model files missing in '{models_dir}'. Please run 'python src/train_model.py' first."
            )
            
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.coefficients = self.model.coef_[0] if hasattr(self.model, 'coef_') else None
        self.intercept = float(self.model.intercept_[0]) if hasattr(self.model, 'intercept_') else 0.0

    @staticmethod
    def calculate_intensity(pos_prob: float) -> Tuple[str, str]:
        """
        Categorizes prediction confidence into granular sentiment tiers.
        """
        if pos_prob >= 0.85:
            return "Strongly Positive", "#10B981"
        elif pos_prob >= 0.60:
            return "Moderately Positive", "#34D399"
        elif pos_prob > 0.45:
            return "Borderline / Mixed", "#F59E0B"
        elif pos_prob >= 0.20:
            return "Moderately Negative", "#F87171"
        else:
            return "Strongly Negative", "#EF4444"

    def predict(self, review_text: str) -> Dict[str, Any]:
        """
        Performs end-to-end inference on raw input text.
        """
        if not review_text or not isinstance(review_text, str) or not review_text.strip():
            return {
                'raw_text': review_text,
                'cleaned_text': '',
                'prediction': 'Neutral / Undetermined',
                'confidence': 0.0,
                'positive_probability': 0.5,
                'negative_probability': 0.5,
                'intensity': 'Neutral / Empty Input',
                'intensity_color': '#64748B',
                'token_contributions': [],
                'positive_contributors': [],
                'negative_contributors': [],
                'note': 'Empty or whitespace-only input.'
            }
            
        # 1. Exact same text normalization
        cleaned = clean_text(review_text)
        
        # 2. Vectorize via fitted TF-IDF
        tfidf_vec = self.vectorizer.transform([cleaned])
        
        # 3. Model classification and probabilities
        pred_label = self.model.predict(tfidf_vec)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(tfidf_vec)[0]
            neg_prob = float(probs[0])
            pos_prob = float(probs[1])
        else:
            score = self.model.decision_function(tfidf_vec)[0]
            pos_prob = float(1 / (1 + np.exp(-score)))
            neg_prob = 1.0 - pos_prob
            
        sentiment = 'Positive' if pred_label == 1 else 'Negative'
        confidence = pos_prob if pred_label == 1 else neg_prob
        intensity, intensity_color = self.calculate_intensity(pos_prob)
        
        # 4. Token-level explainability (Feature Contributions)
        pos_contribs, neg_contribs = self._compute_feature_contributions(tfidf_vec)
        
        return {
            'raw_text': review_text,
            'cleaned_text': cleaned,
            'prediction': sentiment,
            'confidence': confidence,
            'positive_probability': pos_prob,
            'negative_probability': neg_prob,
            'intensity': intensity,
            'intensity_color': intensity_color,
            'positive_contributors': pos_contribs,
            'negative_contributors': neg_contribs
        }

    def _compute_feature_contributions(self, tfidf_vec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculates local term contributions (TF-IDF value * Model Coefficient)
        explaining which words drove the prediction.
        """
        if self.coefficients is None:
            return [], []
            
        indices = tfidf_vec.nonzero()[1]
        pos_list = []
        neg_list = []
        
        for idx in indices:
            word = self.feature_names[idx]
            tfidf_val = float(tfidf_vec[0, idx])
            weight = float(self.coefficients[idx])
            impact = tfidf_val * weight
            
            item = {
                'term': word,
                'impact': round(impact, 4),
                'weight': round(weight, 3),
                'tfidf': round(tfidf_val, 4)
            }
            
            if impact > 0:
                pos_list.append(item)
            elif impact < 0:
                neg_list.append(item)
                
        pos_list.sort(key=lambda x: x['impact'], reverse=True)
        neg_list.sort(key=lambda x: x['impact'])  # Most negative first
        
        return pos_list[:8], neg_list[:8]

    def get_global_top_features(self, top_n: int = 20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns the top globally positive and negative terms from the trained model weights.
        """
        if self.coefficients is None:
            return [], []
            
        top_pos_idx = np.argsort(self.coefficients)[-top_n:][::-1]
        top_neg_idx = np.argsort(self.coefficients)[:top_n]
        
        pos_features = [
            {'term': self.feature_names[i], 'coefficient': round(float(self.coefficients[i]), 3)}
            for i in top_pos_idx
        ]
        neg_features = [
            {'term': self.feature_names[i], 'coefficient': round(float(self.coefficients[i]), 3)}
            for i in top_neg_idx
        ]
        
        return pos_features, neg_features


def predict_sentiment(review_text: str, models_dir: str = 'models') -> Dict[str, Any]:
    """
    Functional helper API.
    """
    predictor = SentimentPredictor(models_dir=models_dir)
    return predictor.predict(review_text)


if __name__ == '__main__':
    predictor = SentimentPredictor()
    sample = "The direction was an absolute masterpiece, but the pacing was not good and rather slow."
    res = predictor.predict(sample)
    
    print(f"Prediction: {res['prediction']} ({res['intensity']})")
    print(f"Confidence: {res['confidence']*100:.2f}%")
    print("\nTop Positive Contributors:")
    for item in res['positive_contributors']:
        print(f"  + {item['term']}: impact = {item['impact']} (w = {item['weight']})")
    print("\nTop Negative Contributors:")
    for item in res['negative_contributors']:
        print(f"  - {item['term']}: impact = {item['impact']} (w = {item['weight']})")
