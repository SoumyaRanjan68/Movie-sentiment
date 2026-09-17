"""
Sentiment Intelligence & Text Analytics Platform
-------------------------------------------------
Enterprise-grade Natural Language Processing & Machine Learning
classification engine featuring real-time inference, Explainable AI (XAI)
token contributions, bulk review processing, and interactive model diagnostics.
"""

import os
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import SentimentPredictor

# Page Configuration
st.set_page_config(
    page_title="Sentiment Intelligence Platform | Enterprise NLP Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise-grade Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .platform-header {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    
    .platform-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.4rem;
        font-weight: 400;
    }
    
    .card-sentiment-pos {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #A7F3D0;
        border-left: 6px solid #10B981;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: #065F46;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .card-sentiment-neg {
        background: linear-gradient(135deg, #FFF1F2 0%, #FEE2E2 100%);
        border: 1px solid #FECDD3;
        border-left: 6px solid #EF4444;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: #991B1B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .token-badge-pos {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .token-badge-neg {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECDD3;
    }
    
    .stTextArea textarea {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return SentimentPredictor(models_dir='models')


@st.cache_data
def load_metadata():
    meta_path = os.path.join('models', 'model_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    # Top Platform Branding Header
    st.markdown('<div class="platform-header">⚡ Sentiment Intelligence & Text Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="platform-subtitle">Production-Grade Natural Language Processing & Machine Learning Classification Engine</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### ⚙️ System Specifications")
    st.sidebar.markdown("""
    - **Architecture**: TF-IDF + Logistic Regression
    - **Optimization**: L-BFGS (L2 Regularized, C=1.0)
    - **Feature Space**: 10,000 N-grams (1-2 words)
    - **Inference Latency**: `< 2.5 ms` (CPU-optimized)
    - **Corpus**: 50,000 Validated Benchmark Records
    """)
    
    metadata = load_metadata()
    if metadata:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Test Evaluation KPIs")
        best_name = metadata.get('best_model', 'Logistic Regression')
        metrics = metadata['metrics_summary'].get(best_name, {})
        
        col_m1, col_m2 = st.sidebar.columns(2)
        col_m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        col_m2.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        
        col_m3, col_m4 = st.sidebar.columns(2)
        col_m3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col_m4.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: **Active & Operational** 🟢 | v1.2.0")

    # Main Navigation Tabs
    tab_single, tab_batch, tab_xai, tab_pipeline, tab_visuals = st.tabs([
        "🎯 Real-Time Inference",
        "📂 Batch Analytics",
        "🔍 Model Interpretability",
        "📐 System Architecture",
        "📈 Performance Diagnostics"
    ])

    # =========================================================================
    # TAB 1: REAL-TIME INFERENCE WITH TOKEN CONTRIBUTIONS
    # =========================================================================
    with tab_single:
        st.markdown("#### Real-Time Text Classification & Feature Contribution")
        st.write("Input raw text or select one of the curated industry test queries:")
        
        # Preset Test Cases
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
        
        if p_col1.button("🌟 Masterpiece", use_container_width=True):
            st.session_state['review_input'] = "This film was an absolute masterpiece! Superb direction, compelling storytelling, and brilliant acting throughout."
        if p_col2.button("👎 Box Office Flop", use_container_width=True):
            st.session_state['review_input'] = "An absolute waste of time. The screenplay was completely incoherent, pacing dragged forever, and the acting was terrible."
        if p_col3.button("🔄 Tricky Negation", use_container_width=True):
            st.session_state['review_input'] = "I expected an engaging experience based on the trailer, but it was not good at all. Never watching it again."
        if p_col4.button("🎭 Sarcastic Tone", use_container_width=True):
            st.session_state['review_input'] = "What a glorious mess this movie was! Truly an astonishing disaster that everyone should avoid."
        if p_col5.button("💬 Viewer Feedback", use_container_width=True):
            st.session_state['review_input'] = "The cinematography was visually stunning, but the dialogue felt somewhat flat and predictable in the third act."
        if p_col6.button("⚡ Punchy Short", use_container_width=True):
            st.session_state['review_input'] = "Pure genius. Loved every single second!"
            
        initial_val = st.session_state.get('review_input', '')
        user_input = st.text_area(
            "Input Review or Feedback Text:",
            value=initial_val,
            height=125,
            placeholder="Type or paste any movie review, audience critique, or customer feedback here..."
        )
        
        char_count = len(user_input)
        word_count = len(user_input.split())
        st.caption(f"Input Telemetry: **{word_count}** words | **{char_count}** characters")
        
        col_btn, _ = st.columns([1, 4])
        analyze_clicked = col_btn.button("⚡ Run Sentiment Analysis", type="primary", use_container_width=True)
        
        if analyze_clicked or (user_input and user_input.strip() != ""):
            if not user_input.strip():
                st.warning("Please enter valid text to analyze.")
            else:
                try:
                    predictor = load_predictor()
                    result = predictor.predict(user_input)
                    
                    st.markdown("---")
                    st.markdown("#### Inference Telemetry & Decision Output")
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        if result['prediction'] == 'Positive':
                            st.markdown(f"""
                            <div class="card-sentiment-pos">
                                <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Classification Outcome</div>
                                <div style="font-size: 1.6rem; font-weight: 700; margin: 0.15rem 0;">🟢 POSITIVE</div>
                                <div style="font-size: 0.95rem; color: #047857;">Sentiment Intensity: <strong>{result['intensity']}</strong></div>
                                <div style="font-size: 0.95rem; color: #065F46; margin-top: 0.25rem;">Decision Confidence: <strong>{result['confidence']*100:.2f}%</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="card-sentiment-neg">
                                <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Classification Outcome</div>
                                <div style="font-size: 1.6rem; font-weight: 700; margin: 0.15rem 0;">🔴 NEGATIVE</div>
                                <div style="font-size: 0.95rem; color: #B91C1C;">Sentiment Intensity: <strong>{result['intensity']}</strong></div>
                                <div style="font-size: 0.95rem; color: #991B1B; margin-top: 0.25rem;">Decision Confidence: <strong>{result['confidence']*100:.2f}%</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with res_col2:
                        st.markdown("**Calibrated Class Probabilities:**")
                        st.progress(result['positive_probability'])
                        p_col1, p_col2 = st.columns(2)
                        p_col1.metric("Positive Probability", f"{result['positive_probability']*100:.2f}%")
                        p_col2.metric("Negative Probability", f"{result['negative_probability']*100:.2f}%")
                        
                    # Explainable AI (XAI) Token Contributions
                    st.markdown("##### 🔍 Local Explainability: Feature Influence on Decision")
                    xai_c1, xai_c2 = st.columns(2)
                    
                    with xai_c1:
                        st.markdown("**Words Pushing Positive (🟢):**")
                        if result['positive_contributors']:
                            for item in result['positive_contributors']:
                                st.markdown(f"<span class='token-badge-pos'>{item['term']} (+{item['impact']:.3f})</span>", unsafe_allow_html=True)
                            df_pos = pd.DataFrame(result['positive_contributors'])[['term', 'impact', 'weight']]
                            st.dataframe(df_pos.rename(columns={'term': 'Term', 'impact': 'Impact Score', 'weight': 'Model Coef'}), height=150, use_container_width=True)
                        else:
                            st.caption("No strong positive vocabulary terms identified.")
                            
                    with xai_c2:
                        st.markdown("**Words Pushing Negative (🔴):**")
                        if result['negative_contributors']:
                            for item in result['negative_contributors']:
                                st.markdown(f"<span class='token-badge-neg'>{item['term']} ({item['impact']:.3f})</span>", unsafe_allow_html=True)
                            df_neg = pd.DataFrame(result['negative_contributors'])[['term', 'impact', 'weight']]
                            st.dataframe(df_neg.rename(columns={'term': 'Term', 'impact': 'Impact Score', 'weight': 'Model Coef'}), height=150, use_container_width=True)
                        else:
                            st.caption("No strong negative vocabulary terms identified.")
                            
                    # Detailed Technical Views
                    with st.expander("🛠️ View Normalized String & Preprocessing Pipeline"):
                        st.code(result['cleaned_text'], language="text")
                        st.caption("Transformations: HTML entity unescaping, markup stripping, contraction expansion ('didn't' -> 'did not'), case folding, and non-alphanumeric removal.")
                        
                    with st.expander("📦 Raw API Response Payload (JSON)"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Inference Engine Error: {e}")

    # =========================================================================
    # TAB 2: BATCH / BULK REVIEW PROCESSING
    # =========================================================================
    with tab_batch:
        st.markdown("#### Batch Sentiment Analysis & Corpus Analytics")
        st.write("Analyze multiple reviews simultaneously using preset batches, custom CSV uploads, or multi-line text input.")
        
        batch_source = st.radio(
            "Select Data Ingestion Mode:",
            ["Curated Sample Batch (10 Reviews)", "Upload Custom CSV", "Paste Multi-Line Text"],
            horizontal=True
        )
        
        batch_reviews = []
        
        if batch_source == "Curated Sample Batch (10 Reviews)":
            batch_reviews = [
                "An incredible tour de force! Exceptional cinematography and profound emotional depth.",
                "Terrible waste of money. Completely boring, disjointed plot and awful performances.",
                "I was skeptical at first, but it was not bad at all. Thoroughly enjoyed the climax.",
                "Uninspired and dull. The director clearly lost control of the storyline halfway through.",
                "Brilliant screenplay with witty dialogue and memorable characters. Highly recommended!",
                "Painfully slow and agonizingly pretentious. Avoid at all costs.",
                "Masterpiece. One of the greatest films produced in the last decade.",
                "Decent popcorn movie with stunning CGI, though the script lacks originality.",
                "Horrible garbage. Acting was wooden and special effects looked ten years outdated.",
                "Remarkable acting by the lead cast. A touching and poignant drama."
            ]
            st.info(f"Loaded {len(batch_reviews)} curated reviews across genres.")
            
        elif batch_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Choose CSV file (must contain a column named 'review' or 'text')", type=['csv'])
            if uploaded_file:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    target_col = None
                    for c in ['review', 'text', 'comment', 'content', 'feedback']:
                        if c in df_up.columns:
                            target_col = c
                            break
                    if target_col:
                        batch_reviews = df_up[target_col].dropna().astype(str).tolist()
                        st.success(f"Extracted {len(batch_reviews)} records from column '{target_col}'.")
                    else:
                        st.error(f"Could not find a review text column. Found columns: {list(df_up.columns)}")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    
        else:
            multi_text = st.text_area(
                "Paste multiple reviews (one per line):",
                value="The movie was fantastic and exciting.\nBoring and way too long.\nNot good at all, completely disappointing.\nA true cinematic masterpiece.",
                height=140
            )
            batch_reviews = [r.strip() for r in multi_text.split('\n') if r.strip()]
            
        if st.button("🚀 Process Batch Dataset", type="primary"):
            if not batch_reviews:
                st.warning("No reviews available for processing.")
            else:
                with st.spinner("Processing batch inference through vectorized pipeline..."):
                    predictor = load_predictor()
                    batch_results = []
                    
                    for text in batch_reviews:
                        pred = predictor.predict(text)
                        batch_results.append({
                            'Review Snippet': text[:80] + ('...' if len(text) > 80 else ''),
                            'Sentiment': pred['prediction'],
                            'Intensity': pred['intensity'],
                            'Confidence': f"{pred['confidence']*100:.1f}%",
                            'P(Positive)': round(pred['positive_probability'], 3),
                            'P(Negative)': round(pred['negative_probability'], 3),
                            'Full Text': text
                        })
                        
                    df_batch = pd.DataFrame(batch_results)
                    
                    st.markdown("---")
                    st.markdown("##### Batch Analysis Summary")
                    
                    total_count = len(df_batch)
                    pos_count = (df_batch['Sentiment'] == 'Positive').sum()
                    neg_count = (df_batch['Sentiment'] == 'Negative').sum()
                    pos_ratio = (pos_count / total_count) * 100 if total_count > 0 else 0
                    
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    b_col1.metric("Total Reviews", f"{total_count:,}")
                    b_col2.metric("Positive Sentiment", f"{pos_count} ({pos_ratio:.1f}%)")
                    b_col3.metric("Negative Sentiment", f"{neg_count} ({100-pos_ratio:.1f}%)")
                    b_col4.metric("Avg Confidence", f"{df_batch['P(Positive)'].mean()*100:.1f}%")
                    
                    # Sentiment Distribution Bar
                    st.progress(pos_ratio / 100)
                    
                    # Results Table
                    st.markdown("##### Detailed Predictions Table")
                    st.dataframe(
                        df_batch[['Review Snippet', 'Sentiment', 'Intensity', 'Confidence', 'P(Positive)', 'P(Negative)']],
                        use_container_width=True,
                        height=280
                    )
                    
                    # CSV Export
                    csv_buffer = io.StringIO()
                    df_batch.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Batch Predictions (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="sentiment_predictions_output.csv",
                        mime="text/csv"
                    )

    # =========================================================================
    # TAB 3: MODEL INTERPRETABILITY (GLOBAL FEATURE IMPORTANCE)
    # =========================================================================
    with tab_xai:
        st.markdown("#### Global Model Interpretability & Vocabulary Weights")
        st.write("Inspect the most discriminative vocabulary terms learned by the Logistic Regression classifier across the 10,000-term feature space.")
        
        try:
            predictor = load_predictor()
            top_pos, top_neg = predictor.get_global_top_features(top_n=15)
            
            xai_g1, xai_g2 = st.columns(2)
            
            with xai_g1:
                st.markdown("##### 🟢 Top 15 Positive Discriminators")
                df_top_pos = pd.DataFrame(top_pos)
                st.dataframe(df_top_pos.rename(columns={'term': 'Keyword / Bigram', 'coefficient': 'Positive Weight'}), use_container_width=True)
                
                # Plot
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_top_pos, x='coefficient', y='term', palette='Greens_r', ax=ax)
                ax.set_title("Top Positive Coefficients", fontweight='bold')
                ax.set_xlabel("Logistic Regression Weight (Log-Odds)")
                plt.tight_layout()
                st.pyplot(fig)
                
            with xai_g2:
                st.markdown("##### 🔴 Top 15 Negative Discriminators")
                df_top_neg = pd.DataFrame(top_neg)
                st.dataframe(df_top_neg.rename(columns={'term': 'Keyword / Bigram', 'coefficient': 'Negative Weight'}), use_container_width=True)
                
                # Plot
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_top_neg, x='coefficient', y='term', palette='Reds', ax=ax)
                ax.set_title("Top Negative Coefficients", fontweight='bold')
                ax.set_xlabel("Logistic Regression Weight (Log-Odds)")
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error generating interpretability charts: {e}")

    # =========================================================================
    # TAB 4: SYSTEM ARCHITECTURE
    # =========================================================================
    with tab_pipeline:
        st.markdown("#### End-to-End System Architecture")
        st.markdown("""
        ```
        Raw Unstructured Input Text
                    │
                    ▼
        [Stage 1: Deterministic Preprocessing]
          • HTML entity resolution (&amp; -> &) & markup excision (<br />, <p>)
          • URL removal (http/https/www)
          • Contraction expansion (e.g., didn't -> did not, preventing negation loss)
          • Case folding to lowercase & non-alphanumeric symbol removal
                    │
                    ▼
        [Stage 2: High-Dimensional TF-IDF Feature Extraction]
          • Unigram & Bigram combinations: ngram_range=(1, 2)
          • Vocabulary constraint: Top 10,000 most informative features
          • Sublinear Term Frequency scaling: 1 + log(tf)
          • Document frequency thresholding: min_df=3
                    │
                    ▼
        [Stage 3: Statistical Classifier]
          • L2-Regularized Logistic Regression (C=1.0, L-BFGS solver)
          • Benchmark Comparator: Multinomial Naive Bayes (Laplace α=1.0)
                    │
                    ▼
        [Stage 4: Probability Estimation & Decision]
          • Sigmoid activation: P(y=1|x) = 1 / (1 + e^(-z))
          • Decision Boundary at P=0.50
          • Granular Sentiment Intensity Tiers (Strong / Moderate / Borderline)
        ```
        
        #### Core Engineering Highlights:
        1. **Strict Data Leakage Isolation**: The TF-IDF vectorizer vocabulary and IDF weights were fitted exclusively on the 80% training partition (`fit_transform`) and applied as a projection operator (`transform`) on test and inference data.
        2. **Explainability Over Black-Boxes**: Every prediction is fully traceable to individual word coefficients ($w_i \times x_i$).
        3. **Sub-3ms Inference Latency**: Designed for high-throughput deployment without GPU infrastructure cost.
        """)

    # =========================================================================
    # TAB 5: PERFORMANCE DIAGNOSTICS & EVALUATION CHARTS
    # =========================================================================
    with tab_visuals:
        st.markdown("#### Empirical Evaluation Diagnostics & Statistical Visualizations")
        
        fig_col1, fig_col2 = st.columns(2)
        
        comp_path = 'reports/figures/model_comparison_metrics.png'
        cm_lr_path = 'reports/figures/confusion_matrix_logistic_regression.png'
        cm_nb_path = 'reports/figures/confusion_matrix_naive_bayes.png'
        dist_path = 'reports/figures/class_distribution.png'
        len_path = 'reports/figures/review_length_distribution.png'
        
        with fig_col1:
            if os.path.exists(comp_path):
                st.image(comp_path, caption="Comparative Metric Performance (Holdout Test Set)", width="stretch")
            if os.path.exists(dist_path):
                st.image(dist_path, caption="Class Balance Distribution (50,000 Records)", width="stretch")
                
        with fig_col2:
            if os.path.exists(cm_lr_path):
                st.image(cm_lr_path, caption="Confusion Matrix - Logistic Regression (Winning Model)", width="stretch")
            if os.path.exists(len_path):
                st.image(len_path, caption="Word Count Distribution by Sentiment Class", width="stretch")


if __name__ == '__main__':
    main()
