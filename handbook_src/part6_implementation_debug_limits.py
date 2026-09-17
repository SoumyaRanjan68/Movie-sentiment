"""
Part 6: End-to-End Implementation, Debugging, and Limitations
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 16 &mdash; END-TO-END MANUAL IMPLEMENTATION FROM SCRATCH (15 STEPS)</h1>
    <p><em>Follow this exact chronological roadmap if you ever need to rebuild this project from absolute zero during an interview live-coding assessment or local setup:</em></p>

    <h2>Step 1: Environment & Directory Setup</h2>
    <pre><code># Create clean directory structure:
mkdir -p movie-sentiment-analysis/{data/raw,data/processed,models,reports/figures,src,notebooks}
cd movie-sentiment-analysis</code></pre>

    <h2>Step 2: Create Dependencies (requirements.txt)</h2>
    <pre><code>pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.28.0</code></pre>

    <h2>Step 3: Download Benchmark Dataset</h2>
    <pre><code>import urllib.request
url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
urllib.request.urlretrieve(url, "data/raw/imdb_reviews.csv")</code></pre>

    <h2>Step 4: Inspect Raw Data Quality</h2>
    <pre><code>import pandas as pd
df = pd.read_csv("data/raw/imdb_reviews.csv")
print("Shape:", df.shape)
print("Missing:", df.isnull().sum())
print("Duplicates:", df.duplicated(subset=['review']).sum())</code></pre>
    <p><strong>Expected Output:</strong> Shape: (50000, 2), Missing: 0, Duplicates: 418.</p>

    <h2>Step 5: Implement Text Cleaning (src/data_preprocessing.py)</h2>
    <pre><code>import re, html

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Contraction expansion preserves "not"
    text = re.sub(r"\bdon't\b", "do not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwasn't\b", "was not", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()</code></pre>

    <h2>Step 6: Deduplicate and Save Processed Data</h2>
    <pre><code>df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
df['cleaned_review'] = df['review'].apply(clean_text)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df.to_csv("data/processed/cleaned_reviews.csv", index=False)</code></pre>

    <h2>Step 7: Perform Stratified Train/Test Split</h2>
    <pre><code>from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment_label'],
    test_size=0.20, random_state=42, stratify=df['sentiment_label']
)</code></pre>

    <h2>Step 8: Fit TF-IDF Feature Extraction Matrix</h2>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)</code></pre>

    <h2>Step 9: Train Classifiers (Logistic Regression & Naive Bayes)</h2>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)</code></pre>

    <h2>Step 10: Evaluate Models on Holdout Test Set</h2>
    <pre><code>from sklearn.metrics import accuracy_score, classification_report
lr_preds = lr_model.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))</code></pre>

    <h2>Step 11: Serialize Best Model and Vectorizer to Disk</h2>
    <pre><code>import joblib
joblib.dump(lr_model, "models/best_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")</code></pre>

    <h2>Step 12: Build Predictor Inference Class (src/predict.py)</h2>
    <pre><code>class SentimentPredictor:
    def __init__(self, models_dir='models'):
        self.model = joblib.load(f"{models_dir}/best_model.joblib")
        self.vectorizer = joblib.load(f"{models_dir}/tfidf_vectorizer.joblib")
    def predict(self, text):
        clean = clean_text(text)
        vec = self.vectorizer.transform([clean])
        prob = self.model.predict_proba(vec)[0]
        label = self.model.predict(vec)[0]
        return {'sentiment': 'Positive' if label == 1 else 'Negative', 'confidence': float(prob[label])}</code></pre>

    <h2>Step 13: Build Streamlit Web Interface (app.py)</h2>
    <pre><code>import streamlit as st
from src.predict import SentimentPredictor

st.title("Sentiment Intelligence Platform")
text = st.text_area("Enter review:")
if st.button("Analyze"):
    pred = SentimentPredictor().predict(text)
    st.write("Sentiment:", pred['sentiment'])
    st.write("Confidence:", f"{pred['confidence']*100:.2f}%")</code></pre>

    <h2>Step 14: Execute and Test Locally</h2>
    <pre><code>streamlit run app.py --server.port 8502</code></pre>

    <h2>Step 15: Push to GitHub and Deploy</h2>
    <pre><code>git init -b main
git add .
git commit -m "Deploy production sentiment system"
git remote add origin https://github.com/&lt;user&gt;/&lt;repo&gt;.git
git push -u origin main</code></pre>
</div>

<div class="page-break-before">
    <h1>PART 17 &mdash; DEBUGGING & TROUBLESHOOTING GUIDE</h1>

    <table>
        <tr>
            <th>Common Error</th>
            <th>Root Cause</th>
            <th>Diagnostic Test</th>
            <th>Permanent Fix</th>
        </tr>
        <tr>
            <td><code>FileNotFoundError: models/best_model.joblib</code></td>
            <td>Inference script executed before training was run.</td>
            <td>Check <code>os.path.exists('models/best_model.joblib')</code>.</td>
            <td>Run <code>python src/train_model.py</code> to train and serialize weights.</td>
        </tr>
        <tr>
            <td><code>ValueError: dimension mismatch</code></td>
            <td>Vectorizer was re-fitted with a different <code>max_features</code> or different corpus.</td>
            <td>Inspect <code>tfidf_vec.shape[1]</code> vs <code>model.coef_.shape[1]</code>.</td>
            <td>Ensure the exact same saved <code>tfidf_vectorizer.joblib</code> is loaded alongside the model.</td>
        </tr>
        <tr>
            <td><code>SyntaxError: Port 8501 is not available</code></td>
            <td>Another process or terminal session is already running Streamlit on default port 8501.</td>
            <td>Run <code>netstat -ano | findstr :8501</code>.</td>
            <td>Launch on an alternate port: <code>streamlit run app.py --server.port 8502</code>.</td>
        </tr>
        <tr>
            <td><code>UnicodeDecodeError: 'utf-8' codec can't decode...</code></td>
            <td>Text file contains Windows-1252 or non-UTF8 byte sequences.</td>
            <td>Check file encoding using <code>chardet</code>.</td>
            <td>Always specify <code>encoding='utf-8'</code> when calling <code>open()</code> or <code>read_csv()</code>.</td>
        </tr>
        <tr>
            <td><code>Empty / Whitespace Input crashes predict()</code></td>
            <td>User clicked analyze on an empty string, yielding zero tokens.</td>
            <td>Check <code>if not review_text.strip():</code></td>
            <td>Add defensive validation returning neutral/undetermined response without invoking model.</td>
        </tr>
        <tr>
            <td><code>MemoryError during fit_transform()</code></td>
            <td><code>ngram_range=(1, 3)</code> without <code>max_features</code> generated 1,000,000+ features.</td>
            <td>Monitor RAM consumption during vectorization.</td>
            <td>Always set <code>max_features=10000</code> and <code>min_df=3</code> to cap dimensional growth.</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 18 &mdash; REAL-WORLD LIMITATIONS & FUTURE ROADMAP</h1>

    <h2>1. The Four Inherent Limitations of Classical N-gram Models</h2>
    <ol>
        <li><strong>Sarcasm & Verbal Irony:</strong>
            <p>Consider the review: <em>"What a glorious, stunning waste of my precious two hours."</em></p>
            <p>Our model identifies positive unigrams <code>"glorious"</code> (+1.8) and <code>"stunning"</code> (+1.6), and negative unigram <code>"waste"</code> (-2.1). Because positive words outweigh the negative term numerically, the review is misclassified as Positive. Classical models cannot comprehend tonal irony.</p>
        </li>
        <li><strong>Long-Distance Negation:</strong>
            <p>Consider: <em>"I did not, even after giving the director multiple opportunities across three agonizing hours, find anything to appreciate."</em></p>
            <p>Because the word <code>"not"</code> is separated from <code>"find"</code> or <code>"appreciate"</code> by 12 intervening words, our bigram window ($N=2$) fails to bind the negation to the sentiment verb.</p>
        </li>
        <li><strong>Out-of-Vocabulary (OOV) Slang:</strong>
            <p>If a teenager reviews a film using new internet slang: <em>"This movie is totally skibidi and mid,"</em> the words do not exist in the 10,000-word IMDB vocabulary and receive a score of 0.</p>
        </li>
        <li><strong>Domain Specificity:</strong>
            <p>A model trained on IMDB movie reviews understands words like "cinematography" and "screenplay". If applied directly to Tata Power electrical distribution complaints ("feeder line tripped", "transformer overheating"), it will struggle because technical electrical vocabulary was not present in movie reviews.</p>
        </li>
    </ol>

    <h2>2. Future Engineering Improvements (Next-Gen Roadmap)</h2>
    <ul>
        <li><strong>Word Embeddings (Word2Vec / FastText):</strong> Represents words as dense 300-dimensional vectors where semantic similarity is geometric ($v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$). FastText uses character n-grams to handle OOV words and typos.</li>
        <li><strong>Bidirectional LSTMs / GRUs:</strong> Recurrent neural architectures that process words sequentially with internal memory gates, resolving long-distance dependencies.</li>
        <li><strong>Transformer Models (DistilBERT / RoBERTa):</strong> Self-Attention mechanisms where every word attends to every other word simultaneously, fully solving long-distance negation and capturing nuanced contextual semantics at the cost of higher compute latency.</li>
    </ul>
</div>
"""
