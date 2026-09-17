"""
Part 5: Project Pipeline, Line-by-Line Code Breakdown, Streamlit & Serialization
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 10 &mdash; THE PROJECT DATASET: IMDB BENCHMARK</h1>

    <h2>1. Dataset Specification & Integrity Checks</h2>
    <table>
        <tr>
            <th>Property</th>
            <th>Raw State</th>
            <th>After Ingestion & Quality Control</th>
            <th>Engineering Action Taken</th>
        </tr>
        <tr>
            <td><strong>Dataset Name</strong></td>
            <td>IMDB Large Movie Review Dataset</td>
            <td>Cleaned Benchmark Corpus</td>
            <td>Benchmark standard used in Stanford AI research.</td>
        </tr>
        <tr>
            <td><strong>Total Record Count</strong></td>
            <td>50,000 reviews</td>
            <td><strong>49,582 reviews</strong></td>
            <td><strong>418 exact duplicates removed</strong> via <code>drop_duplicates(subset=['review'])</code> to eliminate cross-split data leakage.</td>
        </tr>
        <tr>
            <td><strong>Missing Values (NaN)</strong></td>
            <td>0 missing values</td>
            <td>0 missing values</td>
            <td>Verified using <code>df.isnull().sum().sum() == 0</code>.</td>
        </tr>
        <tr>
            <td><strong>Target Feature</strong></td>
            <td><code>sentiment</code> ("positive", "negative")</td>
            <td><code>sentiment_label</code> (1 = positive, 0 = negative)</td>
            <td>Mapped to binary integers for Scikit-Learn classifiers.</td>
        </tr>
        <tr>
            <td><strong>Class Balance</strong></td>
            <td>25,000 Pos / 25,000 Neg</td>
            <td><strong>24,884 Pos (50.19%) / 24,698 Neg (49.81%)</strong></td>
            <td>Near-perfect 50:50 balance. No synthetic oversampling (SMOTE) or class re-weighting necessary.</td>
        </tr>
        <tr>
            <td><strong>Average Review Length</strong></td>
            <td>~230 words per review</td>
            <td>Cleaned review length: ~225 words</td>
            <td>Sufficient text volume to extract informative unigram and bigram features.</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 11 &mdash; END-TO-END PROJECT PIPELINE ARCHITECTURE</h1>

    <h2>1. The Complete Architectural Flowchart</h2>
    <pre><code>[Raw IMDB CSV Dataset: 50,000 Reviews]
                  │
                  ▼
[1. Data Ingestion & Quality Checks] ──> (418 Duplicates Dropped -> 49,582 Clean Records)
                  │
                  ▼
[2. Text Preprocessing Engine] ───────> (HTML Stripped, Contractions Expanded, Lowercased, Cleaned)
                  │
                  ▼
[3. Stratified Partitioning (80/20)] ─> (X_train: 39,665 Samples | X_test: 9,917 Samples)
                  │
                  ▼
[4. TF-IDF Feature Extraction] ───────> (fit_transform on X_train ONLY -> 10,000 N-gram Features)
                  │
                  ├──────────────────────────────┐
                  ▼                              ▼
    [Model A: Logistic Regression]    [Model B: Multinomial Naive Bayes]
                  │                              │
                  ▼                              ▼
    Accuracy: 90.23% | F1: 90.38%     Accuracy: 86.95% | F1: 87.18%
                  │
                  ▼
[5. Model Selection & Serialization] ──> (Saved to models/best_model.joblib & tfidf_vectorizer.joblib)
                  │
                  ▼
[6. Real-Time Inference Engine] ───────> (src/predict.py: Preprocessing -> TF-IDF -> Probability & XAI)
                  │
                  ▼
[7. Production Streamlit Web App] ─────> (Real-Time Inference, Batch Processing, XAI & Diagnostics)</code></pre>

    <h2>2. Step-by-Step Data Transformation Matrix</h2>
    <table>
        <tr>
            <th>Pipeline Step</th>
            <th>What Enters</th>
            <th>What Exits</th>
            <th>Python Tool / Library</th>
            <th>Implementation File</th>
        </tr>
        <tr>
            <td><strong>Data Ingestion</strong></td>
            <td>Raw CSV on disk</td>
            <td>Cleaned Pandas DataFrame</td>
            <td><code>pandas.read_csv()</code></td>
            <td><code>src/data_preprocessing.py</code></td>
        </tr>
        <tr>
            <td><strong>Text Cleaning</strong></td>
            <td>Raw messy text with HTML</td>
            <td>Normalized clean text string</td>
            <td><code>re</code>, <code>html</code></td>
            <td><code>src/data_preprocessing.py</code></td>
        </tr>
        <tr>
            <td><strong>Dataset Splitting</strong></td>
            <td>Full text Series $X$ and labels $y$</td>
            <td>$X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$</td>
            <td><code>sklearn.model_selection.train_test_split</code></td>
            <td><code>src/train_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Feature Extraction</strong></td>
            <td>Text strings in $X_{\text{train}}$</td>
            <td>Sparse Matrix $(39665, 10000)$</td>
            <td><code>sklearn.feature_extraction.text.TfidfVectorizer</code></td>
            <td><code>src/train_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Model Training</strong></td>
            <td>Sparse Matrix & target labels</td>
            <td>Trained estimator coefficients</td>
            <td><code>sklearn.linear_model.LogisticRegression</code></td>
            <td><code>src/train_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Serialization</strong></td>
            <td>In-memory model objects</td>
            <td><code>.joblib</code> byte files on disk</td>
            <td><code>joblib.dump()</code></td>
            <td><code>src/train_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Inference & XAI</strong></td>
            <td>New raw review from user</td>
            <td>Sentiment + Probability + Token impacts</td>
            <td><code>src.predict.SentimentPredictor</code></td>
            <td><code>src/predict.py</code></td>
        </tr>
        <tr>
            <td><strong>User Interface</strong></td>
            <td>User browser interactions</td>
            <td>Interactive web analytics dashboard</td>
            <td><code>streamlit</code></td>
            <td><code>app.py</code></td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 12 &mdash; ACTUAL SOURCE CODE LINE-BY-LINE BREAKDOWN</h1>

    <h2>File 1: <code>src/data_preprocessing.py</code></h2>
    <pre><code># SECTION A: Contraction Expansion Regex
CONTRACTIONS_REGEX = re.compile(
    r'\b(' + r'|'.join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r')\b',
    flags=re.IGNORECASE
)

def expand_contractions(text: str) -> str:
    def replace(match):
        matched_text = match.group(0).lower()
        return CONTRACTION_MAP.get(matched_text, matched_text)
    return CONTRACTIONS_REGEX.sub(replace, text)</code></pre>
    <ul>
        <li><code>re.compile(...)</code>: Pre-compiles the regular expression pattern containing all 50+ contractions (like "didn't", "can't") for high execution speed.</li>
        <li><code>\b...\b</code>: Matches only complete word boundaries (prevents partial matches inside words).</li>
        <li><code>re.IGNORECASE</code>: Matches contractions regardless of capitalization (e.g., "Didn't" and "didn't").</li>
        <li><code>CONTRACTIONS_REGEX.sub(replace, text)</code>: Searches the text string and replaces each match with its expanded equivalent from <code>CONTRACTION_MAP</code>.</li>
    </ul>

    <pre><code># SECTION B: Core clean_text Pipeline
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = html.unescape(text)                   # Line 1: Converts &amp; -> &, &quot; -> "
    text = HTML_TAG_REGEX.sub(' ', text)        # Line 2: Removes &lt;br /&gt;, &lt;p&gt; tags
    text = URL_REGEX.sub(' ', text)             # Line 3: Strips web links (http, www)
    text = expand_contractions(text)            # Line 4: Expands "didn't" -> "did not"
    text = text.lower()                         # Line 5: Normalizes casing
    text = NON_ALPHANUM_REGEX.sub(' ', text)    # Line 6: Removes non-alphanumeric chars
    text = WHITESPACE_REGEX.sub(' ', text).strip() # Line 7: Collapses multi-spaces
    return text</code></pre>

    <h2>File 2: <code>src/train_model.py</code></h2>
    <pre><code># SECTION A: Fitting TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=3,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)</code></pre>
    <ul>
        <li><code>fit_transform(X_train)</code>: Scans <code>X_train</code>, builds the 10,000-word vocabulary dictionary, calculates IDF weights for each term, and projects <code>X_train</code> into a sparse matrix.</li>
        <li><code>transform(X_test)</code>: Uses the <strong>pre-existing training vocabulary and IDF weights</strong> to project <code>X_test</code>. It does NOT update vocabulary or IDF weights (prevents data leakage).</li>
    </ul>

    <pre><code># SECTION B: Logistic Regression Model Fitting
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)</code></pre>
    <ul>
        <li><code>LogisticRegression(C=1.0, ...)</code>: Instantiates the classifier with L2 regularization strength $C=1.0$ and L-BFGS solver.</li>
        <li><code>max_iter=1000</code>: Allows up to 1,000 gradient descent iterations to guarantee mathematical convergence.</li>
        <li><code>fit(X_train_tfidf, y_train)</code>: Optimizes the 10,000 feature coefficients and 1 bias intercept by minimizing binary cross-entropy loss.</li>
        <li><code>predict(X_test_tfidf)</code>: Evaluates all 9,917 holdout test reviews by computing $z = \mathbf{w}^T \mathbf{x} + b$ and applying threshold $z \ge 0$.</li>
    </ul>

    <h2>File 3: <code>src/predict.py</code></h2>
    <pre><code># SECTION A: Probability & Local Token Contributions
probs = self.model.predict_proba(tfidf_vec)[0]
pos_prob = float(probs[1])
neg_prob = float(probs[0])
sentiment = 'Positive' if pred_label == 1 else 'Negative'
confidence = pos_prob if pred_label == 1 else neg_prob

# Local XAI computation:
indices = tfidf_vec.nonzero()[1]
for idx in indices:
    word = self.feature_names[idx]
    tfidf_val = float(tfidf_vec[0, idx])
    weight = float(self.coefficients[idx])
    impact = tfidf_val * weight</code></pre>
    <ul>
        <li><code>self.model.predict_proba(...)</code>: Returns the calibrated class probabilities $[P(\text{Negative}), P(\text{Positive})]$.</li>
        <li><code>tfidf_vec.nonzero()[1]</code>: Extracts only the active column indices in the sparse vector for the input review.</li>
        <li><code>impact = tfidf_val * weight</code>: Computes the exact mathematical contribution of that specific word to the total log-odds decision score. If positive, it pushed toward Positive; if negative, toward Negative.</li>
    </ul>
</div>

<div class="page-break-before">
    <h1>PART 13 &mdash; CODE-TO-THEORY MAPPING REFERENCE MATRIX</h1>

    <table>
        <tr>
            <th>Actual Project Code Snippet</th>
            <th>Underlying Machine Learning / NLP Concept</th>
            <th>Why This Code Exists</th>
            <th>How to Explain in an Interview</th>
        </tr>
        <tr>
            <td><code>drop_duplicates(subset=['review'])</code></td>
            <td>Data Cleaning & Deduplication</td>
            <td>Removes 418 identical reviews from raw dataset.</td>
            <td>"Deduplication prevents identical reviews from appearing in both train and test splits, eliminating synthetic data leakage."</td>
        </tr>
        <tr>
            <td><code>train_test_split(..., stratify=y)</code></td>
            <td>Stratified Sampling</td>
            <td>Preserves 50.2% : 49.8% class balance across train and test partitions.</td>
            <td>"Stratification prevents class imbalance skew between training and evaluation splits."</td>
        </tr>
        <tr>
            <td><code>ngram_range=(1, 2)</code></td>
            <td>N-gram Feature Extraction</td>
            <td>Extracts single words and 2-word pairs (e.g. "not good").</td>
            <td>"Bigrams capture immediate local negation without needing deep recurrent neural networks."</td>
        </tr>
        <tr>
            <td><code>sublinear_tf=True</code></td>
            <td>Logarithmic Term Frequency Scaling</td>
            <td>Replaces raw count with $1 + \log(\text{tf})$.</td>
            <td>"Dampens the influence of repeated words so a word appearing 15 times doesn't dominate 15x over a single occurrence."</td>
        </tr>
        <tr>
            <td><code>fit_transform(X_train)</code></td>
            <td>Feature Matrix Parameter Fitting</td>
            <td>Learns vocabulary and IDF weights exclusively on training data.</td>
            <td>"Strictly isolates the test set to guarantee zero data leakage."</td>
        </tr>
        <tr>
            <td><code>LogisticRegression(C=1.0)</code></td>
            <td>L2-Regularized Linear Classification</td>
            <td>Learns linear decision boundary with Ridge penalty.</td>
            <td>"Provides fast convex optimization with guaranteed convergence and transparent word coefficients."</td>
        </tr>
        <tr>
            <td><code>MultinomialNB(alpha=1.0)</code></td>
            <td>Bayesian Probabilistic Classification</td>
            <td>Provides baseline benchmark comparison.</td>
            <td>"Uses Laplace smoothing ($\alpha=1.0$) to prevent zero probability on unseen words."</td>
        </tr>
        <tr>
            <td><code>joblib.dump(model, '...')</code></td>
            <td>Model Serialization</td>
            <td>Saves trained model weights to disk in binary format.</td>
            <td>"Allows sub-5ms real-time inference in production without needing to retrain on startup."</td>
        </tr>
        <tr>
            <td><code>@st.cache_resource</code></td>
            <td>Application Resource Caching</td>
            <td>Caches loaded model in server memory across sessions.</td>
            <td>"Prevents redundant disk I/O reads on every user interaction in Streamlit."</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 14 &mdash; STREAMLIT WEB ARCHITECTURE & USER FLOW</h1>

    <h2>1. What is Streamlit and Why Did We Choose It?</h2>
    <p>Streamlit is an open-source Python framework designed for rapidly creating interactive data and machine learning web applications directly from pure Python code. We chose Streamlit because it allows an ML engineer to deploy a complete, functional UI with buttons, sliders, progress bars, and dataframes without writing separate frontend JavaScript/HTML/CSS code.</p>

    <h2>2. Complete End-to-End User Flow</h2>
    <pre><code>User Enters Review Text in Browser Text Box
                    │
                    ▼
User Clicks "⚡ Run Sentiment Analysis"
                    │
                    ▼
Streamlit invokes cached `load_predictor()` (@st.cache_resource)
                    │
                    ▼
Text passes to `clean_text()` (HTML stripped, contractions expanded, lowercased)
                    │
                    ▼
Clean text transformed by `vectorizer.transform()` -> Sparse Vector (1, 10000)
                    │
                    ▼
Model runs `predict()` & `predict_proba()` -> Positive % & Negative %
                    │
                    ▼
Local explainability loop extracts top positive and negative token impacts
                    │
                    ▼
Streamlit renders: Sentiment Badge + Confidence Gauge + Token Badges + JSON Payload</code></pre>

    <h2>3. The Significance of <code>@st.cache_resource</code></h2>
    <p>Streamlit operates on a <em>reactive execution model</em>: every time a user types a character or clicks a button, Streamlit re-runs the Python script from top to bottom. Without <code>@st.cache_resource</code>, loading <code>best_model.joblib</code> from disk would happen on every click, creating high latency and memory leaks. The decorator loads the model into memory <strong>once</strong> and shares the object across all sessions.</p>
</div>

<div class="page-break-before">
    <h1>PART 15 &mdash; MODEL SERIALIZATION & JOBLIB</h1>

    <h2>1. Why Model Serialization is Essential</h2>
    <p>Training an NLP model on 50,000 documents requires parsing text, computing vocabulary frequencies across 40,000 reviews, and running iterative mathematical optimization. This takes ~30 seconds. In production, an API or web service cannot pause for 30 seconds every time a customer submits a review. <strong>Serialization</strong> freezes the trained mathematical state (learned weights $w_j$ and vocabulary mappings) and saves them as a binary byte stream on disk.</p>

    <h2>2. Why <code>joblib</code> Over Standard <code>pickle</code>?</h2>
    <p>Python's built-in <code>pickle</code> module serializes generic Python objects. However, <code>joblib</code> is specifically engineered for scientific computing. It handles large NumPy arrays and Scikit-Learn sparse matrices with continuous memory mapping and compression, executing load and save operations significantly faster with lower memory consumption.</p>

    <h2>3. Why the Vectorizer MUST be Serialized Alongside the Model</h2>
    <blockquote><strong>Critical Interview Trap:</strong> If an interviewer asks: <em>"Can I just save my trained Logistic Regression model and use a new TfidfVectorizer during deployment?"</em><br>
    <strong>Answer:</strong> <em>"Absolutely NOT! The Logistic Regression model learned that weight $w_{412}$ belongs to 'masterpiece' and weight $w_{8901}$ belongs to 'not good'. A new vectorizer would assign completely different index numbers to words based on new inputs. The exact same fitted vectorizer dictionary must be saved and loaded alongside the model to guarantee identical feature indices!"</em></blockquote>
</div>
"""
