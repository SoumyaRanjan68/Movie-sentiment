"""
Part 8: Cross-Examination Simulation, Tata Power JD Alignment, Resume Defense, and Cheat Sheets
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 20 &mdash; SIMULATED INTERVIEW: 30+ REALISTIC CROSS-EXAMINATION CHAINS</h1>
    <p><em>In a top-tier technical interview like Tata Power GET AI/ML, interviewers do not ask isolated questions. They drill down in continuous logical chains to see where your knowledge breaks. Below are the exact follow-up chains you will encounter:</em></p>

    <!-- CHAIN 1 -->
    <div class="dialogue-box">
        <h4>Chain 1: Project Genesis & Architecture Selection</h4>
        <p><strong>Interviewer:</strong> "Walk me through your Movie Review Sentiment Analysis project in 90 seconds."</p>
        <p><strong>You:</strong> "I developed an end-to-end sentiment classification system on 50,000 IMDB reviews. The raw text goes through deterministic NLP preprocessing where HTML is excised and contractions are expanded to preserve negation words. I split the data using an 80/20 stratified partition to prevent data leakage, vectorized the text into a 10,000-dimensional unigram and bigram TF-IDF sparse matrix, and benchmarked Logistic Regression against Multinomial Naive Bayes. Logistic Regression emerged as the winning model, delivering 90.23% accuracy and a 90.38% F1-score on 9,917 holdout test reviews. I serialized the model and vectorizer with Joblib and deployed an interactive Streamlit platform featuring real-time Explainable AI token contributions and sub-3ms inference latency."</p>
        
        <p><strong>Interviewer:</strong> "Why did you choose Logistic Regression instead of an advanced Transformer like BERT?"</p>
        <p><strong>You:</strong> "For three specific engineering reasons: First, explainability &mdash; in Logistic Regression, every word has an exact mathematical coefficient ($w_j$), so I can explain every decision to business stakeholders. Second, latency &mdash; our CPU inference executes in 2.1 milliseconds compared to 150-200ms for BERT. Third, infrastructure cost &mdash; our serialized model is only 80 KB and requires zero GPU overhead, while achieving over 90% benchmark accuracy. It was the optimal engineering trade-off."</p>

        <p><strong>Interviewer:</strong> "Fair. But Logistic Regression is a linear classifier. How can it capture non-linear language patterns?"</p>
        <p><strong>You:</strong> "That is why feature engineering with n-grams was crucial. In a raw unigram bag-of-words space, language relationships are non-linear because word order matters. By expanding our feature space to include bigrams ($N=2$) with 10,000 dimensions, combinations like 'not good' and 'must watch' become independent linear dimensions with their own distinct coefficients. In high-dimensional sparse spaces, linear hyperplanes are remarkably powerful."</p>
    </div>

    <!-- CHAIN 2 -->
    <div class="dialogue-box">
        <h4>Chain 2: Data Hygiene & Data Leakage Prevention</h4>
        <p><strong>Interviewer:</strong> "Did you clean the entire dataset before splitting it into train and test?"</p>
        <p><strong>You:</strong> "Row-level text cleaning &mdash; such as lowercasing, HTML stripping, and regex normalization &mdash; is applied to each document independently and does not create data leakage. However, statistical vectorization was strictly isolated: I performed the 80/20 stratified split <em>before</em> vectorization, called <code>fit_transform()</code> exclusively on the training set, and applied only <code>transform()</code> on the test set."</p>

        <p><strong>Interviewer:</strong> "What exactly would leak if you had called <code>fit_transform()</code> on the whole dataset?"</p>
        <p><strong>You:</strong> "Two things would leak: First, the Inverse Document Frequency (IDF) weights &mdash; the model would calculate term rarity using test set document frequencies. Second, the vocabulary list &mdash; the top 10,000 most frequent terms would be influenced by words appearing in the test set. That would give our training model prior knowledge of the test set, creating artificially inflated metrics that fail in production."</p>

        <p><strong>Interviewer:</strong> "I noticed you dropped 418 duplicate reviews. Why does that matter?"</p>
        <p><strong>You:</strong> "If identical reviews exist in the raw dataset, random splitting can place one copy in the training set and another in the test set. The model would then be evaluated on text it had already memorized during training &mdash; another subtle form of data leakage. Deduplication guarantees that test reviews are 100% genuine out-of-sample documents."</p>
    </div>

    <!-- CHAIN 3 -->
    <div class="dialogue-box">
        <h4>Chain 3: Negation Handling & Sarcasm Edge Cases</h4>
        <p><strong>Interviewer:</strong> "If a user enters: 'The movie was not good', how does your model know it's negative?"</p>
        <p><strong>You:</strong> "Through two synchronized steps: First, our preprocessor expands contractions so words like 'wasn't' or 'didn't' become 'was not' and 'did not'. Second, our TF-IDF vectorizer extracts bigrams with <code>ngram_range=(1, 2)</code>. The pair 'not good' is an explicit feature in our vocabulary and has a large negative coefficient of -2.646, which mathematically overrides the standalone positive weight of 'good'."</p>

        <p><strong>Interviewer:</strong> "What if the review is: 'What a glorious disaster this movie was'? How does it perform?"</p>
        <p><strong>You:</strong> "That exposes the classical limitation of n-gram bag-of-words models: sarcasm. The unigram 'glorious' has a positive weight of +1.8, while 'disaster' has a negative weight of -2.2. Depending on other words in the sentence, the model may struggle with tonal irony. To handle complex sarcasm in production, we would need semantic attention models like DistilBERT or sentiment lexicons trained on ironic discourse."</p>
    </div>
</div>

<div class="page-break-before">
    <h1>PART 21 &mdash; TATA POWER AI/ML JOB DESCRIPTION ALIGNMENT</h1>
    <p><em>Connect your project directly to what Tata Power seeks in a Graduate Engineer Trainee (AI/ML):</em></p>

    <table>
        <tr>
            <th>Tata Power GET Requirement</th>
            <th>How This Project Demonstrates It</th>
            <th>What This Project DOES NOT Cover (Be Honest!)</th>
        </tr>
        <tr>
            <td><strong>Python Programming & Scripting</strong></td>
            <td>Modular, object-oriented source code in <code>src/</code> with type hints, custom classes, regex, and clean exception handling.</td>
            <td>Does not cover asynchronous networking (<code>asyncio</code>) or C-extensions.</td>
        </tr>
        <tr>
            <td><strong>Data Wrangling & Analysis (Pandas/NumPy)</strong></td>
            <td>Handling 50,000 tabular records, missing value verification, deduplication, vectorized label transformations.</td>
            <td>Does not cover distributed big-data engines like Apache Spark or PySpark.</td>
        </tr>
        <tr>
            <td><strong>Applied Machine Learning Lifecycle</strong></td>
            <td>Complete lifecycle: Data Ingestion $\rightarrow$ EDA $\rightarrow$ Partitioning $\rightarrow$ Feature Engineering $\rightarrow$ Modeling $\rightarrow$ Evaluation $\rightarrow$ Serialization $\rightarrow$ UI Deployment.</td>
            <td>Does not cover deep reinforcement learning or complex ML pipelines like Kubeflow.</td>
        </tr>
        <tr>
            <td><strong>Statistical Model Evaluation</strong></td>
            <td>Confusion matrices, Type I and Type II errors, Precision, Recall, F1-Score, and threshold analysis.</td>
            <td>Does not cover Bayesian A/B testing frameworks.</td>
        </tr>
        <tr>
            <td><strong>Production Deployment</strong></td>
            <td>Streamlit interactive web dashboard, Joblib model serialization, sub-3ms latency, and Docker configuration.</td>
            <td>Does not cover Kubernetes cluster orchestration or automated CI/CD retraining pipelines.</td>
        </tr>
    </table>

    <h2>How to Tie This Project to Tata Power's Core Business:</h2>
    <ul>
        <li><strong>Consumer Grievance Prioritization:</strong> Tata Power manages millions of customer service interactions across Mumbai, Delhi, and Odisha. This exact NLP classification pipeline can ingest incoming consumer complaint emails and automatically classify sentiment urgency &mdash; routing dangerous complaints (e.g. "live wire sparking", "transformer smoking", "blackout in ICU ward") to emergency response crews within seconds.</li>
        <li><strong>Equipment Maintenance Log Analytics:</strong> Field technicians regularly enter qualitative notes during routine inspection of substations and solar arrays. Our TF-IDF classification engine can analyze these text logs to detect early equipment failure patterns ("heavy vibration", "abnormal oil seepage") before catastrophic power outages occur.</li>
    </ul>
</div>

<div class="page-break-before">
    <h1>PART 22 &mdash; RESUME DEFENSE: WORD-BY-WORD SCRUTINY</h1>

    <h2>Your Exact Verified Resume Bullets:</h2>
    <div class="resume-box">
        <p><strong>Movie Review Sentiment Analysis System | Python, NLP, Scikit-Learn, Streamlit</strong></p>
        <ul>
            <li>Engineered an end-to-end sentiment classification pipeline on 50,000 IMDB reviews, achieving <strong>90.23% Accuracy</strong> and <strong>90.38% F1-Score</strong> using TF-IDF (Unigrams + Bigrams) and L2-regularized Logistic Regression.</li>
            <li>Designed custom sentiment-preserving text normalization and contraction expansion to maintain contextual polarity and accurately resolve negation phrases (e.g., <em>"not good"</em>).</li>
            <li>Evaluated and benchmarked Logistic Regression against Multinomial Naive Bayes using Stratified Train-Test splits, Confusion Matrices, Precision, and Recall to prevent Data Leakage.</li>
            <li>Deployed an interactive web application with <strong>Streamlit</strong> and serialized inference pipelines with <strong>Joblib</strong>, achieving <strong>&lt; 2.5ms</strong> real-time prediction latency on standard CPUs.</li>
        </ul>
    </div>

    <h2>Word-by-Word Defense Matrix:</h2>
    <table>
        <tr>
            <th>Term in Resume</th>
            <th>What the Interviewer Will Ask</th>
            <th>Your Instant Technical Defense</th>
        </tr>
        <tr>
            <td><strong>"50,000 reviews"</strong></td>
            <td>"Did you train on all 50,000?"</td>
            <td>"No, I removed 418 duplicates, leaving 49,582 clean reviews. Then I performed an 80/20 stratified split: 39,665 reviews for training, and 9,917 held out for final testing."</td>
        </tr>
        <tr>
            <td><strong>"90.23% Accuracy"</strong></td>
            <td>"Is that on training or testing data?"</td>
            <td>"That is strictly on the 9,917 holdout test reviews. The model correctly classified 8,947 out of 9,917 unseen test reviews."</td>
        </tr>
        <tr>
            <td><strong>"TF-IDF"</strong></td>
            <td>"What is the mathematical formula?"</td>
            <td>"$\text{TF-IDF} = \text{TF} \times \text{IDF}$. I used sublinear TF ($1 + \log(\text{tf})$) and smooth IDF ($\log((1+N)/(1+DF)) + 1$) with a vocabulary of 10,000 n-grams."</td>
        </tr>
        <tr>
            <td><strong>"Data Leakage"</strong></td>
            <td>"How exactly did you prevent it?"</td>
            <td>"By splitting the data before vectorization, calling <code>fit_transform()</code> only on training data, and calling <code>transform()</code> on test data."</td>
        </tr>
        <tr>
            <td><strong>"&lt; 2.5ms latency"</strong></td>
            <td>"Why is it so fast?"</td>
            <td>"Inference only requires a sparse matrix lookup, a dot product on non-zero terms ($<50$ multiplications), and a single scalar Sigmoid calculation on CPU."</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 23 &mdash; RAPID REVISION CHEAT SHEETS</h1>

    <h2>Cheat Sheet 1: NLP & Preprocessing</h2>
    <table>
        <tr>
            <th>Concept</th>
            <th>Formula / Rule</th>
            <th>Key Takeaway</th>
        </tr>
        <tr>
            <td>HTML Removal</td>
            <td><code>re.sub(r'&lt;.*?&gt;', ' ', text)</code></td>
            <td>Removes web tags without removing text.</td>
        </tr>
        <tr>
            <td>Contraction Expansion</td>
            <td><code>"didn't" -> "did not"</code></td>
            <td><strong>Preserves negation words before punctuation cleaning.</strong></td>
        </tr>
        <tr>
            <td>Lowercasing</td>
            <td><code>text.lower()</code></td>
            <td>Prevents duplicated vocabulary dimensions.</td>
        </tr>
        <tr>
            <td>N-grams</td>
            <td>Unigram (1), Bigram (2), Trigram (3)</td>
            <td>Captures localized phrase order (e.g. "not good").</td>
        </tr>
    </table>

    <h2>Cheat Sheet 2: TF-IDF & Machine Learning</h2>
    <table>
        <tr>
            <th>Concept</th>
            <th>Formula</th>
            <th>Key Takeaway</th>
        </tr>
        <tr>
            <td>Term Frequency (TF)</td>
            <td>$1 + \log(\text{count})$</td>
            <td>Log scaling prevents repeated words from dominating.</td>
        </tr>
        <tr>
            <td>Inverse Doc Frequency (IDF)</td>
            <td>$\log\left(\frac{1+N}{1+DF}\right) + 1$</td>
            <td>Penalizes common words (film); boosts rare words (masterpiece).</td>
        </tr>
        <tr>
            <td>Sigmoid Function</td>
            <td>$\sigma(z) = \frac{1}{1 + e^{-z}}$</td>
            <td>Maps linear score $(-\infty, +\infty)$ to probability $[0, 1]$.</td>
        </tr>
        <tr>
            <td>Decision Threshold</td>
            <td>$P \ge 0.50 \implies \text{Positive}$</td>
            <td>Boundary where linear score $z = 0$.</td>
        </tr>
        <tr>
            <td>L2 Regularization</td>
            <td>$\frac{1}{2C} \sum w_j^2$</td>
            <td>Shrinks model weights to prevent overfitting.</td>
        </tr>
    </table>

    <h2>Cheat Sheet 3: Evaluation Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Formula</th>
            <th>Our Project Score</th>
            <th>Core Interpretation</th>
        </tr>
        <tr>
            <td><strong>Accuracy</strong></td>
            <td>$\frac{TP + TN}{\text{Total}}$</td>
            <td><strong>90.23%</strong></td>
            <td>Overall correct prediction percentage.</td>
        </tr>
        <tr>
            <td><strong>Precision</strong></td>
            <td>$\frac{TP}{TP + FP}$</td>
            <td><strong>89.32%</strong></td>
            <td>Exactness: Low false positive alarm rate.</td>
        </tr>
        <tr>
            <td><strong>Recall</strong></td>
            <td>$\frac{TP}{TP + FN}$</td>
            <td><strong>91.46%</strong></td>
            <td>Completeness: High detection of true positives.</td>
        </tr>
        <tr>
            <td><strong>F1-Score</strong></td>
            <td>$2 \times \frac{P \times R}{P + R}$</td>
            <td><strong>90.38%</strong></td>
            <td>Harmonic mean balancing precision and recall.</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 24 &mdash; 30 MINUTES BEFORE THE INTERVIEW: HIGH-IMPACT REVISION</h1>
    <p><em>Read this single page 30 minutes before entering your interview. It contains the essential facts you must speak with confidence:</em></p>

    <div class="meta-box" style="width: 95%;">
        <h3>⚡ THE 10 GOLDEN FACTS OF YOUR PROJECT:</h3>
        <ol>
            <li><strong>Dataset:</strong> 50,000 IMDB reviews $\rightarrow$ 418 duplicates dropped $\rightarrow$ <strong>49,582 clean reviews</strong> (24,884 Positive / 24,698 Negative &mdash; balanced 50:50).</li>
            <li><strong>Split:</strong> Stratified 80% Train (39,665 reviews) / 20% Test (9,917 reviews). Random state = 42.</li>
            <li><strong>Data Leakage Prevention:</strong> <code>fit_transform()</code> strictly on training set; <code>transform()</code> on test set.</li>
            <li><strong>Preprocessing:</strong> HTML unescaping, contraction expansion (<code>"didn't"</code> $\rightarrow$ <code>"did not"</code>), lowercasing, non-alphanumeric removal, whitespace collapsing.</li>
            <li><strong>Why No Stopword Removal:</strong> Deleting stopwords removes <code>"not"</code> and <code>"no"</code>, inverting sentiment polarity.</li>
            <li><strong>Feature Space:</strong> TF-IDF with unigrams and bigrams (<code>ngram_range=(1, 2)</code>), <code>max_features=10000</code>, <code>min_df=3</code>, <code>sublinear_tf=True</code>.</li>
            <li><strong>Winning Model:</strong> Logistic Regression ($C=1.0$, L-BFGS solver).</li>
            <li><strong>Test Accuracy:</strong> <strong>90.23%</strong> (8,947 / 9,917 correct). Test F1-Score: <strong>90.38%</strong>.</li>
            <li><strong>Baseline Comparison:</strong> Multinomial Naive Bayes achieved 86.95% accuracy and 87.18% F1-score. Logistic Regression won because it does not assume feature independence.</li>
            <li><strong>Inference & Deployment:</strong> Model and vectorizer serialized using <code>joblib</code>; served via Streamlit with sub-2.5ms latency.</li>
        </ol>
    </div>

    <h2>The Top 5 Questions You Are Guaranteed to Be Asked:</h2>
    <ul>
        <li><strong>1. "Why did you use TF-IDF instead of CountVectorizer?"</strong><br>
        &rarr; <em>"CountVectorizer only counts frequencies, letting generic words like 'movie' dominate. TF-IDF penalizes common words and amplifies discriminative sentiment words like 'masterpiece'."</em></li>
        
        <li><strong>2. "How did you prevent data leakage?"</strong><br>
        &rarr; <em>"I performed an 80/20 stratified train-test split before feature extraction. The vectorizer was fitted exclusively on the training set, so no test set vocabulary or IDF frequencies influenced model weights."</em></li>
        
        <li><strong>3. "Why did Logistic Regression beat Naive Bayes?"</strong><br>
        &rarr; <em>"Naive Bayes assumes words are conditionally independent, which natural language violates (e.g. 'not' and 'good'). Logistic Regression directly optimizes the decision boundary without independence constraints."</em></li>
        
        <li><strong>4. "How does your model handle negation like 'not good'?"</strong><br>
        &rarr; <em>"Through contraction expansion to preserve 'not', and bigram feature extraction ($N=2$). The bigram 'not good' received a strong negative coefficient (-2.646), overpowering the positive weight of 'good'."</em></li>
        
        <li><strong>5. "How does this connect to Tata Power?"</strong><br>
        &rarr; <em>"It can be directly deployed to classify incoming consumer complaints and equipment inspection logs, automatically routing urgent safety grievances (e.g. sparking lines, transformer faults) to field crews."</em></li>
    </ul>
</div>

<div class="page-break-before">
    <h1>PART 25 &mdash; MASTER PREPARATION CHECKLIST</h1>
    <p><em>Check off each item once you can explain it out loud without looking at notes:</em></p>

    <div class="checklist">
        <p>[ &nbsp; ] <strong>1. AI/ML Foundations:</strong> I can explain the difference between Supervised vs. Unsupervised Learning, and Classification vs. Regression.</p>
        <p>[ &nbsp; ] <strong>2. Python Core:</strong> I can explain why lists, tuples, dictionaries, and sets are used in different parts of our project.</p>
        <p>[ &nbsp; ] <strong>3. Pandas & NumPy:</strong> I can explain <code>read_csv</code>, <code>drop_duplicates</code>, <code>isnull().sum()</code>, and how NumPy contiguous arrays speed up dot products.</p>
        <p>[ &nbsp; ] <strong>4. NLP Preprocessing:</strong> I can explain why contraction expansion is critical before stripping punctuation to preserve negation.</p>
        <p>[ &nbsp; ] <strong>5. Stopwords Trap:</strong> I can explain why blind stopword removal inverts sentiment from "not good" to "good".</p>
        <p>[ &nbsp; ] <strong>6. Stemming vs. Lemmatization:</strong> I can explain why we avoided aggressive stemming in favor of n-gram TF-IDF.</p>
        <p>[ &nbsp; ] <strong>7. TF-IDF Formula:</strong> I can write and explain both TF ($1+\log(\text{tf})$) and Smooth IDF ($\log((1+N)/(1+DF))+1$).</p>
        <p>[ &nbsp; ] <strong>8. Data Leakage:</strong> I can explain why <code>fit_transform(X_train)</code> and <code>transform(X_test)</code> is mandatory.</p>
        <p>[ &nbsp; ] <strong>9. Logistic Regression Math:</strong> I can write the linear score $z = \mathbf{w}^T \mathbf{x} + b$, the Sigmoid equation $\sigma(z)$, and explain decision threshold 0.50.</p>
        <p>[ &nbsp; ] <strong>10. L2 Regularization:</strong> I can explain the penalty term $\frac{1}{2C} \sum w_j^2$ and the role of parameter $C=1.0$.</p>
        <p>[ &nbsp; ] <strong>11. Naive Bayes:</strong> I can state Bayes' theorem, the "naive" conditional independence assumption, and Laplace smoothing ($\alpha=1.0$).</p>
        <p>[ &nbsp; ] <strong>12. Confusion Matrix:</strong> I can state our exact test numbers (4,395 TN, 545 FP, 425 FN, 4,552 TP) and explain Type I vs. Type II errors.</p>
        <p>[ &nbsp; ] <strong>13. Metrics:</strong> I can define Accuracy (90.23%), Precision (89.32%), Recall (91.46%), and F1-Score (90.38%).</p>
        <p>[ &nbsp; ] <strong>14. Joblib Serialization:</strong> I can explain why the vectorizer must be saved alongside the model to preserve feature index alignments.</p>
        <p>[ &nbsp; ] <strong>15. Streamlit Architecture:</strong> I can explain <code>@st.cache_resource</code> and the reactive execution model.</p>
        <p>[ &nbsp; ] <strong>16. Real-Time Explainability:</strong> I can explain how token contributions are calculated ($w_j \times \text{tfidf}_j$) to show positive vs. negative drivers.</p>
        <p>[ &nbsp; ] <strong>17. Project Limitations:</strong> I can explain how sarcasm and long-distance negation challenge classical n-gram models.</p>
        <p>[ &nbsp; ] <strong>18. Tata Power Connection:</strong> I can articulate how this pipeline translates to consumer grievance prioritization and equipment maintenance logs.</p>
        <p>[ &nbsp; ] <strong>19. Live Application:</strong> I can launch and demonstrate the Streamlit app on <code>http://localhost:8502</code> or the deployed URL.</p>
        <p>[ &nbsp; ] <strong>20. Resume Defense:</strong> I can defend every single word, number, and technology mentioned in my resume bullet points.</p>
    </div>

    <hr>
    <p style="text-align: center; color: #475569; font-weight: 600; font-size: 11pt;">
        &mdash; END OF MASTER HANDBOOK &mdash;<br>
        <em>Master these 25 parts and you will comfortably stand among the top 1% of candidates in your Tata Power AI/ML interview. Good luck!</em>
    </p>
</div>
"""
