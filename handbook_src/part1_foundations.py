"""
Part 1: AI/ML Foundations & Absolute Basics
"""

CONTENT = """
<div class="cover-page">
    <div class="cover-badge">TATA POWER GET AI/ML INTERVIEW COMPENDIUM</div>
    <div class="cover-title">MOVIE REVIEW SENTIMENT ANALYSIS</div>
    <div class="cover-subtitle">Complete Zero-to-Hero Study Handbook, Technical Guide & Interview Defense Manual</div>
    <div class="cover-divider"></div>
    
    <div class="meta-box">
        <div><strong>Target Candidate:</strong> Tata Power Graduate Engineer Trainee (GET) &ndash; AI/ML Track</div>
        <div><strong>Preparation Horizon:</strong> 7-Day Intensive Interview Ready Curriculum</div>
        <div><strong>Project Theme:</strong> Explainable NLP & Binary Sentiment Classification</div>
        <div><strong>Tech Stack:</strong> Python, Pandas, NumPy, Scikit-Learn, TF-IDF, Logistic Regression, Streamlit</div>
        <div><strong>Generalization Benchmark:</strong> 50,000 IMDB Records &bull; 90.23% Accuracy &bull; 90.38% F1-Score</div>
        <div><strong>Repository:</strong> <a href="https://github.com/SoumyaRanjan68/Movie-sentiment">github.com/SoumyaRanjan68/Movie-sentiment</a></div>
    </div>
</div>

<div class="page-break-before">
    <h1>TABLE OF CONTENTS</h1>
    <div class="toc-grid">
        <div>
            <ul>
                <li><strong>PART 1:</strong> AI/ML Foundations (From Absolute Zero)</li>
                <li><strong>PART 2:</strong> Essential Python for ML & Project Line-by-Line</li>
                <li><strong>PART 3:</strong> NumPy & Numerical Vector Operations</li>
                <li><strong>PART 4:</strong> Pandas Data Wrangling & Inspection</li>
                <li><strong>PART 5:</strong> NLP Fundamentals & Text Preprocessing</li>
                <li><strong>PART 6:</strong> TF-IDF Vectorization & Data Leakage</li>
                <li><strong>PART 7:</strong> Machine Learning Classifiers (Logistic Regression & Naive Bayes)</li>
                <li><strong>PART 8:</strong> Train/Test Splitting & Generalization</li>
                <li><strong>PART 9:</strong> Model Evaluation, Metrics & Confusion Matrix</li>
                <li><strong>PART 10:</strong> The Project Dataset (IMDB Benchmark)</li>
                <li><strong>PART 11:</strong> End-to-End Project Pipeline Architecture</li>
                <li><strong>PART 12:</strong> Project Source Code Line-by-Line Breakdown</li>
            </ul>
        </div>
        <div>
            <ul>
                <li><strong>PART 13:</strong> Code-to-Theory Mapping Reference Matrix</li>
                <li><strong>PART 14:</strong> Streamlit Web Interface & Architecture</li>
                <li><strong>PART 15:</strong> Model Serialization & Joblib Artifacts</li>
                <li><strong>PART 16:</strong> End-to-End Implementation from Scratch (15 Steps)</li>
                <li><strong>PART 17:</strong> Common Debugging Scenarios & Troubleshooting</li>
                <li><strong>PART 18:</strong> Real-World Limitations & Future Roadmap</li>
                <li><strong>PART 19:</strong> 100+ Categorized Interview Questions & Answers</li>
                <li><strong>PART 20:</strong> 30+ Realistic Project Cross-Questioning Chains</li>
                <li><strong>PART 21:</strong> Tata Power AI/ML JD Mapping & Alignment</li>
                <li><strong>PART 22:</strong> Resume Bullets & Word-by-Word Defense</li>
                <li><strong>PART 23:</strong> Rapid Revision Cheat Sheets</li>
                <li><strong>PART 24:</strong> 30-Minute Pre-Interview High-Impact Summary</li>
                <li><strong>PART 25:</strong> Final Master Preparation Checklist</li>
            </ul>
        </div>
    </div>
</div>

<div class="page-break-before">
    <h1>PART 1 &mdash; AI/ML FOUNDATIONS (FROM ABSOLUTE ZERO)</h1>
    <p><em>Welcome to the beginning. Even if you have never written a machine learning algorithm before, this section establishes the conceptual bedrock you need to explain and defend AI systems in a technical engineering interview.</em></p>

    <h2>1. What is Artificial Intelligence (AI)?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> Artificial Intelligence is the broad engineering discipline of creating computer systems that can perform tasks normally requiring human intelligence &mdash; such as recognizing images, understanding spoken words, making decisions, or translating languages.</li>
        <li><strong>Real-World Example:</strong> An automated chess game, a power-grid load controller that reroutes electricity during a storm, or a smartphone camera that detects human faces.</li>
        <li><strong>Technical Explanation:</strong> AI encompasses rule-based expert systems (if-then logic), search algorithms (A*, minimax), optimization, and learning algorithms. It is the umbrella term for any machine that mimics human cognitive functions.</li>
        <li><strong>How it Relates to This Project:</strong> Our project is an AI system because it reads unstructured human natural language (a movie review) and makes an intelligent cognitive judgment: "Is this person satisfied or dissatisfied?"</li>
    </ul>

    <h2>2. What is Machine Learning (ML)?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> Machine Learning is a specific branch of AI where, instead of explicitly writing thousands of rules by hand, we give the computer historical examples and let it <em>learn the statistical patterns on its own</em>.</li>
        <li><strong>Real-World Example:</strong> Instead of writing 10,000 rules to identify spam emails ("if contains 'lottery' then spam", "if contains 'wire money' then spam"), you feed the computer 50,000 emails labeled as "Spam" or "Not Spam" and let it discover which word combinations indicate spam.</li>
        <li><strong>Technical Explanation:</strong> Arthur Samuel (1959) defined ML as "the field of study that gives computers the ability to learn without being explicitly programmed." Mathematically, ML finds a mapping function $f: X \rightarrow Y$ by minimizing an objective loss function over training data.</li>
        <li><strong>How it Relates to This Project:</strong> We do not write rules like <code>if "good" in review: return positive</code>. We feed our machine learning model 39,665 labeled reviews, and it learns numerical weights for 10,000 terms automatically.</li>
    </ul>

    <h2>3. AI vs. Machine Learning vs. Deep Learning</h2>
    <table>
        <tr>
            <th>Concept</th>
            <th>What It Is</th>
            <th>How It Works</th>
            <th>Example</th>
        </tr>
        <tr>
            <td><strong>Artificial Intelligence (AI)</strong></td>
            <td>The broad universe of smart machines.</td>
            <td>Rules, heuristics, logic, or algorithms.</td>
            <td>Thermostat with automated control logic.</td>
        </tr>
        <tr>
            <td><strong>Machine Learning (ML)</strong></td>
            <td>Subfield of AI where machines learn from data.</td>
            <td>Statistical models, linear equations, decision trees.</td>
            <td><strong>Our Project: TF-IDF + Logistic Regression</strong></td>
        </tr>
        <tr>
            <td><strong>Deep Learning (DL)</strong></td>
            <td>Subfield of ML using multi-layered artificial neural networks.</td>
            <td>Deep neural architectures (CNNs, RNNs, Transformers, LLMs).</td>
            <td>ChatGPT, self-driving car vision systems.</td>
        </tr>
    </table>
    <blockquote><strong>Interview Tip:</strong> When the interviewer asks <em>"Why did you use classical ML instead of Deep Learning?"</em>, answer: <em>"Classical ML with Logistic Regression provides complete mathematical explainability, sub-5 millisecond inference latency, low memory footprint (~15 MB), and zero GPU requirements, while still delivering 90.23% accuracy on benchmark data."</em></blockquote>

    <h2>4. What is Supervised Learning?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> A type of machine learning where the computer learns from data that already contains the "correct answers" (called <em>labels</em>). It is like studying for an exam with textbook problems accompanied by the answer key at the back.</li>
        <li><strong>Real-World Example:</strong> Teaching a child to identify fruits by showing pictures with labels: "This is an apple", "This is an orange".</li>
        <li><strong>Technical Explanation:</strong> Given a dataset $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$ where $x_i \in \mathbb{R}^d$ are feature vectors and $y_i$ are ground-truth targets, the algorithm optimizes parameters $\mathbf{w}$ to minimize prediction error $\mathcal{L}(f(x_i; \mathbf{w}), y_i)$.</li>
        <li><strong>How it Relates to This Project:</strong> Our project is 100% Supervised Learning. Every review in our training set came paired with its ground-truth sentiment label (<code>positive</code> = 1, <code>negative</code> = 0).</li>
    </ul>

    <h2>5. What is Unsupervised Learning?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> Machine learning where the data has <em>no answer key or labels</em>. The algorithm's job is to discover hidden patterns, clusters, or groupings on its own.</li>
        <li><strong>Real-World Example:</strong> Grouping supermarket customers into 4 distinct buying personas based on their purchase history, without knowing their categories in advance (Customer Segmentation).</li>
        <li><strong>Technical Explanation:</strong> Algorithms like K-Means Clustering, Principal Component Analysis (PCA), and Autoencoders discover intrinsic geometric or statistical structures within $X$ without target $Y$.</li>
        <li><strong>How it Relates to This Project:</strong> We did not use unsupervised learning for our core task because we already possessed 50,000 ground-truth human-annotated sentiment labels.</li>
    </ul>

    <h2>6. What is Classification?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> A supervised learning task where the target output is a <strong>category or discrete label</strong> (e.g., Yes/No, Cat/Dog, High/Medium/Low).</li>
        <li><strong>Real-World Example:</strong> A power transformer oil quality diagnostic system that classifies condition as "Healthy", "Warning", or "Faulty".</li>
        <li><strong>Technical Explanation:</strong> In binary classification, the output $y \in \{0, 1\}$. In multi-class, $y \in \{1, 2, \dots, K\}$. The algorithm computes decision boundaries separating discrete classes in feature space.</li>
        <li><strong>How it Relates to This Project:</strong> Our project is a <strong>Binary Classification</strong> task: predicting whether $y = 1$ (Positive) or $y = 0$ (Negative).</li>
    </ul>

    <h2>7. What is Regression?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> A supervised learning task where the target output is a <strong>continuous numerical quantity</strong> (e.g., price, temperature, power demand).</li>
        <li><strong>Real-World Example:</strong> Predicting tomorrow's peak electrical demand in Megawatts (MW) based on weather temperature and industrial activity.</li>
        <li><strong>Technical Explanation:</strong> In regression, the output $y \in \mathbb{R}$. The model fits a continuous function to minimize Mean Squared Error (MSE).</li>
        <li><strong>Note:</strong> Even though our algorithm is named <em>Logistic Regression</em>, it is strictly used for <strong>Classification</strong> because its continuous output is passed through a Sigmoid function to output class probabilities!</li>
    </ul>

    <h2>8. What is a Feature?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> An individual measurable property or characteristic used by the machine learning model to make its decision.</li>
        <li><strong>Real-World Example:</strong> When predicting house prices, features are: square footage, number of bedrooms, location, and year built.</li>
        <li><strong>Technical Explanation:</strong> An input variable $x_j$ representing an axis in a $d$-dimensional feature space. A collection of $d$ features forms a feature vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$.</li>
        <li><strong>How it Relates to This Project:</strong> In our project, raw words cannot be fed directly to the model. Each unique vocabulary word or bigram (e.g., <code>"masterpiece"</code>, <code>"not good"</code>, <code>"boring"</code>) becomes a distinct <strong>feature column</strong> whose numerical value is its TF-IDF score. Our model uses exactly 10,000 features.</li>
    </ul>

    <h2>9. What is a Target (or Label)?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> The actual outcome or answer that the model is trying to predict.</li>
        <li><strong>Real-World Example:</strong> The actual selling price of the house, or whether an electrical transformer tripped (1) or did not trip (0).</li>
        <li><strong>Technical Explanation:</strong> The dependent variable $y$. In our dataset, the raw column <code>sentiment</code> has text strings <code>"positive"</code> and <code>"negative"</code>.</li>
        <li><strong>How it Relates to This Project:</strong> We map <code>"positive"</code> $\rightarrow 1$ and <code>"negative"</code> $\rightarrow 0$. This binary column <code>sentiment_label</code> is our target variable.</li>
    </ul>

    <h2>10. What is Training Data?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> The portion of your historical dataset (usually 70%&ndash;80%) given to the model during the learning phase so it can discover patterns.</li>
        <li><strong>Real-World Example:</strong> The practice problems and textbook exercises a student studies throughout the semester.</li>
        <li><strong>How it Relates to This Project:</strong> We allocated 80% of our 49,582 clean reviews &mdash; exactly <strong>39,665 reviews</strong> &mdash; as our Training Set.</li>
    </ul>

    <h2>11. What is Testing Data (Holdout Set)?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> The remaining portion of the dataset (usually 20%&ndash;30%) hidden from the model during training, used exclusively to test how well the model performs on completely unseen data.</li>
        <li><strong>Real-World Example:</strong> The final unseen exam questions given to the student at the end of the year to test genuine comprehension rather than rote memorization.</li>
        <li><strong>How it Relates to This Project:</strong> We set aside 20% &mdash; exactly <strong>9,917 reviews</strong> &mdash; as our Test Set. The model was evaluated on these 9,917 reviews to calculate our final 90.23% accuracy.</li>
    </ul>

    <h2>12. What is a Model?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> The mathematical artifact or "formula" created by the machine learning algorithm after it finishes studying the training data.</li>
        <li><strong>Real-World Example:</strong> A recipe created by a chef after experimenting with 1,000 ingredient combinations.</li>
        <li><strong>Technical Explanation:</strong> A parameterized function $f(\mathbf{x}; \mathbf{w})$ where $\mathbf{w}$ are learned internal weights/coefficients. Before training, $\mathbf{w}$ are random or zero. After training, $\mathbf{w}$ represent fine-tuned values that accurately map inputs to predictions.</li>
    </ul>

    <h2>13. What Does "Training a Model" Actually Mean?</h2>
    <ul>
        <li><strong>Simple Definition:</strong> Training is the automated trial-and-error process where the computer adjusts its internal mathematical knobs (weights) to make its predictions as close as possible to the real answers.</li>
        <li><strong>Real-World Example:</strong> An archer adjusting their bow angle after every shot until they consistently hit the bullseye.</li>
        <li><strong>Technical Explanation:</strong> An optimization loop (e.g., L-BFGS, Stochastic Gradient Descent) that calculates the gradient of the loss function with respect to weights $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ and iteratively updates weights until the loss reaches a minimum.</li>
    </ul>

    <div class="checkpoint-box">
        <h3>🎯 CHECK YOUR UNDERSTANDING &mdash; CHAPTER 1</h3>
        <ol>
            <li><strong>Q:</strong> Is our sentiment analysis project supervised or unsupervised? Why?</li>
            <li><strong>Q:</strong> Why is our project considered a classification task and not a regression task?</li>
            <li><strong>Q:</strong> What would happen if we trained our model on 100% of the data without saving a test set?</li>
            <li><strong>Q:</strong> In one sentence, what is a "feature" in our movie review sentiment analysis project?</li>
        </ol>
        <div class="answers-toggle">
            <strong>Answers:</strong>
            <p>1. Supervised, because every review in our dataset comes with a human-provided ground truth label (positive or negative).<br>
            2. Because the output is a discrete category (Positive or Negative), not a continuous numeric quantity.<br>
            3. We would have no way to verify if the model truly learned generalizable patterns or simply memorized the training examples (overfitting).<br>
            4. A feature is an individual vocabulary word or bigram whose numerical value is its TF-IDF score in that review.</p>
        </div>
    </div>
</div>
"""
