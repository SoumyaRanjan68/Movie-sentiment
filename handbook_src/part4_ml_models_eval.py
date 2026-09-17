"""
Part 4: Machine Learning Models, Training, and Evaluation Metrics
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 7 &mdash; MACHINE LEARNING CLASSIFIERS: LOGISTIC REGRESSION & NAIVE BAYES</h1>

    <h2>1. The Classification Task in Feature Space</h2>
    <p>Once TF-IDF transforms each review into a 10,000-dimensional numerical vector $\mathbf{x} = [x_1, x_2, \dots, x_{10000}]^T$, our machine learning task is to learn a mathematical decision boundary that separates positive vectors from negative vectors.</p>

    <h2>2. Model 1: Logistic Regression (Winning Model &mdash; 90.23% Accuracy)</h2>
    <p>Despite its name containing "Regression", Logistic Regression is a <strong>linear classification algorithm</strong> that predicts the probability of a categorical outcome.</p>

    <h3>Mathematical Formulation:</h3>
    <ol>
        <li><strong>Linear Score Calculation:</strong> The model computes a weighted sum of all input features plus an intercept (bias) term:
            $$z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = \mathbf{w}^T \mathbf{x} + b$$
            Here, each $w_j$ represents the learned coefficient for word $j$. A positive weight (e.g. $+2.4$ for <code>"masterpiece"</code>) pushes the score toward Positive; a negative weight (e.g. $-2.6$ for <code>"not good"</code>) pulls it toward Negative.
        </li>
        <li><strong>Sigmoid (Logistic) Function:</strong> The continuous linear score $z \in (-\infty, +\infty)$ is mapped into a valid probability $P \in (0, 1)$ using the standard Sigmoid curve:
            $$\sigma(z) = \frac{1}{1 + e^{-z}} = P(y=1|\mathbf{x})$$
            <ul>
                <li>If $z = 0 \implies \sigma(0) = \frac{1}{1 + 1} = 0.50$ (Exact boundary).</li>
                <li>If $z = +3 \implies \sigma(3) \approx 0.95$ (95% confident Positive).</li>
                <li>If $z = -3 \implies \sigma(-3) \approx 0.05$ (5% positive $\implies$ 95% confident Negative).</li>
            </ul>
        </li>
        <li><strong>Decision Threshold:</strong> By default, if $P(y=1|\mathbf{x}) \ge 0.50$, the review is classified as <strong>Positive (1)</strong>; otherwise, <strong>Negative (0)</strong>.</li>
        <li><strong>Binary Cross-Entropy Loss (Log-Loss):</strong> The objective function minimized during training:
            $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
            The term $\frac{1}{2C} \|\mathbf{w}\|_2^2$ is <strong>L2 Regularization (Ridge penalty)</strong>, which shrinks large weights to prevent overfitting on specific words. The parameter $C=1.0$ controls the trade-off between fitting training data and keeping weights small.
        </li>
    </ol>

    <h2>3. Model 2: Multinomial Naive Bayes (Baseline &mdash; 86.95% Accuracy)</h2>
    <p>A probabilistic classifier based on <strong>Bayes' Theorem</strong> tailored for discrete word counts and frequencies.</p>

    <h3>Bayes' Theorem:</h3>
    <p>$$P(\text{Class}|X) = \frac{P(\text{Class}) \times P(X|\text{Class})}{P(X)}$$</p>
    <ul>
        <li>$P(\text{Class})$ is the <strong>Prior Probability</strong> (50% for positive, 50% for negative in our balanced dataset).</li>
        <li>$P(X|\text{Class})$ is the <strong>Likelihood</strong> of observing these specific words given the sentiment class.</li>
    </ul>

    <h3>The "Naive" Assumption:</h3>
    <p>Naive Bayes assumes that all words in the document are <strong>conditionally independent of one another</strong> given the class label:</p>
    <p>$$P(w_1, w_2, \dots, w_n|\text{Class}) = \prod_{i=1}^n P(w_i|\text{Class})$$</p>

    <h3>Laplace (Additive) Smoothing ($\alpha = 1.0$):</h3>
    <p>$$\hat{P}(w_i|\text{Class}) = \frac{\text{Count}(w_i, \text{Class}) + \alpha}{\sum_{w} \text{Count}(w, \text{Class}) + \alpha \cdot |V|}$$</p>
    <p><em>Why is it mandatory? Without Laplace smoothing ($\alpha=0$), if a test review contains a single word never seen in positive training reviews, $P(w|\text{Positive}) = 0$. Because all probabilities are multiplied together, the entire calculation collapses to zero ($0 \times \dots = 0$)! Adding $\alpha=1$ ensures every word has a tiny non-zero probability.</em></p>

    <h2>4. Why Did Logistic Regression Outperform Naive Bayes?</h2>
    <table>
        <tr>
            <th>Dimension</th>
            <th>Logistic Regression</th>
            <th>Multinomial Naive Bayes</th>
        </tr>
        <tr>
            <td><strong>Paradigm</strong></td>
            <td><strong>Discriminative</strong>: Directly models decision boundary $P(y|X)$.</td>
            <td><strong>Generative</strong>: Models joint distribution $P(X, y)$.</td>
        </tr>
        <tr>
            <td><strong>Feature Correlation</strong></td>
            <td><strong>Robust</strong>: Learns optimal joint weights for correlated words and bigrams.</td>
            <td><strong>Fragile</strong>: Falsely assumes words are independent (e.g. "special" and "effects").</td>
        </tr>
        <tr>
            <td><strong>Measured Accuracy</strong></td>
            <td><strong>90.23% (Winning Model)</strong></td>
            <td>86.95%</td>
        </tr>
        <tr>
            <td><strong>Measured F1-Score</strong></td>
            <td><strong>90.38%</strong></td>
            <td>87.18%</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 8 &mdash; TRAIN/TEST SPLIT & GENERALIZATION</h1>

    <h2>1. The Goal of Machine Learning: Generalization</h2>
    <p>The true measure of a machine learning system is not how well it remembers training examples, but how accurately it performs on <strong>fresh, unseen data</strong> (Generalization). Memorizing the training data is called <strong>Overfitting</strong>.</p>

    <h2>2. Stratified Train/Test Split (80% Train, 20% Test)</h2>
    <pre><code># Actual Project Code in src/train_model.py:
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)</code></pre>
    <ul>
        <li><code>test_size=0.20</code>: Exactly 20% of data (9,917 reviews) is held out as the test set. 80% (39,665 reviews) is reserved for training.</li>
        <li><code>random_state=42</code>: Fixes the pseudo-random number generator seed, ensuring identical data partitions every time the code is executed (100% scientific reproducibility).</li>
        <li><code>stratify=y</code>: <strong>Stratification</strong> ensures the exact percentage of positive (50.2%) and negative (49.8%) reviews is preserved in both train and test partitions. Prevents sampling bias.</li>
    </ul>
</div>

<div class="page-break-before">
    <h1>PART 9 &mdash; MODEL EVALUATION, METRICS & CONFUSION MATRIX</h1>

    <h2>1. The Confusion Matrix ($2 \times 2$)</h2>
    <p>A confusion matrix is an unvarnished cross-tabulation of ground-truth labels vs. model predictions:</p>
    <table>
        <tr>
            <th colspan="2" rowspan="2"></th>
            <th colspan="2">Model Prediction</th>
        </tr>
        <tr>
            <th>Predicted Negative (0)</th>
            <th>Predicted Positive (1)</th>
        </tr>
        <tr>
            <th rowspan="2">Ground Truth</th>
            <th>Actual Negative (0)</th>
            <td><strong>True Negative (TN)</strong><br>Correctly identified negative</td>
            <td><strong>False Positive (FP)</strong><br>Type I Error (False Alarm)</td>
        </tr>
        <tr>
            <th>Actual Positive (1)</th>
            <td><strong>False Negative (FN)</strong><br>Type II Error (Missed Case)</td>
            <td><strong>True Positive (TP)</strong><br>Correctly identified positive</td>
        </tr>
    </table>

    <h2>2. Evaluation Formulas & Numerical Walkthrough</h2>
    <ul>
        <li><strong>Accuracy:</strong> The proportion of all predictions that were correct:
            $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
        </li>
        <li><strong>Precision:</strong> When the model predicts Positive, how often is it right?
            $$\text{Precision} = \frac{TP}{TP + FP}$$
        </li>
        <li><strong>Recall (Sensitivity):</strong> Out of all actual Positive samples, how many did the model catch?
            $$\text{Recall} = \frac{TP}{TP + FN}$$
        </li>
        <li><strong>F1-Score:</strong> The harmonic mean of Precision and Recall:
            $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
        </li>
    </ul>

    <h2>3. When to Prioritize Precision vs. Recall?</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Primary Objective</th>
            <th>Real-World Industrial Example</th>
        </tr>
        <tr>
            <td><strong>High Precision</strong></td>
            <td>Minimize False Positives (False alarms are costly or disruptive).</td>
            <td>Spam filters (never send a critical client email to spam) or legal fraud accusations.</td>
        </tr>
        <tr>
            <td><strong>High Recall</strong></td>
            <td>Minimize False Negatives (Missed detections are dangerous or catastrophic).</td>
            <td>Power transformer overheating alarms (never miss a fire risk, even if false alarms occur).</td>
        </tr>
        <tr>
            <td><strong>High F1-Score</strong></td>
            <td>Balance between Precision and Recall.</td>
            <td>Customer sentiment classification where both false positive and false negative errors matter equally.</td>
        </tr>
    </table>

    <h2>4. Actual Measured Results on 9,917 Holdout Reviews</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Logistic Regression (Winning Model)</th>
            <th>Multinomial Naive Bayes</th>
        </tr>
        <tr>
            <td><strong>Accuracy</strong></td>
            <td><strong>90.23% (8,947 / 9,917)</strong></td>
            <td>86.95% (8,623 / 9,917)</td>
        </tr>
        <tr>
            <td><strong>Precision</strong></td>
            <td><strong>89.32%</strong></td>
            <td>85.97%</td>
        </tr>
        <tr>
            <td><strong>Recall</strong></td>
            <td><strong>91.46%</strong></td>
            <td>88.43%</td>
        </tr>
        <tr>
            <td><strong>F1-Score</strong></td>
            <td><strong>90.38%</strong></td>
            <td>87.18%</td>
        </tr>
        <tr>
            <td><strong>True Negatives (TN)</strong></td>
            <td><strong>4,395</strong></td>
            <td>4,220</td>
        </tr>
        <tr>
            <td><strong>False Positives (FP)</strong></td>
            <td><strong>545</strong></td>
            <td>720</td>
        </tr>
        <tr>
            <td><strong>False Negatives (FN)</strong></td>
            <td><strong>425</strong></td>
            <td>574</td>
        </tr>
        <tr>
            <td><strong>True Positives (TP)</strong></td>
            <td><strong>4,552</strong></td>
            <td>4,403</td>
        </tr>
    </table>

    <div class="checkpoint-box">
        <h3>🎯 CHECK YOUR UNDERSTANDING &mdash; CHAPTERS 7, 8 & 9</h3>
        <ol>
            <li><strong>Q:</strong> What is the output range of the Sigmoid function and why is it useful?</li>
            <li><strong>Q:</strong> Why can Accuracy be misleading if a dataset is heavily imbalanced?</li>
            <li><strong>Q:</strong> If our model has 545 False Positives and 425 False Negatives, what does that indicate about model bias?</li>
        </ol>
        <div class="answers-toggle">
            <strong>Answers:</strong>
            <p>1. The Sigmoid function outputs values strictly between 0.0 and 1.0, allowing continuous linear scores to be interpreted as calibrated probabilities.<br>
            2. If 99% of data is negative, a trivial model predicting "negative" for everything achieves 99% accuracy while failing 100% on the positive class.<br>
            3. The near-equal error distribution demonstrates that our model is balanced and exhibits no systemic bias toward either class.</p>
        </div>
    </div>
</div>
"""
