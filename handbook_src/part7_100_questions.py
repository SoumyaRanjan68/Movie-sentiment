"""
Part 7: 100+ Categorized Technical Interview Questions & Answers
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 19 &mdash; 100+ CATEGORIZED INTERVIEW QUESTIONS & ANSWERS</h1>
    <p><em>Every question in this compendium includes: the punchy Short Answer, the in-depth Technical Answer, the Interviewer's Underlying Motivation, and the Critical Mistake to Avoid.</em></p>

    <!-- SECTION A: PYTHON -->
    <h2>Section A: Python Core Essentials</h2>
    
    <h3>Q1. What is the difference between mutable and immutable objects in Python?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Mutable objects (lists, dicts, sets) can be altered in-place after creation without changing their memory address. Immutable objects (ints, floats, strings, tuples) cannot be altered; any modification allocates a new object in memory.</li>
        <li><strong>Detailed Answer:</strong> In Python, variable names are references pointing to memory addresses (checked via <code>id(x)</code>). When modifying a list (<code>list.append(x)</code>), the underlying array is updated in-place. When modifying a string (<code>s += 'a'</code>), Python allocates a new string object in memory. In ML pipelines, immutability protects hyperparameters and configurations from unintended side-effects.</li>
        <li><strong>Why They Ask:</strong> Tests core Python memory mechanics and prevention of reference-mutation bugs.</li>
        <li><strong>Common Mistake:</strong> Saying tuples are "just lists with parentheses" without explaining memory allocation or hashability.</li>
    </ul>

    <h3>Q2. How do list comprehensions differ from standard <code>for</code> loops in terms of execution?</h3>
    <ul>
        <li><strong>Short Answer:</strong> List comprehensions are more concise, readable, and execute faster because the looping and appending bytecode is evaluated in optimized C within Python's interpreter.</li>
        <li><strong>Detailed Answer:</strong> In a standard <code>for</code> loop with <code>.append()</code>, Python performs an attribute lookup for the <code>append</code> method and dynamic function call on every iteration. List comprehensions use the specialized <code>LIST_APPEND</code> bytecode instruction, avoiding repeated method resolution overhead.</li>
        <li><strong>Why They Ask:</strong> Tests code optimization skills and Python internals.</li>
        <li><strong>Common Mistake:</strong> Using overly complex nested list comprehensions that destroy code readability.</li>
    </ul>

    <h3>Q3. What is a lambda function in Python and where would you use it?</h3>
    <ul>
        <li><strong>Short Answer:</strong> An anonymous, inline function defined with the <code>lambda</code> keyword that takes any number of arguments but contains only a single expression.</li>
        <li><strong>Detailed Answer:</strong> Syntax: <code>lambda args: expression</code>. In our project, it is used inside Pandas <code>apply()</code> for simple transformations: <code>df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))</code>.</li>
        <li><strong>Why They Ask:</strong> Checks functional programming fluency in data manipulation.</li>
        <li><strong>Common Mistake:</strong> Writing multi-line logic inside a lambda; complex logic should always be a standard <code>def</code> function.</li>
    </ul>

    <h3>Q4. What is the difference between <code>==</code> and <code>is</code> in Python?</h3>
    <ul>
        <li><strong>Short Answer:</strong> <code>==</code> checks for equality of value (do both objects contain the same data?), whereas <code>is</code> checks for identity of memory address (are both variables pointing to the exact same object in RAM?).</li>
        <li><strong>Detailed Answer:</strong> <code>a = [1, 2]</code> and <code>b = [1, 2]</code> $\implies a == b$ is <code>True</code>, but $a \text{ is } b$ is <code>False</code> because they reside at different memory addresses (<code>id(a) != id(b)</code>). <code>is</code> should strictly be used for singleton comparisons like <code>x is None</code>.</li>
        <li><strong>Why They Ask:</strong> Basic language precision test.</li>
        <li><strong>Common Mistake:</strong> Using <code>if x is 1:</code> instead of <code>if x == 1:</code>.</li>
    </ul>

    <h3>Q5. What is the Global Interpreter Lock (GIL) and how does it affect Python ML workflows?</h3>
    <ul>
        <li><strong>Short Answer:</strong> A mutex that allows only one native thread to execute Python bytecode at a time, preventing true CPU multi-threading in pure Python.</li>
        <li><strong>Detailed Answer:</strong> The GIL prevents race conditions in CPython's reference counting memory management. However, in machine learning, numerical libraries like NumPy, Scikit-Learn, and SciPy release the GIL during heavy vector calculations, executing multi-threaded BLAS/OpenMP routines in compiled C/Fortran.</li>
        <li><strong>Why They Ask:</strong> High-level systems and concurrency understanding.</li>
        <li><strong>Common Mistake:</strong> Claiming that Python ML cannot use multiple CPU cores (it does, via external C-libraries and multiprocessing).</li>
    </ul>

    <!-- SECTION B: NUMPY & PANDAS -->
    <h2>Section B: NumPy & Pandas Foundations</h2>

    <h3>Q6. Why is NumPy significantly faster than Python native lists for numerical operations?</h3>
    <ul>
        <li><strong>Short Answer:</strong> NumPy arrays store contiguous blocks of homogeneous memory, avoid Python object overhead, and leverage SIMD CPU hardware vectorization.</li>
        <li><strong>Detailed Answer:</strong> Python lists store pointers to scattered objects with type-tagging and reference counts. NumPy arrays are contiguous blocks of typed memory (e.g. 64-bit floats), maximizing CPU cache locality and allowing vectorized C-loops and SIMD registers to process multiple numbers in a single CPU cycle.</li>
        <li><strong>Why They Ask:</strong> Foundational knowledge of why data science stacks are built on NumPy.</li>
        <li><strong>Common Mistake:</strong> Saying "it's written in C" without explaining contiguous memory or cache locality.</li>
    </ul>

    <h3>Q7. How does Pandas represent missing values, and how does <code>isnull().sum()</code> work?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Missing values are represented as floating-point <code>np.nan</code>. <code>isnull()</code> returns a boolean mask (True for NaN, False otherwise), and <code>sum()</code> adds up the True values (treating True as 1).</li>
        <li><strong>Detailed Answer:</strong> Because <code>NaN != NaN</code> by IEEE floating-point standard, direct comparison <code>x == np.nan</code> fails. Pandas provides <code>isnull()</code> to detect bitwise NaN representations across rows. Summing the resulting boolean series gives the exact count of missing records per column.</li>
        <li><strong>Why They Ask:</strong> Data hygiene and quality inspection fundamentals.</li>
        <li><strong>Common Mistake:</strong> Trying to check <code>if row == np.nan:</code>.</li>
    </ul>

    <h3>Q8. What is the difference between <code>df.drop_duplicates()</code> and filtering manually?</h3>
    <ul>
        <li><strong>Short Answer:</strong> <code>drop_duplicates()</code> uses an optimized internal hash-set algorithm in C to identify and purge identical rows across specified column subsets in a single vectorized pass.</li>
        <li><strong>Detailed Answer:</strong> In our project, <code>df.drop_duplicates(subset=['review'])</code> hashed review texts and dropped 418 duplicates. Manual looping would take $O(N^2)$ time; Pandas executes it in $O(N)$ amortized time.</li>
        <li><strong>Why They Ask:</strong> Big-O algorithmic efficiency awareness.</li>
        <li><strong>Common Mistake:</strong> Forgetting to pass <code>subset=['review']</code> and checking all columns instead.</li>
    </ul>

    <!-- SECTION C: NLP FUNDAMENTALS -->
    <h2>Section C: Natural Language Processing (NLP)</h2>

    <h3>Q9. What is the fundamental difference between Stemming and Lemmatization?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Stemming chops off word suffixes using crude heuristic rules (often creating non-words like <code>'univers'</code>). Lemmatization uses morphological analysis and dictionary lookups to find the true root word (<code>'better'</code> $\rightarrow$ <code>'good'</code>).</li>
        <li><strong>Detailed Answer:</strong> Porter or Snowball stemmers apply algorithmic suffix-stripping without linguistic understanding. Lemmatization (WordNet, spaCy) incorporates Part-of-Speech (POS) tags to distinguish whether a word is a noun, verb, or adjective before converting it to its lemma. Lemmatization is cleaner but computationally slower.</li>
        <li><strong>Why They Ask:</strong> Core NLP text normalization theory.</li>
        <li><strong>Common Mistake:</strong> Claiming your project used Lemmatization when you actually used regex-based normalization.</li>
    </ul>

    <h3>Q10. Why is blind stopword removal dangerous in sentiment analysis?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Standard stopword lists include negation words like <code>'not'</code>, <code>'no'</code>, and <code>'never'</code>. Deleting them inverts sentiment polarity (turning <code>'not good'</code> into <code>'good'</code>).</li>
        <li><strong>Detailed Answer:</strong> Standard stopword corpora (such as NLTK's English stopword list) were developed for topic modeling and document retrieval, where negation is irrelevant. In sentiment analysis, polarity is dictated by modifiers. We preserve negations through contraction expansion (<code>"didn't"</code> $\rightarrow$ <code>"did not"</code>) and bigram feature binding.</li>
        <li><strong>Why They Ask:</strong> Distinguishes practitioners from textbook memorizers.</li>
        <li><strong>Common Mistake:</strong> Saying "I always remove all stopwords in every NLP project".</li>
    </ul>

    <!-- SECTION D: TF-IDF -->
    <h2>Section D: TF-IDF & Feature Extraction</h2>

    <h3>Q11. Derive and explain the TF-IDF formula used in Scikit-Learn.</h3>
    <ul>
        <li><strong>Short Answer:</strong> $\text{TF-IDF} = \text{TF} \times \text{IDF}$. TF measures term frequency in a document; IDF measures the rarity of the term across all documents ($\log(\frac{1+N}{1+DF}) + 1$).</li>
        <li><strong>Detailed Answer:</strong> Scikit-Learn uses the Smooth IDF formulation: $\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$, followed by L2 Euclidean normalization on each row vector $\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$. This ensures document length does not bias cosine distances between vectors.</li>
        <li><strong>Why They Ask:</strong> Mathematical precision and Scikit-Learn framework depth.</li>
        <li><strong>Common Mistake:</strong> Stating the raw textbook formula $\log(N/DF)$ without the $+1$ smoothing constants.</li>
    </ul>

    <h3>Q12. What does <code>sublinear_tf=True</code> do and why did you use it?</h3>
    <ul>
        <li><strong>Short Answer:</strong> It replaces raw term frequency $\text{TF}$ with $1 + \log(\text{TF})$. It prevents a word appearing 20 times from having 20 times more influence than a word appearing once.</li>
        <li><strong>Detailed Answer:</strong> Human perception of frequency is logarithmic (Weber-Fechner Law). In movie reviews, an author repeating the word "boring" 10 times is not 10 times more dissatisfied than an author writing it twice. Log scaling dampens repetition bias.</li>
        <li><strong>Why They Ask:</strong> Tests advanced hyperparameter tuning knowledge.</li>
        <li><strong>Common Mistake:</strong> Confusing sublinear TF with sublinear IDF.</li>
    </ul>

    <!-- SECTION E: MACHINE LEARNING & LOGISTIC REGRESSION -->
    <h2>Section E: Machine Learning Classifiers</h2>

    <h3>Q13. Why is Logistic Regression considered a linear model if the Sigmoid curve is non-linear?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Because the <strong>decision boundary</strong> in the feature space is a linear hyperplane ($z = \mathbf{w}^T \mathbf{x} + b = 0$).</li>
        <li><strong>Detailed Answer:</strong> The log-odds (logit) function is a strictly linear combination of input features: $\log\left(\frac{P}{1 - P}\right) = \mathbf{w}^T \mathbf{x} + b$. The non-linear Sigmoid function $\sigma(z)$ only acts as a monotonic squashing function to map the linear score into probability space $[0, 1]$. The boundary between classes is a flat hyperplane.</li>
        <li><strong>Why They Ask:</strong> Deep conceptual check on model linearity.</li>
        <li><strong>Common Mistake:</strong> Saying "it is non-linear because the sigmoid curve is an S-shape".</li>
    </ul>

    <h3>Q14. How does Logistic Regression optimize its weights? What solver did you use?</h3>
    <ul>
        <li><strong>Short Answer:</strong> It optimizes weights by minimizing Binary Cross-Entropy loss via gradient-based numerical optimization. We used the **L-BFGS** solver.</li>
        <li><strong>Detailed Answer:</strong> Unlike Ordinary Least Squares (OLS) regression, Logistic Regression has no closed-form analytical solution. It uses numerical optimization. L-BFGS (Limited-memory Broyden&ndash;Fletcher&ndash;Goldfarb&ndash;Shanno) is a quasi-Newton method that approximates the inverse Hessian matrix using gradient evaluations, making it fast and memory-efficient for 10,000 sparse features.</li>
        <li><strong>Why They Ask:</strong> Tests mathematical solver and optimization mechanics.</li>
        <li><strong>Common Mistake:</strong> Saying it uses "exact matrix inversion $(X^T X)^{-1}$".</li>
    </ul>

    <h3>Q15. What is the mathematical significance of parameter $C$ in Logistic Regression?</h3>
    <ul>
        <li><strong>Short Answer:</strong> $C$ is the inverse of regularization strength ($C = 1/\lambda$). Smaller $C$ values enforce stronger weight shrinkage (preventing overfitting); larger $C$ values prioritize fitting training data closely.</li>
        <li><strong>Detailed Answer:</strong> The objective function is $\min_{\mathbf{w}} C \times \text{Loss} + \frac{1}{2} \|\mathbf{w}\|_2^2$. Dividing by $C$ shows that $C \rightarrow 0$ penalizes any non-zero weight heavily, while $C \rightarrow \infty$ disables regularization entirely. We set $C=1.0$ as a balanced default.</li>
        <li><strong>Why They Ask:</strong> Regularization and bias-variance trade-off understanding.</li>
        <li><strong>Common Mistake:</strong> Claiming larger $C$ means stronger regularization (it is the exact opposite!).</li>
    </ul>

    <!-- SECTION F: EVALUATION METRICS -->
    <h2>Section F: Model Evaluation & Diagnostics</h2>

    <h3>Q16. What is the Harmonic Mean and why does F1-Score use it instead of the Arithmetic Mean?</h3>
    <ul>
        <li><strong>Short Answer:</strong> The harmonic mean gives greater weight to smaller numbers. If either Precision or Recall is near zero, the F1-Score collapses to zero, exposing failure.</li>
        <li><strong>Detailed Answer:</strong> If Precision = 1.0 and Recall = 0.0 (the model predicted zero positives correctly), the Arithmetic Mean is $\frac{1.0 + 0.0}{2} = 0.50$ (misleadingly acceptable!). The Harmonic Mean is $2 \times \frac{1.0 \times 0.0}{1.0 + 0.0} = \mathbf{0.0}$. It ensures a model cannot hide poor recall behind high precision.</li>
        <li><strong>Why They Ask:</strong> Mathematical intuition behind evaluation metrics.</li>
        <li><strong>Common Mistake:</strong> Saying "because it's the standard formula" without explaining the penalty on extreme values.</li>
    </ul>

    <h3>Q17. Explain your confusion matrix numbers: 4395, 545, 425, 4552.</h3>
    <ul>
        <li><strong>Short Answer:</strong> On 9,917 holdout test reviews: 4,395 True Negatives, 545 False Positives, 425 False Negatives, and 4,552 True Positives. Accuracy = 90.23%.</li>
        <li><strong>Detailed Answer:</strong> False Positives (545) were negative reviews mistaken as positive (often due to sarcasm like "brilliant waste of time"). False Negatives (425) were positive reviews mistaken as negative (often nuanced critiques with balanced phrasing). The symmetric error profile confirms balanced performance with zero bias toward either class.</li>
        <li><strong>Why They Ask:</strong> Proves you actually ran and analyzed your model.</li>
        <li><strong>Common Mistake:</strong> Mixing up False Positives (Type I) with False Negatives (Type II).</li>
    </ul>

    <!-- SECTION G: TOUGH PROJECT SPECIFICS -->
    <h2>Section G: Tough Project-Specific Scenarios</h2>

    <h3>Q18. What happens if a user submits a review containing completely new words never seen in training?</h3>
    <ul>
        <li><strong>Short Answer:</strong> The new Out-of-Vocabulary (OOV) words are simply ignored by the TF-IDF vectorizer (receiving zero weight). The model bases its prediction on whatever remaining known words exist.</li>
        <li><strong>Detailed Answer:</strong> The TF-IDF matrix has a fixed dimension of 10,000 columns. Any word not in the 10,000-vocabulary dictionary has no assigned column and is silently dropped during <code>transform()</code>. If a review consists <em>entirely</em> of unseen words, the feature vector is all zeros, and the model outputs the base prior probability via the intercept $b \approx 50\%$.</li>
        <li><strong>Why They Ask:</strong> Production edge-case handling.</li>
        <li><strong>Common Mistake:</strong> Claiming the vectorizer throws an error or crashes (it does not).</li>
    </ul>

    <h3>Q19. How would you modify your pipeline if you needed to support Hindi or mixed "Hinglish" reviews?</h3>
    <ul>
        <li><strong>Short Answer:</strong> Replace the English contraction and regex cleaner with a multilingual tokenizer, train on a Romanized Hindi/Hinglish dataset, or use a multilingual Transformer like mBERT or XLM-RoBERTa.</li>
        <li><strong>Detailed Answer:</strong> Classical English TF-IDF cannot parse Devanagari script or transliterated slang ("mast picture thi", "bakwas movie"). We would collect a bilingual customer review dataset, apply subword tokenization (Byte-Pair Encoding / SentencePiece), and fine-tune a multilingual embedding model like <code>indic-bert</code> or <code>xlm-roberta-base</code>.</li>
        <li><strong>Why They Ask:</strong> Tests adaptability to Indian industrial contexts (like Tata Power consumer feedback).</li>
        <li><strong>Common Mistake:</strong> Saying "I will just pass it to Google Translate first" (adds external API dependency and latency).</li>
    </ul>

    <h3>Q20. Why didn't you use BERT or GPT for this project?</h3>
    <ul>
        <li><strong>Short Answer:</strong> For explainability, inference speed (&lt; 2.5 ms vs 200 ms), zero GPU requirements, and lightweight deployment (~15 MB vs 500 MB+).</li>
        <li><strong>Detailed Answer:</strong> Engineering is about selecting the simplest architecture that solves the problem within operational constraints. For standard binary movie review sentiment, TF-IDF + Logistic Regression achieves **90.23% accuracy** on CPU with near-instantaneous inference and 100% auditable word weights. Deploying BERT would increase infrastructure costs by &times;20 for a marginal 3% accuracy gain.</li>
        <li><strong>Why They Ask:</strong> Evaluates practical engineering judgment vs. blind trend-chasing.</li>
        <li><strong>Common Mistake:</strong> Apologizing for not using BERT; defend your architecture proudly!</li>
    </ul>
</div>
"""
