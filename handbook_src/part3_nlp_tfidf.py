"""
Part 3: NLP Fundamentals and TF-IDF Deep Dive
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 5 &mdash; NATURAL LANGUAGE PROCESSING (NLP) FUNDAMENTALS</h1>

    <h2>1. What is NLP and Why is it Necessary?</h2>
    <p>Natural Language Processing (NLP) is the intersection of computer science, artificial intelligence, and computational linguistics. Human language is inherently unstructured, ambiguous, metaphorical, and non-linear. Computers, by contrast, are deterministic calculating engines built on binary arithmetic and linear algebra.</p>
    <p><strong>The Core Challenge:</strong> A computer has no intrinsic understanding of what the letters <code>'g-o-o-d'</code> mean. To enable a machine learning model to classify sentiment, we must convert unstructured character sequences into a structured, numerical geometric representation where semantic relationships can be analyzed mathematically.</p>

    <h2>2. Basic NLP Terminology</h2>
    <ul>
        <li><strong>Token:</strong> The fundamental atomic unit of text (typically an individual word or subword).</li>
        <li><strong>Sentence:</strong> An ordered grammatical sequence of tokens terminated by punctuation.</li>
        <li><strong>Document:</strong> A complete individual text record (in our project, a single movie review).</li>
        <li><strong>Corpus:</strong> A collection of all documents under analysis (in our project, all 50,000 IMDB reviews).</li>
        <li><strong>Vocabulary:</strong> The set of all unique, distinct tokens present across the corpus.</li>
    </ul>

    <h2>3. The Preprocessing Techniques: What, Why, and Project Usage</h2>
    <table>
        <tr>
            <th>Technique</th>
            <th>What It Does</th>
            <th>Why We Need It</th>
            <th>Used in Project?</th>
            <th>Engineering Justification</th>
        </tr>
        <tr>
            <td><strong>HTML Stripping</strong></td>
            <td>Removes web tags like <code>&lt;br /&gt;</code>, <code>&lt;p&gt;</code>.</td>
            <td>HTML tags are website rendering artifacts, not movie review sentiment.</td>
            <td><strong>YES</strong></td>
            <td>Regex <code>re.sub(r'&lt;.*?&gt;', ' ', text)</code> cleanly purges 100% of HTML artifacts.</td>
        </tr>
        <tr>
            <td><strong>Contraction Expansion</strong></td>
            <td>Expands <code>"didn't"</code> $\rightarrow$ <code>"did not"</code>.</td>
            <td>Preserves the critical negation word <code>"not"</code> before punctuation cleaning.</td>
            <td><strong>YES</strong></td>
            <td>Essential for sentiment polarity. Prevents <code>"didn't like"</code> from turning into <code>"like"</code>.</td>
        </tr>
        <tr>
            <td><strong>Lowercasing</strong></td>
            <td>Converts all characters to lowercase.</td>
            <td>Unifies vocabulary so <code>"Masterpiece"</code>, <code>"masterpiece"</code>, and <code>"MASTERPIECE"</code> map to one token.</td>
            <td><strong>YES</strong></td>
            <td><code>text.lower()</code> prevents vocabulary explosion and duplicate dimensions.</td>
        </tr>
        <tr>
            <td><strong>Punctuation Removal</strong></td>
            <td>Removes symbols like commas, periods, quotes.</td>
            <td>Removes noise so <code>"acting,"</code> and <code>"acting."</code> map to <code>"acting"</code>.</td>
            <td><strong>YES</strong></td>
            <td>Replaces non-alphanumeric characters with spaces using regex <code>[^a-zA-Z0-9\s]</code>.</td>
        </tr>
        <tr>
            <td><strong>Whitespace Collapsing</strong></td>
            <td>Reduces multiple consecutive spaces to one.</td>
            <td>Ensures clean token boundaries during tokenization.</td>
            <td><strong>YES</strong></td>
            <td>Regex <code>\s+</code> replaced by single space, followed by <code>.strip()</code>.</td>
        </tr>
        <tr>
            <td><strong>Stopword Stripping (Blind)</strong></td>
            <td>Deletes high-frequency English words.</td>
            <td>Reduces feature space dimensionality.</td>
            <td><strong>NO (Avoided)</strong></td>
            <td><strong>THE NEGATION TRAP:</strong> Standard stopword lists contain <code>"not"</code>, <code>"no"</code>, <code>"never"</code>. Blindly deleting them turns <code>"not good"</code> into <code>"good"</code>, inverting sentiment!</td>
        </tr>
        <tr>
            <td><strong>Stemming</strong></td>
            <td>Heuristically chops word suffixes (e.g. <code>"running"</code> $\rightarrow$ <code>"run"</code>).</td>
            <td>Groups morphological variants of words together.</td>
            <td><strong>NO</strong></td>
            <td>Stemmers (like Porter) corrupt words (e.g. <code>"caring"</code> $\rightarrow$ <code>"car"</code>), distorting semantic tone.</td>
        </tr>
        <tr>
            <td><strong>Lemmatization</strong></td>
            <td>Uses vocabulary and grammar to find dictionary base form (lemma).</td>
            <td>More linguistically accurate than stemming.</td>
            <td><strong>NO</strong></td>
            <td>Requires Part-of-Speech (POS) tagging which increases latency 10x with zero accuracy gain over our 10,000-ngram TF-IDF space.</td>
        </tr>
    </table>

    <h2>4. N-Grams: Unigrams, Bigrams, and Trigrams</h2>
    <ul>
        <li><strong>Unigram (1-gram):</strong> A single isolated word: <code>["the", "movie", "was", "not", "good"]</code>.</li>
        <li><strong>Bigram (2-gram):</strong> A sequence of two consecutive words: <code>["the movie", "movie was", "was not", "not good"]</code>.</li>
        <li><strong>Trigram (3-gram):</strong> A sequence of three consecutive words: <code>["the movie was", "was not good"]</code>.</li>
    </ul>
    <blockquote><strong>Why Bigrams are Essential in Our Project:</strong> Under unigrams alone, the sentence <em>"not good"</em> is split into two separate words: <code>"not"</code> and <code>"good"</code>. The word <code>"good"</code> has a positive weight, confusing the classifier. By setting <code>ngram_range=(1, 2)</code>, our vectorizer creates the bigram <code>"not good"</code> as an independent feature. In our trained Logistic Regression model, <code>"not good"</code> received a massive negative coefficient of <strong>-2.646</strong>, completely resolving the negation!</blockquote>
</div>

<div class="page-break-before">
    <h1>PART 6 &mdash; TF-IDF VECTORIZATION & DATA LEAKAGE</h1>

    <h2>1. The Limitations of Simple Bag-of-Words (CountVectorizer)</h2>
    <p>The simplest way to convert text to numbers is <em>Bag of Words</em>: count how many times each word appears in a document. For example, if "great" appears 3 times, its feature value is 3.</p>
    <p><strong>The Fatal Flaw:</strong> In any movie review corpus, words like <code>"movie"</code>, <code>"film"</code>, <code>"watch"</code>, and <code>"scene"</code> appear in virtually every single review. In Bag-of-Words, these generic words receive the highest counts, dominating the feature space even though they contain <strong>zero discriminative sentiment information</strong>!</p>

    <h2>2. The Core Intuition of TF-IDF</h2>
    <p>TF-IDF fixes Bag-of-Words with a simple, brilliant intuition:</p>
    <ol>
        <li>If a word appears frequently in a <em>specific review</em>, it must be important to that review (&uarr; <strong>High Term Frequency</strong>).</li>
        <li>However, if that same word appears in <em>almost every review across the entire dataset</em>, it is useless for distinguishing positive from negative (&darr; <strong>Low Inverse Document Frequency penalty</strong>).</li>
        <li>Therefore, words that are frequent in <em>one review</em> but rare across the <em>whole corpus</em> (like <code>"masterpiece"</code> or <code>"abysmal"</code>) receive the <strong>highest TF-IDF score</strong>!</li>
    </ol>

    <h2>3. The Mathematical Formula (Scikit-Learn Implementation)</h2>
    <p>$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$</p>
    
    <h3>Term Frequency (TF):</h3>
    <p>In our project, we enabled <code>sublinear_tf=True</code>. Instead of raw counts, term frequency is scaled logarithmically:</p>
    <p>$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for count} > 0$$</p>
    <p><em>Why? A word appearing 20 times in a long review is not 20 times more important than a word appearing once. Log scaling dampens the dominance of repeated words.</em></p>

    <h3>Inverse Document Frequency (IDF) with Smooth IDF:</h3>
    <p>$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$</p>
    <ul>
        <li>$N$: Total number of documents in the training corpus ($N = 39,665$).</li>
        <li>$\text{DF}(t)$: Document Frequency &mdash; the number of documents containing term $t$.</li>
        <li>The constant $+1$ in the numerator and denominator prevents division by zero.</li>
        <li>The $+1$ outside the logarithm prevents terms with $\text{DF}(t) = N$ from receiving zero weight entirely.</li>
    </ul>

    <h2>4. Step-by-Step Numerical Example</h2>
    <p>Suppose our corpus has $N = 10,000$ reviews. Consider two terms in a review:</p>
    <table>
        <tr>
            <th>Word</th>
            <th>Count in this review</th>
            <th>DF (Total reviews containing word)</th>
            <th>Calculated IDF</th>
            <th>Final TF-IDF Score</th>
            <th>Interpretation</th>
        </tr>
        <tr>
            <td><code>"movie"</code></td>
            <td>4</td>
            <td>9,900</td>
            <td>$\log\left(\frac{10001}{9901}\right) + 1 = \mathbf{1.01}$</td>
            <td>$(1 + \log(4)) \times 1.01 = \mathbf{2.41}$</td>
            <td>Penalized heavily because it appears everywhere.</td>
        </tr>
        <tr>
            <td><code>"masterpiece"</code></td>
            <td>2</td>
            <td>85</td>
            <td>$\log\left(\frac{10001}{86}\right) + 1 = \mathbf{5.75}$</td>
            <td>$(1 + \log(2)) \times 5.75 = \mathbf{9.74}$</td>
            <td><strong>Boosted &times;4 higher</strong> because it is rare and informative!</td>
        </tr>
    </table>

    <h2>5. Critical Parameters of TfidfVectorizer in Our Project</h2>
    <pre><code>vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=3,
    sublinear_tf=True
)</code></pre>
    <ul>
        <li><code>ngram_range=(1, 2)</code>: Extracts both single words and consecutive 2-word combinations to capture phrases like <code>"not good"</code>.</li>
        <li><code>max_features=10000</code>: Selects the top 10,000 terms ordered by term frequency across the corpus. Discards the long tail of 200,000+ noisy terms, preventing overfitting and reducing RAM usage.</li>
        <li><code>min_df=3</code>: Discards terms appearing in fewer than 3 documents. Purges rare typos (e.g. <code>"goooood"</code>) and random noise.</li>
        <li><code>sublinear_tf=True</code>: Applies logarithmic scaling $1 + \log(\text{tf})$ to term frequency.</li>
    </ul>

    <h2>6. What is Data Leakage and How Did We Prevent It?</h2>
    <p><strong>Data Leakage</strong> is a fatal machine learning flaw where information from the testing/evaluation dataset leaks into the training process before model fitting. This causes models to look amazing during development but completely fail in production.</p>

    <blockquote>
        <h3>THE CARDINAL RULE OF VECTORIZATION:</h3>
        <p><code>fit_transform(X_train)</code> on Training Data &bull; <code>transform(X_test)</code> on Testing Data</p>
    </blockquote>
    <ul>
        <li><code>fit()</code> calculates the vocabulary list and the global IDF weights ($\text{DF}(t)$ for all words).</li>
        <li>If you call <code>fit(X)</code> on the full dataset, your model learns IDF scores and vocabulary frequencies calculated from the <strong>unseen test set</strong> &mdash; a direct data leak!</li>
        <li>In our project:
            <pre><code># Correct Implementation in src/train_model.py:
X_train_tfidf = vectorizer.fit_transform(X_train) # Learns vocab & IDF weights ONLY on train
X_test_tfidf = vectorizer.transform(X_test)       # Reuses train weights to transform test</code></pre>
        </li>
    </ul>

    <div class="checkpoint-box">
        <h3>🎯 CHECK YOUR UNDERSTANDING &mdash; CHAPTERS 5 & 6</h3>
        <ol>
            <li><strong>Q:</strong> Why does TF-IDF assign a low score to the word "film" in a movie dataset?</li>
            <li><strong>Q:</strong> Why do we expand contractions like <code>"didn't"</code> before removing punctuation?</li>
            <li><strong>Q:</strong> What is the difference between <code>fit_transform()</code> and <code>transform()</code>?</li>
            <li><strong>Q:</strong> What is a sparse matrix and why is it necessary for TF-IDF?</li>
        </ol>
        <div class="answers-toggle">
            <strong>Answers:</strong>
            <p>1. Because "film" appears in almost all reviews, its document frequency $\text{DF}(t)$ is near $N$, making its IDF weight close to 1.0.<br>
            2. To prevent "didn't" from becoming "didnt" or losing the negation "not" when punctuation is stripped.<br>
            3. <code>fit_transform()</code> computes vocabulary and IDF parameters and transforms data. <code>transform()</code> applies pre-computed parameters to new data without updating them.<br>
            4. In a 10,000-word vocabulary, any single review only uses ~150 words. A sparse matrix only stores the non-zero indices and values in memory, saving 98% of RAM.</p>
        </div>
    </div>
</div>
"""
