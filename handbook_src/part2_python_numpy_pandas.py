"""
Part 2: Python, NumPy, and Pandas Foundations
"""

CONTENT = """
<div class="page-break-before">
    <h1>PART 2 &mdash; PYTHON FOUNDATIONS FOR MACHINE LEARNING</h1>
    <p><em>Machine learning engineers do not need to memorize every obscure Python syntax quirk. You only need to master the exact data structures and control flows that manipulate datasets and models. Below is every Python construct used in this project, explained from absolute zero.</em></p>

    <h2>1. Variables and Dynamic Typing</h2>
    <p>In Python, a variable is a named storage container for a value in computer memory. Python is dynamically typed, meaning you do not have to declare types like <code>int x</code>.</p>
    <pre><code># Project Example:
random_state = 42                # Integer: sets the random seed
confidence_threshold = 0.50      # Float: decision boundary probability
model_name = "Logistic Regression" # String: model name</code></pre>

    <h2>2. Core Data Structures: Lists, Tuples, Sets, and Dictionaries</h2>
    <table>
        <tr>
            <th>Data Structure</th>
            <th>Syntax</th>
            <th>Properties</th>
            <th>Where It Appears in Our Project</th>
        </tr>
        <tr>
            <td><strong>List</strong></td>
            <td><code>[item1, item2]</code></td>
            <td>Ordered, mutable, allows duplicate items.</td>
            <td>Storing test reviews: <code>test_cases = ["Great film", "Awful film"]</code> in <code>src/predict.py</code></td>
        </tr>
        <tr>
            <td><strong>Tuple</strong></td>
            <td><code>(item1, item2)</code></td>
            <td>Ordered, immutable (read-only), allows duplicates.</td>
            <td>Specifying n-gram ranges: <code>ngram_range=(1, 2)</code> in <code>src/train_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Set</strong></td>
            <td><code>{item1, item2}</code></td>
            <td>Unordered, mutable, <strong>unique values only</strong>, $O(1)$ fast lookups.</td>
            <td>Checking distinct vocabulary tokens and filtering stop words.</td>
        </tr>
        <tr>
            <td><strong>Dictionary</strong></td>
            <td><code>{key: value}</code></td>
            <td>Key-value pairs, unique keys, $O(1)$ lookup time.</td>
            <td>Contraction mapping: <code>CONTRACTION_MAP = {"don't": "do not"}</code> in <code>src/data_preprocessing.py</code></td>
        </tr>
    </table>

    <h3>Line-by-Line Breakdown of Project Dictionary:</h3>
    <pre><code>CONTRACTION_MAP = {
    "don't": "do not",
    "wasn't": "was not",
    "can't": "cannot"
}</code></pre>
    <ul>
        <li><code>CONTRACTION_MAP</code> is a Python dictionary variable.</li>
        <li><code>"don't"</code> is the dictionary key (what we search for in raw reviews).</li>
        <li><code>"do not"</code> is the dictionary value (the expanded replacement that preserves the word "not").</li>
    </ul>

    <h2>3. Conditional Statements (if / elif / else)</h2>
    <p>Executes different blocks of code depending on whether a boolean expression is <code>True</code> or <code>False</code>.</p>
    <pre><code># Actual Code from src/predict.py:
sentiment = 'Positive' if pred_label == 1 else 'Negative'
confidence = pos_prob if pred_label == 1 else neg_prob</code></pre>
    <p><strong>Line-by-Line Explanation:</strong> This is a Python <em>ternary operator</em> (one-line if-else). If <code>pred_label</code> equals <code>1</code>, the variable <code>sentiment</code> becomes <code>'Positive'</code>; otherwise, it becomes <code>'Negative'</code>.</p>

    <h2>4. Loops (for and while) and List Comprehensions</h2>
    <p>A <code>for</code> loop iterates over items in a sequence (like a list or dictionary keys). A <em>list comprehension</em> is a concise, vectorized Python shorthand for creating lists.</p>
    <pre><code># Standard For Loop vs. List Comprehension in src/predict.py:
# Comprehension to extract top positive features:
pos_features = [
    {'term': self.feature_names[i], 'coefficient': round(float(self.coefficients[i]), 3)}
    for i in top_pos_idx
]</code></pre>
    <p><strong>Line-by-Line Explanation:</strong> Loops over every index <code>i</code> inside the array <code>top_pos_idx</code>, creates a dictionary with the term name and its rounded coefficient, and gathers all of them into a new list named <code>pos_features</code> in a single line.</p>

    <h2>5. Functions and Type Hints</h2>
    <p>A function is a reusable block of code that takes inputs (arguments), executes logic, and returns an output.</p>
    <pre><code># Actual Function from src/predict.py:
def predict_sentiment(review_text: str, models_dir: str = 'models') -> Dict[str, Any]:
    predictor = SentimentPredictor(models_dir=models_dir)
    return predictor.predict(review_text)</code></pre>
    <ul>
        <li><code>def predict_sentiment(...)</code>: Declares the function name.</li>
        <li><code>review_text: str</code>: Parameter type hint indicating input should be a string.</li>
        <li><code>models_dir: str = 'models'</code>: Default parameter; if caller doesn't specify a folder, it defaults to <code>'models'</code>.</li>
        <li><code>-> Dict[str, Any]</code>: Return type hint indicating the function sends back a dictionary containing mixed data types.</li>
    </ul>

    <h2>6. Object-Oriented Programming: Classes and Objects</h2>
    <p>A <strong>Class</strong> is a blueprint for creating objects (bundling data and functions together). An <strong>Object</strong> is a concrete instance of that blueprint.</p>
    <pre><code># Actual Class from src/predict.py:
class SentimentPredictor:
    def __init__(self, models_dir: str = 'models'):
        self.model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
        self.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
        
    def predict(self, review_text: str) -> Dict[str, Any]:
        # inference logic...</code></pre>
    <ul>
        <li><code>class SentimentPredictor:</code> Defines the class blueprint.</li>
        <li><code>def __init__(self, ...):</code> The constructor method. Automatically runs when an object is instantiated. It loads the model and vectorizer from disk into memory <em>only once</em>.</li>
        <li><code>self.model = ...</code>: Stores the loaded model inside the instance so all other methods can access it without reloading from disk.</li>
    </ul>

    <h2>7. Exception Handling (try / except)</h2>
    <p>Prevents an application from crashing when an unexpected runtime error occurs.</p>
    <pre><code>try:
    predictor = load_predictor()
    result = predictor.predict(user_input)
except Exception as e:
    st.error(f"Inference Engine Error: {e}")</code></pre>
    <p><strong>Explanation:</strong> If the model file is missing or corrupted, the <code>try</code> block catches the error, jumps to <code>except</code>, and displays a clean user-friendly alert in Streamlit rather than crashing the web server.</p>
</div>

<div class="page-break-before">
    <h1>PART 3 &mdash; NUMPY: HIGH-PERFORMANCE NUMERICAL COMPUTING</h1>

    <h2>1. What is NumPy and Why Does Machine Learning Need It?</h2>
    <p>NumPy (Numerical Python) is the foundational library for scientific computing in Python. Standard Python lists are slow because every element is a full Python object requiring runtime type-checking. NumPy introduces the <code>ndarray</code> (N-dimensional array), which stores homogeneous data in <strong>continuous memory blocks</strong> and executes mathematical operations in compiled C.</p>

    <h2>2. Array Dimensions and Shape</h2>
    <ul>
        <li><strong>1D Array (Vector):</strong> A single row of numbers (e.g., predicted probabilities: <code>[0.05, 0.95]</code>, shape: <code>(2,)</code>).</li>
        <li><strong>2D Array (Matrix):</strong> A grid of numbers with rows and columns (e.g., our TF-IDF feature matrix with 39,665 rows and 10,000 columns, shape: <code>(39665, 10000)</code>).</li>
    </ul>

    <h2>3. Key NumPy Functions Used in Our Project</h2>
    <table>
        <tr>
            <th>Function</th>
            <th>What It Does</th>
            <th>Project Application</th>
        </tr>
        <tr>
            <td><code>np.exp(z)</code></td>
            <td>Calculates natural exponential $e^z$.</td>
            <td>Computes the Sigmoid activation formula $\frac{1}{1 + e^{-z}}$ for probability estimation.</td>
        </tr>
        <tr>
            <td><code>np.argsort(array)</code></td>
            <td>Returns the indices that would sort an array.</td>
            <td>Finds the top 15 most positive and negative coefficients in <code>get_global_top_features()</code>.</td>
        </tr>
        <tr>
            <td><code>vec.nonzero()</code></td>
            <td>Returns row and column indices of non-zero sparse matrix values.</td>
            <td>Finds which specific words from the 10,000-vocabulary are present in the user's input review.</td>
        </tr>
    </table>
</div>

<div class="page-break-before">
    <h1>PART 4 &mdash; PANDAS: DATA WRANGLING & EXPLORATION</h1>

    <h2>1. What is Pandas?</h2>
    <p>Pandas is Python's premier data manipulation library built directly on top of NumPy. It introduces two essential structures:
    <ul>
        <li><strong>Series:</strong> A 1-dimensional labeled array (represents a single column).</li>
        <li><strong>DataFrame:</strong> A 2-dimensional tabular structure (rows and columns, exactly like an SQL table or Excel spreadsheet).</li>
    </ul>

    <h2>2. Critical Pandas Methods and Their Project Implementation</h2>
    <table>
        <tr>
            <th>Method</th>
            <th>What It Does</th>
            <th>Actual Project Code Line</th>
        </tr>
        <tr>
            <td><code>pd.read_csv()</code></td>
            <td>Loads a CSV file from disk into a DataFrame.</td>
            <td><code>df = pd.read_csv(raw_data_path)</code></td>
        </tr>
        <tr>
            <td><code>df.shape</code></td>
            <td>Returns a tuple of (rows, columns).</td>
            <td><code>print(f"Raw dataset shape: {df.shape}")</code> $\rightarrow (50000, 2)$</td>
        </tr>
        <tr>
            <td><code>df.isnull().sum()</code></td>
            <td>Counts missing (NaN) values per column.</td>
            <td><code>missing_count = df.isnull().sum().sum()</code></td>
        </tr>
        <tr>
            <td><code>df.duplicated()</code></td>
            <td>Identifies duplicate rows based on column subsets.</td>
            <td><code>duplicate_count = df.duplicated(subset=['review']).sum()</code></td>
        </tr>
        <tr>
            <td><code>df.drop_duplicates()</code></td>
            <td>Removes duplicate rows from DataFrame.</td>
            <td><code>df = df.drop_duplicates(subset=['review']).reset_index(drop=True)</code></td>
        </tr>
        <tr>
            <td><code>df['col'].apply()</code></td>
            <td>Applies a custom Python function to every row.</td>
            <td><code>df['cleaned_review'] = df['review'].apply(clean_text)</code></td>
        </tr>
        <tr>
            <td><code>df['col'].map()</code></td>
            <td>Substitutes each value with another value via dictionary.</td>
            <td><code>df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})</code></td>
        </tr>
        <tr>
            <td><code>df['col'].value_counts()</code></td>
            <td>Counts occurrences of each unique value.</td>
            <td><code>print(df['sentiment'].value_counts())</code></td>
        </tr>
    </table>

    <h3>Line-by-Line Breakdown of Project Data Ingestion:</h3>
    <pre><code># From src/data_preprocessing.py:
df = pd.read_csv('data/raw/imdb_reviews.csv')
duplicate_count = df.duplicated(subset=['review']).sum()
if duplicate_count > 0:
    df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})</code></pre>
    <ul>
        <li><code>df.duplicated(subset=['review']).sum()</code> checks the <code>review</code> text column. It finds 418 reviews that are verbatim duplicates.</li>
        <li><code>df.drop_duplicates(...)</code> discards those 418 rows so duplicate text cannot leak across training and test splits.</li>
        <li><code>.reset_index(drop=True)</code> rebuilds the sequential row index from 0 to 49,581 without holes.</li>
        <li><code>.map({'positive': 1, 'negative': 0})</code> transforms string categories into numeric binary ground truth for Scikit-Learn.</li>
    </ul>

    <div class="checkpoint-box">
        <h3>🎯 CHECK YOUR UNDERSTANDING &mdash; CHAPTERS 2, 3 & 4</h3>
        <ol>
            <li><strong>Q:</strong> Why do we use a dictionary for our contraction mapping instead of a list?</li>
            <li><strong>Q:</strong> Why must we convert target strings <code>"positive"</code> and <code>"negative"</code> to <code>1</code> and <code>0</code>?</li>
            <li><strong>Q:</strong> What is the purpose of calling <code>.reset_index(drop=True)</code> after dropping duplicates?</li>
        </ol>
        <div class="answers-toggle">
            <strong>Answers:</strong>
            <p>1. Dictionaries provide $O(1)$ constant-time lookup for matching contractions, whereas lists require $O(N)$ scanning.<br>
            2. Machine learning optimization algorithms rely on mathematical loss functions (e.g. log loss), which require numeric values.<br>
            3. Dropping rows leaves missing gaps in the index (e.g., 0, 1, 3, 7). Resetting the index ensures a continuous sequence from 0 to $N-1$.</p>
        </div>
    </div>
</div>
"""
