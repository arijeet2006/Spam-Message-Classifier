# 📱 Intelligent SMS Spam Classifier Suite

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Natural Language Processing](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning and Natural Language Processing (NLP) infrastructure engineered to analyze, process, and classify mobile text messages. Built from scratch on raw SMS communication data, this system translates chaotic text arrays into pristine numerical vectors, runs probabilistic statistical classification, and surfaces real-time predictions through an elegant, interactive Streamlit production dashboard.

---

## 📌 Project Overview & Intent

In modern telecommunications, short-message spam presents a continuous security and optimization problem. This project serves as an end-to-end solution mapping out:
1. **Data Sanitization & Structuring:** Turning raw, inconsistent corporate/personal SMS logs into balanced tabular assets.
2. **Feature Engineering & Behavioral Analytics:** Isolating distinct linguistic profiles (character bounds, word clusters, and sentence limits) that distinguish safe communication channels from adversarial tracking hooks.
3. **Probabilistic Class Extraction:** Minimizing False-Positive events using tuned Bayesian classification models to ensure legitimate messages are never accidentally filtered.

---

## 🚀 Key Framework Architecture

### 1. Robust Natural Language Processing Pipeline
Raw text arrays vary widely in spelling, grammar, and syntax. To capture the actual predictive semantics of each text, user inputs undergo an explicit multi-stage pipeline using the **Natural Language Toolkit (NLTK)**:
* **Lowercasing:** Converts all inputs to lowercase to normalize tokens (e.g., matching `SPAM`, `Spam`, and `spam` identically).
* **Word Tokenization:** Parses string structures into concrete word units and punctuation boundaries using `nltk.word_tokenize`.
* **Alphanumeric Scrubbing:** Scrubs text by discarding mathematical syntax variables, specialized structural punctuation icons, and stray characters.
* **Stopword Purging:** Dynamically drops recurring grammatical placeholders (e.g., "our", "have", "you") matching the standard NLTK English stopword array.
* **Porter Stemming Translation:** Normalizes structural suffixes down to core morphological roots via a `PorterStemmer` instance (e.g., transforming `winning`, `wins`, and `winner` to `win`).

### 2. Behavioral Exploratory Data Analysis (EDA) Insights
During model engineering inside `1.ipynb`, advanced custom structural features were calculated to discover text density correlations:
* **Metrics Injected:** `num_characters`, `num_words`, and `num_sentences`.
* **The Structural Contrast Rule:** Statistical distribution mapping proves that safe messages (**Ham**) follow localized, compact spatial distributions (typically under 50 characters). Adversarial structures (**Spam**) heavily scale toward higher character and word density limits, packing sentences with sensory text hooks like "FREE", "CLAIM", and "CASH WINNER".

---

## 🛠️ Mathematical Model Selection & Benchmarks

The project shifts from experimental code to absolute production deployment by training and evaluating three distinct structural branches of Naive Bayes Classifiers using **Bag of Words (CountVectorizer)** feature extractions:

| Engine | Global Accuracy | Precision Score | Validation Performance Profile |
| :--- | :--- | :--- | :--- |
| **Bernoulli Naive Bayes (`BernoulliNB`)** | **97.00%** | **97.34%** | 🏆 **Production Winner:** Unmatched precision profile. |
| **Multinomial Naive Bayes (`MultinomialNB`)** | 96.42% | 83.44% | **Sub-optimal:** Prone to high False-Positive rates. |
| **Gaussian Naive Bayes (`GaussianNB`)** | 88.00% | 53.15% | **Failed Standard:** Struggles with sparse matrix arrays. |

### 🔍 Why Bernoulli Naive Bayes Outperforms the Rest
While standard text mining engines default to Multinomial tracking, the `BernoulliNB` architecture thrives here. Because SMS structures are incredibly brief, tracking **word occurrences** (presence vs. absence represented by boolean flags) maps far more cleanly to spam indicators than **word frequency counts**. 

Most importantly, it maximizes **Precision**. In production email/SMS filters, marking a critical, real-time personal message as spam (a False Positive) causes immediate system friction. Our Bernoulli selection ensures that when a message is tagged as **🚨 SPAM**, the accuracy metric is almost definitive.

---

## 📁 Repository Directory Maps

```text
├── 1.ipynb            # Experimental lab space (Data cleaning, NLTK analysis, EDA plots, Model Benchmarks)
├── app.py             # Active production web script orchestrating Streamlit frontend and caching mechanics
├── bnb_model.pkl      # Serialized Production Bernoulli Naive Bayes Classifier model 
├── vectorizer.pkl     # Serialized Production CountVectorizer sparse-matrix text transformer
└── README.md          # Project documentation suite
